#!/usr/bin/env python3
"""
T047 — GPU-accelerated autoregressive KRR training via PyTorch CUDA.

This replaces the pure-NumPy CPU pipeline with batched GPU operations:
  - Token embedding lookup: GPU tensor indexing
  - Multi-head causal attention: torch.bmm with causal mask
  - RFF projection: batched cos(F @ omega + bias)
  - ZtZ/ZtY accumulation: batched matmul on GPU
  - CG solve: GPU matrix-matrix products

Expected speedup: 12-25× on A6000 vs. Apple M-class CPU.

Usage:
  python train_gpu.py --corpus corpus_large.txt --tokenizer bpe_large.json \\
                      --D 12288 --layers 2 --heads 4 --ctx 64 --cg-maxiter 1000

  # Quick test
  python train_gpu.py --D 4096 --cg-maxiter 100
"""
import os, sys, time, math, argparse, json, pickle
import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser(description='GPU-accelerated AR-KRR training')
    p.add_argument('--corpus', default='data/autoregressive/corpus_large.txt')
    p.add_argument('--tokenizer', default='data/autoregressive/bpe_large.json')
    p.add_argument('--output', default='data/autoregressive/model_gpu.pkl')
    p.add_argument('--D', type=int, default=6144, help='RFF dimension')
    p.add_argument('--emb-dim', type=int, default=64, help='Token embedding dim')
    p.add_argument('--dk', type=int, default=32, help='Attention key dim per head')
    p.add_argument('--dv', type=int, default=64, help='Attention value dim per head')
    p.add_argument('--heads', type=int, default=4, help='Number of attention heads')
    p.add_argument('--layers', type=int, default=2, help='Number of attention layers')
    p.add_argument('--dff', type=int, default=256, help='FFN hidden dim')
    p.add_argument('--ctx', type=int, default=64, help='Context window length')
    p.add_argument('--sigma', type=float, default=2.0, help='RFF kernel bandwidth')
    p.add_argument('--lam', type=float, default=1e-5, help='Ridge regularization')
    p.add_argument('--cg-maxiter', type=int, default=800, help='CG max iterations')
    p.add_argument('--cg-tol', type=float, default=1e-5, help='CG tolerance')
    p.add_argument('--batch-size', type=int, default=4096, help='Accumulation batch size')
    p.add_argument('--val-frac', type=float, default=0.05, help='Validation fraction')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--w2v-epochs', type=int, default=15)
    p.add_argument('--results-csv', default='benchmarks/ar_krr_experiments.csv')
    return p.parse_args()


def sinusoidal_pe(max_len, d_emb, device):
    pe = torch.zeros(max_len, d_emb, device=device)
    pos = torch.arange(max_len, device=device).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_emb, 2, device=device).float() * (-math.log(10000.0) / d_emb))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def block_cg_gpu(A, B, tol=1e-5, max_iter=800, verbose=True):
    """Block-PCG with diagonal preconditioner on GPU (Float32)."""
    D, V = B.shape
    diag_A = torch.diagonal(A).clone()
    diag_A[diag_A.abs() < 1e-30] = 1.0
    inv_diag = 1.0 / diag_A

    X = torch.zeros_like(B)
    R = B.clone()
    Z = R * inv_diag.unsqueeze(1)
    P = Z.clone()
    rz_old = (R * Z).sum(dim=0)
    init_res = R.norm(dim=0).max().item()

    for it in range(max_iter):
        AP = A @ P
        pAp = (P * AP).sum(dim=0)
        pAp_safe = torch.where(pAp.abs() > 1e-30, pAp, torch.ones_like(pAp))
        alpha = rz_old / pAp_safe
        X = X + P * alpha.unsqueeze(0)
        R = R - AP * alpha.unsqueeze(0)
        Z = R * inv_diag.unsqueeze(1)
        rz_new = (R * Z).sum(dim=0)
        rel_res = R.norm(dim=0).max().item() / init_res

        if verbose and (it % 50 == 0 or it < 5):
            print(f"    CG iter {it+1}: rel_res = {rel_res:.2e}")

        if rel_res < tol:
            if verbose:
                print(f"    CG converged: {it+1} iters, rel_res={rel_res:.2e}")
            return X, {'iterations': it+1, 'residual': rel_res, 'converged': True}

        rz_old_safe = torch.where(rz_old.abs() > 1e-30, rz_old, torch.ones_like(rz_old))
        beta = rz_new / rz_old_safe
        P = Z + P * beta.unsqueeze(0)
        rz_old = rz_new

    if verbose:
        print(f"    CG max_iter={max_iter}, rel_res={rel_res:.2e}")
    return X, {'iterations': max_iter, 'residual': rel_res, 'converged': False}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)

    print(f"Device: {dev}")
    if dev.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # ---- Tokenizer + corpus ----
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer)
    V = tokenizer.get_vocab_size()
    with open(args.corpus) as f:
        corpus_text = f.read()
    all_ids = np.array(tokenizer.encode(corpus_text).ids, dtype=np.int64)
    n_val = int(len(all_ids) * args.val_frac)
    train_ids = all_ids[:len(all_ids) - n_val]
    val_ids = all_ids[len(all_ids) - n_val:]
    N = len(train_ids) - args.ctx
    print(f"Corpus: {len(all_ids):,} tokens, V={V}, Train N={N:,}, Val={len(val_ids):,}")

    # ---- Config ----
    CTX, EMB, DK, DV, H, L, DFF = args.ctx, args.emb_dim, args.dk, args.dv, args.heads, args.layers, args.dff
    FEAT = 2 * EMB  # final layer state + last-token emb
    D = args.D
    print(f"Architecture: L={L}, H={H}, dk={DK}, dv={DV}, dff={DFF}, CTX={CTX}, EMB={EMB}")
    print(f"FEAT={FEAT}, D={D}, V={V}")
    print(f"W size: {D * V * 4 / 1024**2:.0f} MB (Float32)")

    # ---- Word2Vec embeddings ----
    print("\nWord2Vec on BPE stream...")
    from gensim.models import Word2Vec
    str_toks = [str(i) for i in train_ids]
    sents = [str_toks[i:i+100] for i in range(0, len(str_toks), 100)]
    w2v = Word2Vec(sents, vector_size=EMB, window=8, min_count=1,
                   workers=4, sg=0, epochs=args.w2v_epochs, seed=args.seed)
    emb_np = np.zeros((V, EMB), dtype=np.float32)
    for tid in range(V):
        k = str(tid)
        if k in w2v.wv:
            emb_np[tid] = w2v.wv[k]
        else:
            emb_np[tid] = np.random.randn(EMB).astype(np.float32) * 0.01
    emb_np /= (np.linalg.norm(emb_np, axis=1, keepdims=True) + 1e-8)
    emb_t = torch.from_numpy(emb_np).to(dev)
    del w2v, emb_np

    # ---- Positional encoding ----
    pe_t = sinusoidal_pe(CTX, EMB, dev)

    # ---- Random attention parameters per layer (all on GPU) ----
    si = 1.0 / math.sqrt(EMB)
    layers_params = []
    for ell in range(L):
        g = torch.Generator(device='cpu').manual_seed(args.seed + 10*ell)
        lp = {
            'W_Q': torch.randn(EMB, H*DK, generator=g).to(dev) * si,
            'W_K': torch.randn(EMB, H*DK, generator=g).to(dev) * si,
            'W_V': torch.randn(EMB, H*DV, generator=g).to(dev) * si,
            'W_O': torch.randn(H*DV, EMB, generator=g).to(dev) / math.sqrt(H*DV),
            'W_in': torch.randn(EMB, DFF, generator=g).to(dev) / math.sqrt(EMB),
            'W_out': torch.randn(DFF, EMB, generator=g).to(dev) / math.sqrt(DFF),
        }
        layers_params.append(lp)

    # ---- RFF projection ----
    g_rff = torch.Generator(device='cpu').manual_seed(args.seed + 1000)
    omega_t = (torch.randn(FEAT, D, generator=g_rff) / args.sigma).to(dev)
    bias_t = (torch.rand(D, generator=g_rff) * 2 * math.pi).to(dev)
    rff_scale = math.sqrt(2.0 / D)

    # ---- Causal attention mask (upper triangular = masked) ----
    causal_mask = torch.triu(torch.ones(CTX, CTX, device=dev), diagonal=1).bool()

    # ---- Batched encoder ----
    def encode_batch(batch_ctx_ids):
        """
        batch_ctx_ids: (B, CTX) int64 tensor of token IDs (right-aligned, padded with 0)
        Returns: (B, FEAT) float32 feature vectors
        """
        B = batch_ctx_ids.shape[0]
        # Embeddings + PE
        E = emb_t[batch_ctx_ids]                    # (B, CTX, EMB)
        X = E + pe_t.unsqueeze(0)                   # (B, CTX, EMB) broadcast PE

        # Multi-layer attention
        for lp in layers_params:
            Q = (X @ lp['W_Q']).view(B, CTX, H, DK)  # (B, CTX, H, DK)
            K = (X @ lp['W_K']).view(B, CTX, H, DK)
            Vv = (X @ lp['W_V']).view(B, CTX, H, DV)

            # Attention scores: (B, H, CTX, CTX)
            Q_ = Q.permute(0, 2, 1, 3)   # (B, H, CTX, DK)
            K_ = K.permute(0, 2, 3, 1)   # (B, H, DK, CTX)
            scores = (Q_ @ K_) / math.sqrt(DK)
            scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            weights = torch.softmax(scores, dim=-1)  # (B, H, CTX, CTX)

            Vv_ = Vv.permute(0, 2, 1, 3)  # (B, H, CTX, DV)
            attn_out = (weights @ Vv_).permute(0, 2, 1, 3).reshape(B, CTX, H*DV)  # (B, CTX, H*DV)
            attn_out = attn_out @ lp['W_O']  # (B, CTX, EMB)

            # Residual + LayerNorm
            X1 = X + attn_out
            X1 = F.layer_norm(X1, [EMB])

            # FFN
            h = torch.relu(X1 @ lp['W_in'])
            ff = h @ lp['W_out']
            X = F.layer_norm(X1 + ff, [EMB])

        # Feature = last position's state + last token's embedding
        last_state = X[:, -1, :]     # (B, EMB)
        last_emb = E[:, -1, :]      # (B, EMB)
        return torch.cat([last_state, last_emb], dim=1)  # (B, FEAT)

    # ---- Build context windows ----
    print(f"\nBuilding context windows (N={N:,})...")
    t0 = time.time()
    # Pre-extract all (CTX,) windows: for position t, context is train_ids[t:t+CTX]
    # Target is train_ids[t+CTX]
    train_ids_t = torch.from_numpy(train_ids).long()
    # Create a (N, CTX) index matrix efficiently
    indices = torch.arange(CTX).unsqueeze(0) + torch.arange(N).unsqueeze(1)  # (N, CTX)
    all_ctx = train_ids_t[indices]  # (N, CTX)
    all_targets = train_ids_t[args.ctx:args.ctx+N]  # (N,)
    print(f"  Context windows built in {time.time()-t0:.1f}s")
    print(f"  all_ctx: {all_ctx.shape}, all_targets: {all_targets.shape}")

    # ---- Streaming ZtZ + ZtY accumulation on GPU ----
    print(f"\nAccumulating ZtZ + ZtY (batch_size={args.batch_size})...")
    t_accum_start = time.time()

    ZtZ = torch.zeros(D, D, dtype=torch.float64, device=dev)
    ZtY = torch.zeros(D, V, dtype=torch.float64, device=dev)
    BS = args.batch_size

    for start in range(0, N, BS):
        end = min(start + BS, N)
        batch_ctx = all_ctx[start:end].to(dev)     # (bs, CTX)
        batch_tgt = all_targets[start:end]          # (bs,) on CPU

        with torch.no_grad():
            feat = encode_batch(batch_ctx)           # (bs, FEAT)
            z = rff_scale * torch.cos(feat @ omega_t + bias_t.unsqueeze(0))  # (bs, D)
            z64 = z.double()
            ZtZ += z64.T @ z64                       # (D, D)
            # Scatter-add into ZtY
            for j in range(end - start):
                ZtY[:, batch_tgt[j].item()] += z64[j]

        if start % (BS * 10) == 0:
            elapsed = time.time() - t_accum_start
            pct = 100 * start / N
            print(f"  {start:>9}/{N:,} ({pct:5.1f}%)  {elapsed:6.1f}s")

    ZtZ += args.lam * torch.eye(D, dtype=torch.float64, device=dev)
    t_accum = time.time() - t_accum_start
    print(f"Accumulation: {t_accum:.1f}s")

    # ---- CG Solve on GPU ----
    print(f"\nSolving via Block-PCG on {dev} (maxiter={args.cg_maxiter})...")
    t_solve_start = time.time()
    A_f32 = ZtZ.float()
    B_f32 = ZtY.float()
    del ZtZ, ZtY
    torch.cuda.empty_cache() if dev.type == 'cuda' else None

    W_gpu, cg_info = block_cg_gpu(A_f32, B_f32, tol=args.cg_tol, max_iter=args.cg_maxiter)
    t_solve = time.time() - t_solve_start
    print(f"Solve: {t_solve:.1f}s, {cg_info}")
    del A_f32, B_f32

    # ---- Evaluation ----
    print("\nEvaluation...")
    def eval_topk(ids_np, n_eval=5000, label=''):
        ids_t = torch.from_numpy(ids_np).long()
        N_e = len(ids_t) - CTX
        if N_e <= 0: return 0.0, 0.0
        sample = np.random.choice(N_e, min(n_eval, N_e), replace=False)
        top1 = top5 = 0
        for i in sample:
            ctx = ids_t[i:i+CTX].unsqueeze(0).to(dev)
            with torch.no_grad():
                feat = encode_batch(ctx)
                z = rff_scale * torch.cos(feat @ omega_t + bias_t.unsqueeze(0))
                scores = (z @ W_gpu).squeeze(0)
            pred = scores.argmax().item()
            tgt = ids_t[i+CTX].item()
            if pred == tgt: top1 += 1
            _, top5_idx = scores.topk(5)
            if tgt in top5_idx.cpu().numpy(): top5 += 1
        t1 = top1/len(sample); t5 = top5/len(sample)
        print(f"  {label} Top-1: {t1*100:.1f}%, Top-5: {t5*100:.1f}%")
        return t1, t5

    train_t1, train_t5 = eval_topk(train_ids, 5000, 'Train')
    val_t1, val_t5 = eval_topk(val_ids, 5000, 'Val  ')

    # ---- Save model (CPU tensors for portability) ----
    print(f"\nSaving to {args.output}...")
    model = {
        'W': W_gpu.cpu().numpy(),
        'omega': omega_t.cpu().numpy(),
        'bias': bias_t.cpu().numpy(),
        'emb': emb_t.cpu().numpy(),
        'pe': pe_t.cpu().numpy(),
        'layers': [{k: v.cpu().numpy() for k, v in lp.items()} for lp in layers_params],
        'config': {
            'CTX': CTX, 'EMB_DIM': EMB, 'D_K': DK, 'D_V': DV,
            'N_HEADS': H, 'N_LAYERS': L, 'D_FF': DFF,
            'FEAT': FEAT, 'D': D, 'SIGMA': args.sigma,
            'LAMBDA': args.lam, 'V': V,
        },
        'stats': {
            'train_top1': train_t1, 'train_top5': train_t5,
            'val_top1': val_t1, 'val_top5': val_t5,
            't_accum': t_accum, 't_solve': t_solve,
            'cg_info': cg_info, 'device': str(dev),
            'n_train': N, 'n_tokens': len(all_ids),
        },
    }
    with open(args.output, 'wb') as f:
        pickle.dump(model, f)

    # ---- Summary ----
    total = t_accum + t_solve
    print(f"\n{'='*60}")
    print(f"DONE — {args.device}")
    print(f"  Corpus:  {len(all_ids):,} tokens, V={V}")
    print(f"  Arch:    L={L} H={H} dk={DK} dv={DV} dff={DFF} CTX={CTX}")
    print(f"  D:       {D}")
    print(f"  Accum:   {t_accum:.1f}s")
    print(f"  Solve:   {t_solve:.1f}s ({cg_info['iterations']} iters)")
    print(f"  Total:   {total:.1f}s")
    print(f"  Train:   Top-1 {train_t1*100:.1f}%, Top-5 {train_t5*100:.1f}%")
    print(f"  Val:     Top-1 {val_t1*100:.1f}%, Top-5 {val_t5*100:.1f}%")
    print(f"  Gap:     {(train_t1-val_t1)*100:.1f}pp")
    print(f"{'='*60}")

    # ---- Append to CSV ----
    csv_line = (f"gpu_{L}L_D{D},{args.device},{time.strftime('%Y-%m-%d')},"
                f"\"{L}L {H}H D={D} CTX={CTX} {len(all_ids)/1e6:.1f}M tokens\","
                f"{train_t1:.4f},{train_t5:.4f},{val_t1:.4f},{val_t5:.4f},,"
                f"{t_accum:.0f},{t_solve:.0f},{cg_info['iterations']},"
                f"{'true' if cg_info['converged'] else 'false'},,"
                f"{N},{V},\"{json.dumps(model['config'])}\"")
    try:
        with open(args.results_csv, 'a') as f:
            f.write(csv_line + '\n')
        print(f"Appended to {args.results_csv}")
    except:
        print(f"CSV line (append manually):\n{csv_line}")


if __name__ == '__main__':
    main()

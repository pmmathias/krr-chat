#!/usr/bin/env python3
"""
T-Day1: GPU training with three improvements over train_gpu.py:

  J) Orthogonal Random Features — QR-decompose all random matrices
  B) ROCKET Random Convolutions — 5000 random 1D conv filters as extra features
  C) N-gram Shallow Fusion — applied at inference time (separate script)

Usage:
  # Baseline (same as train_gpu.py but with orthogonal RF):
  python train_gpu_v2.py --D 12288 --layers 1 --corpus corpus_15m.txt

  # With ROCKET convolutions:
  python train_gpu_v2.py --D 12288 --layers 1 --rocket 5000

  # Comparison run (no rocket, no ortho — same as v1):
  python train_gpu_v2.py --D 12288 --layers 1 --no-ortho --rocket 0
"""
import os, sys, time, math, argparse, json, pickle
import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser(description='GPU AR-KRR v2 (Ortho RF + ROCKET)')
    p.add_argument('--corpus', default='data/autoregressive/corpus_15m.txt')
    p.add_argument('--tokenizer', default='data/autoregressive/bpe_50m.json')
    p.add_argument('--output', default='data/autoregressive/model_gpu_v2.pkl')
    p.add_argument('--D', type=int, default=12288)
    p.add_argument('--emb-dim', type=int, default=64)
    p.add_argument('--dk', type=int, default=32)
    p.add_argument('--dv', type=int, default=64)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--layers', type=int, default=1)
    p.add_argument('--dff', type=int, default=256)
    p.add_argument('--ctx', type=int, default=64)
    p.add_argument('--sigma', type=float, default=2.0)
    p.add_argument('--lam', type=float, default=1e-5)
    p.add_argument('--cg-maxiter', type=int, default=500)
    p.add_argument('--cg-tol', type=float, default=1e-5)
    p.add_argument('--batch-size', type=int, default=4096)
    p.add_argument('--val-frac', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--w2v-epochs', type=int, default=15)
    # New v2 features:
    p.add_argument('--no-ortho', action='store_true', help='Disable orthogonal RF (use i.i.d. random)')
    p.add_argument('--rocket', type=int, default=5000, help='Number of ROCKET random conv kernels (0=disable)')
    p.add_argument('--rocket-lengths', default='3,5,7,9', help='Conv kernel lengths (comma-separated)')
    return p.parse_args()


def orthogonalize(M):
    """QR-orthogonalize a random matrix. Provably reduces approximation error
    (Yu et al. 2016, Choromanski et al. 2021 FAVOR+).
    Handles both tall (rows>=cols) and fat (rows<cols) matrices."""
    rows, cols = M.shape
    if rows >= cols:
        Q, R = torch.linalg.qr(M)
        d = torch.diag(R)
        return Q * d.sign().unsqueeze(0)
    else:
        # Fat matrix: orthogonalize rows via QR of transpose
        Q, R = torch.linalg.qr(M.T)
        d = torch.diag(R)
        return (Q * d.sign().unsqueeze(0)).T


def sinusoidal_pe(max_len, d_emb, device):
    pe = torch.zeros(max_len, d_emb, device=device)
    pos = torch.arange(max_len, device=device).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_emb, 2, device=device).float() * (-math.log(10000.0) / d_emb))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def generate_rocket_kernels(n_kernels, kernel_lengths, input_dim, seed=42, device='cpu'):
    """Generate ROCKET-style random 1D convolutional kernels.

    Each kernel: random weights (normal), random bias (uniform),
    random dilation (exponential), random padding (valid or same).
    Returns a list of (weight, bias, dilation, padding) tuples.

    Reference: Dempster et al. 2020, "ROCKET: Exceptionally fast and accurate
    time series classification using random convolutional kernels"
    """
    rng = np.random.default_rng(seed)
    lengths = [int(l) for l in kernel_lengths.split(',')]
    kernels = []
    for i in range(n_kernels):
        klen = lengths[i % len(lengths)]
        # Random weights (normalized)
        w = rng.standard_normal(klen).astype(np.float32)
        w -= w.mean()  # zero-mean (important for ROCKET)
        # Random bias
        b = np.float32(rng.uniform(-1, 1))
        # Random dilation (exponential distribution, capped)
        max_dilation = max(1, (input_dim - 1) // (klen - 1)) if klen > 1 else 1
        dilation = int(2 ** rng.uniform(0, math.log2(max_dilation + 1)))
        dilation = min(dilation, max_dilation)
        # Random padding
        padding = rng.choice([0, (klen - 1) * dilation // 2])
        kernels.append((
            torch.tensor(w, device=device).reshape(1, 1, klen),
            torch.tensor([b], device=device),
            dilation,
            padding,
        ))
    return kernels


def apply_rocket(x_seq, rocket_kernels):
    """Apply ROCKET kernels to a batch of sequences.

    x_seq: (B, seq_len, emb_dim) — we apply convolutions along seq_len
           for each embedding dimension independently, then aggregate.
    Returns: (B, 2 * n_kernels) — max-pool + proportion-of-positives per kernel.
    """
    B, T, E = x_seq.shape
    # Average embeddings to get 1D signal per sample: (B, T)
    x_1d = x_seq.mean(dim=-1).unsqueeze(1)  # (B, 1, T)

    features = []
    for w, b, dilation, padding in rocket_kernels:
        # Conv1d: (B, 1, T) -> (B, 1, T')
        out = F.conv1d(x_1d, w, bias=b, dilation=dilation, padding=padding)
        # Two features per kernel: global max-pool + proportion of positives (PPV)
        max_val = out.max(dim=-1).values.squeeze(1)  # (B,)
        ppv = (out > 0).float().mean(dim=-1).squeeze(1)  # (B,)
        features.append(max_val)
        features.append(ppv)

    return torch.stack(features, dim=1)  # (B, 2*n_kernels)


def block_cg_gpu(A, B, tol=1e-5, max_iter=800, verbose=True):
    """Block-PCG with diagonal preconditioner on GPU."""
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

    print(f"\nFeatures enabled:")
    print(f"  Orthogonal RF: {'YES' if not args.no_ortho else 'NO (i.i.d. random)'}")
    print(f"  ROCKET convolutions: {args.rocket} kernels" if args.rocket > 0 else "  ROCKET: disabled")

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
    print(f"\nCorpus: {len(all_ids):,} tokens, V={V}, Train N={N:,}")

    CTX, EMB, DK, DV, H, L, DFF = args.ctx, args.emb_dim, args.dk, args.dv, args.heads, args.layers, args.dff
    D = args.D

    # FEAT = attention output (2*EMB) + optionally ROCKET features (2*n_rocket)
    FEAT_ATTN = 2 * EMB
    FEAT_ROCKET = 2 * args.rocket if args.rocket > 0 else 0
    FEAT = FEAT_ATTN + FEAT_ROCKET
    print(f"Architecture: L={L}, H={H}, dk={DK}, dv={DV}, CTX={CTX}")
    print(f"FEAT = {FEAT_ATTN} (attention) + {FEAT_ROCKET} (ROCKET) = {FEAT}")
    print(f"D={D}, V={V}")

    # ---- Word2Vec embeddings ----
    print("\nWord2Vec...")
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

    pe_t = sinusoidal_pe(CTX, EMB, dev)

    # ---- Attention parameters (with orthogonalization if enabled) ----
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
        if not args.no_ortho:
            # J: Orthogonalize Q/K/V projections (FAVOR+, Yu et al. 2016)
            for key in ['W_Q', 'W_K', 'W_V']:
                lp[key] = orthogonalize(lp[key]) * si
        layers_params.append(lp)

    # ---- RFF projection (also orthogonalized) ----
    g_rff = torch.Generator(device='cpu').manual_seed(args.seed + 1000)
    omega_t = (torch.randn(FEAT, D, generator=g_rff) / args.sigma).to(dev)
    if not args.no_ortho and FEAT <= D:
        omega_t = orthogonalize(omega_t.T).T / args.sigma
    bias_t = (torch.rand(D, generator=g_rff) * 2 * math.pi).to(dev)
    rff_scale = math.sqrt(2.0 / D)

    # ---- ROCKET kernels (if enabled) ----
    rocket_kernels = []
    if args.rocket > 0:
        print(f"\nGenerating {args.rocket} ROCKET kernels (lengths: {args.rocket_lengths})...")
        rocket_kernels = generate_rocket_kernels(
            args.rocket, args.rocket_lengths, CTX, seed=args.seed + 2000, device=dev)
        print(f"  Generated {len(rocket_kernels)} kernels")

    causal_mask = torch.triu(torch.ones(CTX, CTX, device=dev), diagonal=1).bool()

    # ---- Batched encoder ----
    def encode_batch(batch_ctx_ids):
        B = batch_ctx_ids.shape[0]
        E = emb_t[batch_ctx_ids]
        X = E + pe_t.unsqueeze(0)

        for lp in layers_params:
            Q_ = (X @ lp['W_Q']).view(B, CTX, H, DK).permute(0, 2, 1, 3)
            K_ = (X @ lp['W_K']).view(B, CTX, H, DK).permute(0, 2, 3, 1)
            Vv_ = (X @ lp['W_V']).view(B, CTX, H, DV).permute(0, 2, 1, 3)
            scores = (Q_ @ K_) / math.sqrt(DK)
            scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            weights = torch.softmax(scores, dim=-1)
            attn_out = (weights @ Vv_).permute(0, 2, 1, 3).reshape(B, CTX, H*DV) @ lp['W_O']
            X1 = F.layer_norm(X + attn_out, [EMB])
            h = torch.relu(X1 @ lp['W_in'])
            X = F.layer_norm(X1 + h @ lp['W_out'], [EMB])

        # Attention features: last position state + last token embedding
        feat_attn = torch.cat([X[:, -1, :], E[:, -1, :]], dim=1)  # (B, 2*EMB)

        if rocket_kernels:
            feat_rocket = apply_rocket(E, rocket_kernels)  # (B, 2*n_rocket)
            return torch.cat([feat_attn, feat_rocket], dim=1)  # (B, FEAT)
        return feat_attn

    # ---- Build context windows ----
    print(f"\nBuilding context windows (N={N:,})...")
    t0 = time.time()
    train_ids_t = torch.from_numpy(train_ids).long()
    indices = torch.arange(CTX).unsqueeze(0) + torch.arange(N).unsqueeze(1)
    all_ctx = train_ids_t[indices]
    all_targets = train_ids_t[args.ctx:args.ctx+N]
    print(f"  Built in {time.time()-t0:.1f}s")

    # ---- Streaming ZtZ + ZtY ----
    print(f"\nAccumulating ZtZ + ZtY (batch={args.batch_size}, FEAT={FEAT}, D={D})...")
    t_accum_start = time.time()
    ZtZ = torch.zeros(D, D, dtype=torch.float64, device=dev)
    ZtY = torch.zeros(D, V, dtype=torch.float64, device=dev)
    BS = args.batch_size

    for start in range(0, N, BS):
        end = min(start + BS, N)
        batch_ctx = all_ctx[start:end].to(dev)
        batch_tgt = all_targets[start:end]
        with torch.no_grad():
            feat = encode_batch(batch_ctx)
            z = rff_scale * torch.cos(feat @ omega_t + bias_t.unsqueeze(0))
            z64 = z.double()
            ZtZ += z64.T @ z64
            for j in range(end - start):
                ZtY[:, batch_tgt[j].item()] += z64[j]
        if start % (BS * 10) == 0:
            elapsed = time.time() - t_accum_start
            pct = 100 * start / N
            print(f"  {start:>9}/{N:,} ({pct:5.1f}%)  {elapsed:6.1f}s")

    ZtZ += args.lam * torch.eye(D, dtype=torch.float64, device=dev)
    t_accum = time.time() - t_accum_start
    print(f"Accumulation: {t_accum:.1f}s")

    # ---- CG Solve ----
    print(f"\nBlock-PCG (maxiter={args.cg_maxiter})...")
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

    # ---- Save ----
    print(f"\nSaving to {args.output}...")
    model = {
        'W': W_gpu.cpu().numpy(),
        'omega': omega_t.cpu().numpy(),
        'bias': bias_t.cpu().numpy(),
        'emb': emb_t.cpu().numpy(),
        'pe': pe_t.cpu().numpy(),
        'layers': [{k: v.cpu().numpy() for k, v in lp.items()} for lp in layers_params],
        'rocket_kernels': [(w.cpu().numpy(), b.cpu().numpy(), d, p)
                           for w, b, d, p in rocket_kernels] if rocket_kernels else [],
        'config': {
            'CTX': CTX, 'EMB_DIM': EMB, 'D_K': DK, 'D_V': DV,
            'N_HEADS': H, 'N_LAYERS': L, 'D_FF': DFF,
            'FEAT': FEAT, 'FEAT_ATTN': FEAT_ATTN, 'FEAT_ROCKET': FEAT_ROCKET,
            'D': D, 'SIGMA': args.sigma, 'LAMBDA': args.lam, 'V': V,
            'ortho': not args.no_ortho, 'n_rocket': args.rocket,
        },
        'stats': {
            'train_top1': train_t1, 'train_top5': train_t5,
            'val_top1': val_t1, 'val_top5': val_t5,
            't_accum': t_accum, 't_solve': t_solve,
            'cg_info': cg_info, 'device': str(dev), 'n_train': N,
        },
    }
    with open(args.output, 'wb') as f:
        pickle.dump(model, f)

    total = t_accum + t_solve
    print(f"\n{'='*60}")
    print(f"DONE — {args.device}")
    print(f"  Corpus:   {len(all_ids):,} tokens, V={V}")
    print(f"  Arch:     L={L} H={H} CTX={CTX}")
    print(f"  Features: FEAT={FEAT} (attn={FEAT_ATTN} + rocket={FEAT_ROCKET})")
    print(f"  Ortho RF: {'YES' if not args.no_ortho else 'NO'}")
    print(f"  D:        {D}")
    print(f"  Accum:    {t_accum:.1f}s")
    print(f"  Solve:    {t_solve:.1f}s ({cg_info['iterations']} iters)")
    print(f"  Total:    {total:.1f}s")
    print(f"  Train:    Top-1 {train_t1*100:.1f}%, Top-5 {train_t5*100:.1f}%")
    print(f"  Val:      Top-1 {val_t1*100:.1f}%, Top-5 {val_t5*100:.1f}%")
    print(f"  Gap:      {(train_t1-val_t1)*100:.1f}pp")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

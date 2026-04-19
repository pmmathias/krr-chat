#!/usr/bin/env python3
"""
Experiment I+G combined: CCA Spectral Embeddings + MoE K=8.

Replaces Word2Vec with CCA-based embeddings that maximize context→target
correlation via SVD on the co-occurrence matrix. Then runs MoE K=8 on top.

CCA Embeddings (Dhillon et al. 2015 "Eigenwords"):
  1. Build co-occurrence matrix C[i,j] = count(token j appears within window of token i)
  2. SVD: C = U Σ V^T
  3. Embeddings = U[:, :d] · Σ[:d, :d]^{1/2}

This is closed-form (eigendecomposition), optimized for context-target
relationships (unlike Word2Vec which uses gradient descent internally).
"""
import os, sys, time, math, argparse, pickle
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import svds


def parse_args():
    p = argparse.ArgumentParser(description='CCA Embeddings + MoE')
    p.add_argument('--corpus', default='data/autoregressive/corpus_15m.txt')
    p.add_argument('--tokenizer', default='data/autoregressive/bpe_50m.json')
    p.add_argument('--output', default='data/autoregressive/model_cca_moe.pkl')
    p.add_argument('--D', type=int, default=12288)
    p.add_argument('--emb-dim', type=int, default=64)
    p.add_argument('--dk', type=int, default=32)
    p.add_argument('--dv', type=int, default=64)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--dff', type=int, default=256)
    p.add_argument('--ctx', type=int, default=64)
    p.add_argument('--sigma', type=float, default=2.0)
    p.add_argument('--lam', type=float, default=1e-5)
    p.add_argument('--cg-maxiter', type=int, default=300)
    p.add_argument('--cg-tol', type=float, default=1e-5)
    p.add_argument('--batch-size', type=int, default=4096)
    p.add_argument('--val-frac', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--experts', type=int, default=8)
    p.add_argument('--kmeans-iters', type=int, default=50)
    p.add_argument('--kmeans-sample', type=int, default=500000)
    p.add_argument('--cca-window', type=int, default=5, help='CCA co-occurrence window')
    return p.parse_args()


def build_cca_embeddings(token_ids, V, emb_dim, window=5):
    """Build CCA/spectral embeddings via SVD on co-occurrence matrix."""
    print(f"  Building co-occurrence matrix (V={V}, window={window})...")
    t0 = time.time()
    # Sparse co-occurrence matrix
    C = lil_matrix((V, V), dtype=np.float32)
    N = len(token_ids)
    for i in range(N):
        tid = token_ids[i]
        for j in range(max(0, i - window), min(N, i + window + 1)):
            if j != i:
                C[tid, token_ids[j]] += 1
        if i % 2000000 == 0 and i > 0:
            print(f"    {i:,}/{N:,} ({100*i/N:.0f}%)")
    C = C.tocsr()
    print(f"  Co-occurrence built in {time.time()-t0:.1f}s, nnz={C.nnz:,}")

    # PPMI (Positive Pointwise Mutual Information) transform
    print(f"  Computing PPMI...")
    row_sums = np.array(C.sum(axis=1)).flatten() + 1e-10
    col_sums = np.array(C.sum(axis=0)).flatten() + 1e-10
    total = C.sum() + 1e-10
    # For efficiency, apply PPMI only to nonzero entries
    C_coo = C.tocoo()
    pmi_data = np.log(C_coo.data * total / (row_sums[C_coo.row] * col_sums[C_coo.col]) + 1e-10)
    pmi_data = np.maximum(pmi_data, 0)  # PPMI: clip negatives
    from scipy.sparse import csr_matrix
    C_ppmi = csr_matrix((pmi_data, (C_coo.row, C_coo.col)), shape=(V, V))

    # Truncated SVD
    print(f"  SVD (k={emb_dim})...")
    t0 = time.time()
    U, S, Vt = svds(C_ppmi, k=emb_dim)
    print(f"  SVD done in {time.time()-t0:.1f}s")

    # Embeddings = U · sqrt(Σ) (weighted by singular values)
    embeddings = U * np.sqrt(S)[None, :]
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    embeddings = (embeddings / norms).astype(np.float32)

    return embeddings


def sinusoidal_pe(max_len, d_emb, device):
    pe = torch.zeros(max_len, d_emb, device=device)
    pos = torch.arange(max_len, device=device).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_emb, 2, device=device).float() * (-math.log(10000.0) / d_emb))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def block_cg_gpu(A, B, tol=1e-5, max_iter=500, verbose=False):
    D, V = B.shape
    diag_A = torch.diagonal(A).clone(); diag_A[diag_A.abs() < 1e-30] = 1.0
    inv_diag = 1.0 / diag_A
    X = torch.zeros_like(B); R = B.clone()
    Z = R * inv_diag.unsqueeze(1); P = Z.clone()
    rz_old = (R * Z).sum(dim=0); init_res = R.norm(dim=0).max().item()
    for it in range(max_iter):
        AP = A @ P; pAp = (P * AP).sum(dim=0); pAp[pAp.abs() < 1e-30] = 1.0
        alpha = rz_old / pAp
        X += P * alpha.unsqueeze(0); R -= AP * alpha.unsqueeze(0)
        Z = R * inv_diag.unsqueeze(1); rz_new = (R * Z).sum(dim=0)
        rel_res = R.norm(dim=0).max().item() / init_res
        if rel_res < tol: return X, {'iterations': it+1, 'residual': rel_res, 'converged': True}
        rz_old_s = rz_old.clone(); rz_old_s[rz_old_s.abs() < 1e-30] = 1.0
        P = Z + P * (rz_new / rz_old_s).unsqueeze(0); rz_old = rz_new
    return X, {'iterations': max_iter, 'residual': rel_res, 'converged': False}


def gpu_kmeans(features, K, n_iters=50, seed=42):
    N, d = features.shape
    rng = np.random.default_rng(seed)
    centers = features[rng.choice(N, K, replace=False)].clone()
    for it in range(n_iters):
        dists = torch.cdist(features, centers)
        assignments = dists.argmin(dim=1)
        new_centers = torch.zeros_like(centers)
        for k in range(K):
            mask = (assignments == k)
            if mask.sum() > 0: new_centers[k] = features[mask].mean(dim=0)
            else: new_centers[k] = centers[k]
        shift = (new_centers - centers).norm()
        centers = new_centers
        if shift < 1e-6: break
    counts = [(assignments == k).sum().item() for k in range(K)]
    return centers, assignments, counts


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device(args.device)
    K = args.experts

    print(f"CCA Embeddings + MoE K={K}, D={args.D}")

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer)
    V = tokenizer.get_vocab_size()
    with open(args.corpus) as f: text = f.read()
    all_ids = np.array(tokenizer.encode(text).ids, dtype=np.int64)
    n_val = int(len(all_ids) * args.val_frac)
    train_ids = all_ids[:len(all_ids) - n_val]
    val_ids = all_ids[len(all_ids) - n_val:]
    N = len(train_ids) - args.ctx
    print(f"Corpus: {len(all_ids):,} tokens, V={V}, N={N:,}")

    CTX, EMB, DK, DV, H, DFF, D = args.ctx, args.emb_dim, args.dk, args.dv, args.heads, args.dff, args.D
    FEAT = 2 * EMB

    # ---- CCA Embeddings (replaces Word2Vec) ----
    print("\n=== CCA Spectral Embeddings ===")
    emb_np = build_cca_embeddings(train_ids, V, EMB, window=args.cca_window)
    emb_t = torch.from_numpy(emb_np).to(dev)
    print(f"  CCA embeddings: {emb_np.shape}")

    # ---- Also train Word2Vec for comparison ----
    print("\n=== Word2Vec (for comparison) ===")
    from gensim.models import Word2Vec
    str_toks = [str(i) for i in train_ids]
    sents = [str_toks[i:i+100] for i in range(0, len(str_toks), 100)]
    w2v = Word2Vec(sents, vector_size=EMB, window=8, min_count=1, workers=4, sg=0, epochs=15, seed=args.seed)
    w2v_np = np.zeros((V, EMB), dtype=np.float32)
    for tid in range(V):
        k = str(tid)
        if k in w2v.wv: w2v_np[tid] = w2v.wv[k]
        else: w2v_np[tid] = np.random.randn(EMB).astype(np.float32) * 0.01
    w2v_np /= (np.linalg.norm(w2v_np, axis=1, keepdims=True) + 1e-8)
    del w2v

    # ---- Shared attention + RFF ----
    pe_t = sinusoidal_pe(CTX, EMB, dev)
    si = 1.0 / math.sqrt(EMB)
    g = torch.Generator(device='cpu').manual_seed(args.seed)
    attn_params = {
        'W_Q': torch.randn(EMB, H*DK, generator=g).to(dev) * si,
        'W_K': torch.randn(EMB, H*DK, generator=g).to(dev) * si,
        'W_V': torch.randn(EMB, H*DV, generator=g).to(dev) * si,
        'W_O': torch.randn(H*DV, EMB, generator=g).to(dev) / math.sqrt(H*DV),
        'W_in': torch.randn(EMB, DFF, generator=g).to(dev) / math.sqrt(EMB),
        'W_out': torch.randn(DFF, EMB, generator=g).to(dev) / math.sqrt(DFF),
    }
    g_rff = torch.Generator(device='cpu').manual_seed(args.seed + 1000)
    omega_t = (torch.randn(FEAT, D, generator=g_rff) / args.sigma).to(dev)
    bias_t = (torch.rand(D, generator=g_rff) * 2 * math.pi).to(dev)
    rff_scale = math.sqrt(2.0 / D)
    causal_mask = torch.triu(torch.ones(CTX, CTX, device=dev), diagonal=1).bool()

    def encode_all(emb_tensor, label=""):
        """Encode all training contexts with given embeddings."""
        print(f"\n  Encoding {N:,} contexts ({label})...")
        train_ids_t = torch.from_numpy(train_ids).long()
        indices = torch.arange(CTX).unsqueeze(0) + torch.arange(N).unsqueeze(1)
        all_ctx = train_ids_t[indices]
        all_feats = torch.zeros(N, FEAT, dtype=torch.float32)
        BS = args.batch_size
        t0 = time.time()
        for start in range(0, N, BS):
            end = min(start + BS, N)
            batch_ctx = all_ctx[start:end].to(dev); B = batch_ctx.shape[0]
            E = emb_tensor[batch_ctx]; X = E + pe_t.unsqueeze(0)
            Q_ = (X @ attn_params['W_Q']).view(B, CTX, H, DK).permute(0, 2, 1, 3)
            K_ = (X @ attn_params['W_K']).view(B, CTX, H, DK).permute(0, 2, 3, 1)
            Vv_ = (X @ attn_params['W_V']).view(B, CTX, H, DV).permute(0, 2, 1, 3)
            sc = (Q_ @ K_) / math.sqrt(DK)
            sc.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            ao = (torch.softmax(sc, -1) @ Vv_).permute(0, 2, 1, 3).reshape(B, CTX, H*DV) @ attn_params['W_O']
            X1 = F.layer_norm(X + ao, [EMB])
            h = torch.relu(X1 @ attn_params['W_in'])
            X2 = F.layer_norm(X1 + h @ attn_params['W_out'], [EMB])
            feat = torch.cat([X2[:, -1, :], E[:, -1, :]], dim=1)
            all_feats[start:end] = feat.cpu()
        print(f"    Done in {time.time()-t0:.1f}s")
        return all_feats, all_ctx

    def train_moe(all_feats, all_ctx, emb_tensor, label=""):
        """Train MoE K=8 on given features."""
        all_targets = torch.from_numpy(train_ids).long()[args.ctx:args.ctx+N]
        BS = args.batch_size

        # K-means
        print(f"  K-means (K={K})...")
        n_sub = min(args.kmeans_sample, N)
        sub_idx = np.random.choice(N, n_sub, replace=False)
        centers, _, _ = gpu_kmeans(all_feats[sub_idx].to(dev), K, n_iters=args.kmeans_iters)
        all_assign = torch.zeros(N, dtype=torch.long)
        for start in range(0, N, BS * 4):
            end = min(start + BS * 4, N)
            dists = torch.cdist(all_feats[start:end].to(dev), centers)
            all_assign[start:end] = dists.argmin(dim=1).cpu()
        sizes = [(all_assign == k).sum().item() for k in range(K)]
        print(f"    Cluster sizes: {sizes}")

        # Train experts
        expert_W = []
        for k in range(K):
            mask = (all_assign == k); n_k = mask.sum().item()
            if n_k < 100:
                expert_W.append(None); continue
            print(f"    Expert {k}: {n_k:,} samples", end="", flush=True)
            t0 = time.time()
            ZtZ_k = torch.zeros(D, D, dtype=torch.float64, device=dev)
            ZtY_k = torch.zeros(D, V, dtype=torch.float64, device=dev)
            cluster_idx = torch.where(mask)[0]
            for bs in range(0, n_k, BS):
                be = min(bs + BS, n_k)
                idx = cluster_idx[bs:be]
                feat_batch = all_feats[idx].to(dev)
                tgt_batch = all_targets[idx]
                with torch.no_grad():
                    z = rff_scale * torch.cos(feat_batch @ omega_t + bias_t.unsqueeze(0))
                    z64 = z.double(); ZtZ_k += z64.T @ z64
                    for j in range(len(idx)):
                        ZtY_k[:, tgt_batch[j].item()] += z64[j]
            ZtZ_k += args.lam * torch.eye(D, dtype=torch.float64, device=dev)
            W_k, info = block_cg_gpu(ZtZ_k.float(), ZtY_k.float(), tol=args.cg_tol, max_iter=args.cg_maxiter)
            print(f" → {info['iterations']} iters, {time.time()-t0:.0f}s")
            expert_W.append(W_k)
            del ZtZ_k, ZtY_k

        # Global fallback
        print(f"    Global W...", end="", flush=True)
        t0 = time.time()
        ZtZ_g = torch.zeros(D, D, dtype=torch.float64, device=dev)
        ZtY_g = torch.zeros(D, V, dtype=torch.float64, device=dev)
        for start in range(0, N, BS):
            end = min(start + BS, N)
            feat_batch = all_feats[start:end].to(dev)
            tgt_batch = all_targets[start:end]
            with torch.no_grad():
                z = rff_scale * torch.cos(feat_batch @ omega_t + bias_t.unsqueeze(0))
                z64 = z.double(); ZtZ_g += z64.T @ z64
                for j in range(end - start):
                    ZtY_g[:, tgt_batch[j].item()] += z64[j]
        ZtZ_g += args.lam * torch.eye(D, dtype=torch.float64, device=dev)
        W_global, info_g = block_cg_gpu(ZtZ_g.float(), ZtY_g.float(), tol=args.cg_tol, max_iter=args.cg_maxiter)
        print(f" → {info_g['iterations']} iters, {time.time()-t0:.0f}s")
        del ZtZ_g, ZtY_g

        for k in range(K):
            if expert_W[k] is None: expert_W[k] = W_global

        # Eval
        print(f"\n  Evaluation ({label})...")
        val_ids_t = torch.from_numpy(val_ids).long()
        N_val = len(val_ids_t) - CTX
        sample = np.random.choice(N_val, min(5000, N_val), replace=False)
        top1_moe = top1_glob = top5_moe = 0
        for i in sample:
            ctx = val_ids_t[i:i+CTX].to(dev)
            E = emb_tensor[ctx].unsqueeze(0); X = E + pe_t[-CTX:].unsqueeze(0)
            Q_ = (X @ attn_params['W_Q']).view(1, CTX, H, DK).permute(0, 2, 1, 3)
            K_ = (X @ attn_params['W_K']).view(1, CTX, H, DK).permute(0, 2, 3, 1)
            Vv_ = (X @ attn_params['W_V']).view(1, CTX, H, DV).permute(0, 2, 1, 3)
            sc = (Q_ @ K_) / math.sqrt(DK)
            sc.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            ao = (torch.softmax(sc, -1) @ Vv_).permute(0, 2, 1, 3).reshape(1, CTX, H*DV) @ attn_params['W_O']
            X1 = F.layer_norm(X + ao, [EMB])
            hh = torch.relu(X1 @ attn_params['W_in'])
            X2 = F.layer_norm(X1 + hh @ attn_params['W_out'], [EMB])
            feat = torch.cat([X2[0, -1], E[0, -1]])
            with torch.no_grad():
                z = rff_scale * torch.cos(feat @ omega_t + bias_t)
                d = torch.cdist(feat.unsqueeze(0), centers)
                eidx = d.argmin().item()
                sc_moe = (z @ expert_W[eidx]).squeeze()
                sc_glob = (z @ W_global).squeeze()
            tgt = val_ids[i + CTX]
            if sc_moe.argmax().item() == tgt: top1_moe += 1
            _, t5 = sc_moe.topk(5)
            if tgt in t5.cpu().numpy(): top5_moe += 1
            if sc_glob.argmax().item() == tgt: top1_glob += 1
        n = len(sample)
        print(f"    {label} MoE  Top-1: {top1_moe/n*100:.1f}%, Top-5: {top5_moe/n*100:.1f}%")
        print(f"    {label} Glob Top-1: {top1_glob/n*100:.1f}%")
        return top1_moe/n, top5_moe/n, expert_W, W_global, centers

    # ---- Run with CCA embeddings ----
    print("\n" + "="*60)
    print("EXPERIMENT: CCA Embeddings + MoE K=8")
    print("="*60)
    cca_feats, cca_ctx = encode_all(emb_t, "CCA")
    cca_t1, cca_t5, cca_experts, cca_global, cca_centers = train_moe(cca_feats, cca_ctx, emb_t, "CCA")

    # ---- Run with Word2Vec for comparison (same attn/RFF params) ----
    print("\n" + "="*60)
    print("COMPARISON: Word2Vec + MoE K=8 (same attn/RFF)")
    print("="*60)
    w2v_t = torch.from_numpy(w2v_np).to(dev)
    w2v_feats, w2v_ctx = encode_all(w2v_t, "W2V")
    w2v_t1, w2v_t5, _, _, _ = train_moe(w2v_feats, w2v_ctx, w2v_t, "W2V")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"RESULTS COMPARISON")
    print(f"{'='*60}")
    print(f"  CCA  + MoE K=8: Val Top-1 = {cca_t1*100:.1f}%, Top-5 = {cca_t5*100:.1f}%")
    print(f"  W2V  + MoE K=8: Val Top-1 = {w2v_t1*100:.1f}%, Top-5 = {w2v_t5*100:.1f}%")
    print(f"  Previous best (W2V MoE K=8): 18.3%")
    print(f"{'='*60}")

    # Save best
    with open(args.output, 'wb') as f:
        pickle.dump({
            'expert_W': [w.cpu().numpy() for w in cca_experts],
            'W_global': cca_global.cpu().numpy(),
            'centers': cca_centers.cpu().numpy(),
            'omega': omega_t.cpu().numpy(), 'bias': bias_t.cpu().numpy(),
            'emb_cca': emb_np, 'emb_w2v': w2v_np,
            'pe': pe_t.cpu().numpy(),
            'attn': {k: v.cpu().numpy() for k, v in attn_params.items()},
            'config': {'CTX': CTX, 'EMB_DIM': EMB, 'D_K': DK, 'D_V': DV,
                       'N_HEADS': H, 'D_FF': DFF, 'FEAT': FEAT, 'D': D,
                       'SIGMA': args.sigma, 'LAMBDA': args.lam, 'V': V, 'K': K},
            'stats': {'cca_top1': cca_t1, 'w2v_top1': w2v_t1},
        }, f)
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()

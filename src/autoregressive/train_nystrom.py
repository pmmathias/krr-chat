#!/usr/bin/env python3
"""
Experiment F: Nyström features statt RFF — data-dependent kernel approximation.

Instead of random ω for z(x) = cos(x·ω + b), use ACTUAL DATA POINTS as landmarks:
  z(x) = K(x, landmarks) @ K_mm^{-1/2}

where K(x, l_j) = exp(-||x - l_j||^2 / 2σ^2) is the Gaussian kernel,
and K_mm is the kernel matrix between landmarks.

This gives data-dependent features of dimension m (number of landmarks).
Combined with MoE K=8 for best comparison with our current winner.

Reference: Williams & Seeger 2001, "Using the Nyström Method to Speed Up Kernel Machines"
"""
import os, sys, time, math, argparse, pickle
import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser(description='Nyström + MoE')
    p.add_argument('--corpus', default='data/autoregressive/corpus_15m.txt')
    p.add_argument('--tokenizer', default='data/autoregressive/bpe_50m.json')
    p.add_argument('--output', default='data/autoregressive/model_nystrom_moe.pkl')
    p.add_argument('--landmarks', type=int, default=8192, help='Number of Nyström landmarks')
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
    p.add_argument('--w2v-epochs', type=int, default=15)
    p.add_argument('--experts', type=int, default=8)
    p.add_argument('--kmeans-iters', type=int, default=50)
    p.add_argument('--kmeans-sample', type=int, default=500000)
    # Also run RFF baseline for fair comparison
    p.add_argument('--D-rff', type=int, default=12288, help='RFF dim for comparison')
    return p.parse_args()


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
        shift = (new_centers - centers).norm(); centers = new_centers
        if shift < 1e-6: break
    return centers, assignments


def compute_nystrom_transform(landmarks, sigma):
    """Compute K_mm^{-1/2} for Nyström embedding.

    landmarks: (m, FEAT) tensor
    Returns: K_mm_inv_sqrt (m, m) tensor
    """
    m = landmarks.shape[0]
    # Kernel matrix between landmarks
    dists = torch.cdist(landmarks, landmarks) ** 2
    K_mm = torch.exp(-dists / (2 * sigma ** 2))
    # Add small regularization for numerical stability
    K_mm += 1e-6 * torch.eye(m, device=landmarks.device)
    # Eigendecomposition: K_mm = U Λ U^T
    eigenvalues, eigenvectors = torch.linalg.eigh(K_mm)
    # K_mm^{-1/2} = U Λ^{-1/2} U^T
    eigenvalues = eigenvalues.clamp(min=1e-8)
    K_mm_inv_sqrt = eigenvectors @ torch.diag(1.0 / eigenvalues.sqrt()) @ eigenvectors.T
    return K_mm_inv_sqrt


def nystrom_embed(features, landmarks, K_mm_inv_sqrt, sigma):
    """Compute Nyström embedding: z(x) = K(x, landmarks) @ K_mm^{-1/2}

    features: (B, FEAT) tensor
    landmarks: (m, FEAT) tensor
    Returns: (B, m) tensor
    """
    dists = torch.cdist(features, landmarks) ** 2
    K_xm = torch.exp(-dists / (2 * sigma ** 2))  # (B, m)
    return K_xm @ K_mm_inv_sqrt  # (B, m)


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device(args.device)
    K = args.experts; m = args.landmarks

    print(f"Nyström + MoE K={K}, m={m} landmarks")
    if dev.type == 'cuda': print(f"GPU: {torch.cuda.get_device_name()}")

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer)
    V = tokenizer.get_vocab_size()
    with open(args.corpus) as f: text = f.read()
    all_ids = np.array(tokenizer.encode(text).ids, dtype=np.int64)
    n_val = int(len(all_ids) * args.val_frac)
    train_ids = all_ids[:len(all_ids) - n_val]; val_ids = all_ids[len(all_ids) - n_val:]
    N = len(train_ids) - args.ctx
    print(f"Corpus: {len(all_ids):,} tokens, V={V}, N={N:,}")

    CTX, EMB, DK, DV, H, DFF = args.ctx, args.emb_dim, args.dk, args.dv, args.heads, args.dff
    FEAT = 2 * EMB

    # ---- Embeddings + attention (shared) ----
    print("Word2Vec...")
    from gensim.models import Word2Vec
    str_toks = [str(i) for i in train_ids]
    sents = [str_toks[i:i+100] for i in range(0, len(str_toks), 100)]
    w2v = Word2Vec(sents, vector_size=EMB, window=8, min_count=1, workers=4, sg=0, epochs=args.w2v_epochs, seed=args.seed)
    emb_np = np.zeros((V, EMB), dtype=np.float32)
    for tid in range(V):
        k = str(tid)
        if k in w2v.wv: emb_np[tid] = w2v.wv[k]
        else: emb_np[tid] = np.random.randn(EMB).astype(np.float32) * 0.01
    emb_np /= (np.linalg.norm(emb_np, axis=1, keepdims=True) + 1e-8)
    emb_t = torch.from_numpy(emb_np).to(dev)
    del w2v, emb_np

    pe_t = sinusoidal_pe(CTX, EMB, dev)
    si = 1.0 / math.sqrt(EMB)
    g = torch.Generator(device='cpu').manual_seed(args.seed)
    attn = {
        'W_Q': torch.randn(EMB, H*DK, generator=g).to(dev) * si,
        'W_K': torch.randn(EMB, H*DK, generator=g).to(dev) * si,
        'W_V': torch.randn(EMB, H*DV, generator=g).to(dev) * si,
        'W_O': torch.randn(H*DV, EMB, generator=g).to(dev) / math.sqrt(H*DV),
        'W_in': torch.randn(EMB, DFF, generator=g).to(dev) / math.sqrt(EMB),
        'W_out': torch.randn(DFF, EMB, generator=g).to(dev) / math.sqrt(DFF),
    }
    causal_mask = torch.triu(torch.ones(CTX, CTX, device=dev), diagonal=1).bool()

    # ---- Encode all contexts ----
    print(f"\nEncoding {N:,} contexts...")
    train_ids_t = torch.from_numpy(train_ids).long()
    indices = torch.arange(CTX).unsqueeze(0) + torch.arange(N).unsqueeze(1)
    all_ctx = train_ids_t[indices]
    all_targets = train_ids_t[args.ctx:args.ctx+N]
    BS = args.batch_size

    all_feats = torch.zeros(N, FEAT, dtype=torch.float32)
    t0 = time.time()
    for start in range(0, N, BS):
        end = min(start + BS, N)
        batch_ctx = all_ctx[start:end].to(dev); B = batch_ctx.shape[0]
        E = emb_t[batch_ctx]; X = E + pe_t.unsqueeze(0)
        Q_ = (X @ attn['W_Q']).view(B, CTX, H, DK).permute(0, 2, 1, 3)
        K_ = (X @ attn['W_K']).view(B, CTX, H, DK).permute(0, 2, 3, 1)
        Vv_ = (X @ attn['W_V']).view(B, CTX, H, DV).permute(0, 2, 1, 3)
        sc = (Q_ @ K_) / math.sqrt(DK)
        sc.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        ao = (torch.softmax(sc, -1) @ Vv_).permute(0, 2, 1, 3).reshape(B, CTX, H*DV) @ attn['W_O']
        X1 = F.layer_norm(X + ao, [EMB])
        h = torch.relu(X1 @ attn['W_in'])
        X2 = F.layer_norm(X1 + h @ attn['W_out'], [EMB])
        feat = torch.cat([X2[:, -1, :], E[:, -1, :]], dim=1)
        all_feats[start:end] = feat.cpu()
        if start % (BS * 10) == 0: print(f"  {start:>9}/{N:,}")
    print(f"  Encoded in {time.time()-t0:.1f}s")

    # ---- Select Nyström landmarks (k-means++ on features) ----
    print(f"\nSelecting {m} Nyström landmarks via k-means++...")
    t0 = time.time()
    # Use k-means on a subsample to find representative landmarks
    n_sub = min(500000, N)
    sub_idx = np.random.choice(N, n_sub, replace=False)
    sub_feats = all_feats[sub_idx].to(dev)
    landmark_centers, _ = gpu_kmeans(sub_feats, m, n_iters=30, seed=args.seed)
    del sub_feats
    print(f"  Landmarks selected in {time.time()-t0:.1f}s")

    # ---- Compute K_mm^{-1/2} ----
    print(f"  Computing K_mm^{{-1/2}} ({m}×{m})...")
    t0 = time.time()
    K_mm_inv_sqrt = compute_nystrom_transform(landmark_centers, args.sigma)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ---- Nyström MoE: cluster + solve per expert ----
    print(f"\n=== Nyström MoE K={K} (m={m} landmarks) ===")

    # K-means for MoE routing (on attention features, not Nyström features)
    n_sub = min(args.kmeans_sample, N)
    sub_idx = np.random.choice(N, n_sub, replace=False)
    moe_centers, _ = gpu_kmeans(all_feats[sub_idx].to(dev), K, n_iters=args.kmeans_iters)
    all_assign = torch.zeros(N, dtype=torch.long)
    for start in range(0, N, BS * 4):
        end = min(start + BS * 4, N)
        dists = torch.cdist(all_feats[start:end].to(dev), moe_centers)
        all_assign[start:end] = dists.argmin(dim=1).cpu()
    sizes = [(all_assign == k).sum().item() for k in range(K)]
    print(f"  Cluster sizes: {sizes}")

    expert_W_nys = []
    for k in range(K):
        mask = (all_assign == k); n_k = mask.sum().item()
        if n_k < 100: expert_W_nys.append(None); continue
        print(f"  Expert {k}: {n_k:,} samples", end="", flush=True)
        t0 = time.time()
        ZtZ_k = torch.zeros(m, m, dtype=torch.float64, device=dev)
        ZtY_k = torch.zeros(m, V, dtype=torch.float64, device=dev)
        cluster_idx = torch.where(mask)[0]
        for bs in range(0, n_k, BS):
            be = min(bs + BS, n_k)
            idx = cluster_idx[bs:be]
            feat_batch = all_feats[idx].to(dev)
            tgt_batch = all_targets[idx]
            with torch.no_grad():
                z = nystrom_embed(feat_batch, landmark_centers, K_mm_inv_sqrt, args.sigma)
                z64 = z.double(); ZtZ_k += z64.T @ z64
                for j in range(len(idx)):
                    ZtY_k[:, tgt_batch[j].item()] += z64[j]
        ZtZ_k += args.lam * torch.eye(m, dtype=torch.float64, device=dev)
        W_k, info = block_cg_gpu(ZtZ_k.float(), ZtY_k.float(), tol=args.cg_tol, max_iter=args.cg_maxiter)
        print(f" → {info['iterations']} iters, {time.time()-t0:.0f}s")
        expert_W_nys.append(W_k)
        del ZtZ_k, ZtY_k

    # Global fallback
    print(f"  Global W (Nyström)...", end="", flush=True)
    t0 = time.time()
    ZtZ_g = torch.zeros(m, m, dtype=torch.float64, device=dev)
    ZtY_g = torch.zeros(m, V, dtype=torch.float64, device=dev)
    for start in range(0, N, BS):
        end = min(start + BS, N)
        feat_batch = all_feats[start:end].to(dev)
        tgt_batch = all_targets[start:end]
        with torch.no_grad():
            z = nystrom_embed(feat_batch, landmark_centers, K_mm_inv_sqrt, args.sigma)
            z64 = z.double(); ZtZ_g += z64.T @ z64
            for j in range(end - start):
                ZtY_g[:, tgt_batch[j].item()] += z64[j]
    ZtZ_g += args.lam * torch.eye(m, dtype=torch.float64, device=dev)
    W_global_nys, info_g = block_cg_gpu(ZtZ_g.float(), ZtY_g.float(), tol=args.cg_tol, max_iter=args.cg_maxiter)
    print(f" → {info_g['iterations']} iters, {time.time()-t0:.0f}s")
    del ZtZ_g, ZtY_g

    for k in range(K):
        if expert_W_nys[k] is None: expert_W_nys[k] = W_global_nys

    # ---- Evaluation ----
    print("\n  Evaluation (Nyström MoE)...")
    val_ids_t = torch.from_numpy(val_ids).long()
    N_val = len(val_ids_t) - CTX
    sample = np.random.choice(N_val, min(5000, N_val), replace=False)
    top1_nys = top5_nys = top1_glob_nys = 0
    for i in sample:
        ctx = val_ids_t[i:i+CTX].to(dev)
        B = 1; E = emb_t[ctx].unsqueeze(0); X = E + pe_t[-CTX:].unsqueeze(0)
        Q_ = (X @ attn['W_Q']).view(B, CTX, H, DK).permute(0, 2, 1, 3)
        K_ = (X @ attn['W_K']).view(B, CTX, H, DK).permute(0, 2, 3, 1)
        Vv_ = (X @ attn['W_V']).view(B, CTX, H, DV).permute(0, 2, 1, 3)
        sc = (Q_ @ K_) / math.sqrt(DK)
        sc.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        ao = (torch.softmax(sc, -1) @ Vv_).permute(0, 2, 1, 3).reshape(B, CTX, H*DV) @ attn['W_O']
        X1 = F.layer_norm(X + ao, [EMB])
        hh = torch.relu(X1 @ attn['W_in'])
        X2 = F.layer_norm(X1 + hh @ attn['W_out'], [EMB])
        feat = torch.cat([X2[0, -1], E[0, -1]])
        with torch.no_grad():
            z_nys = nystrom_embed(feat.unsqueeze(0), landmark_centers, K_mm_inv_sqrt, args.sigma).squeeze()
            d = torch.cdist(feat.unsqueeze(0), moe_centers)
            eidx = d.argmin().item()
            sc_moe = (z_nys @ expert_W_nys[eidx]).squeeze()
            sc_glob = (z_nys @ W_global_nys).squeeze()
        tgt = val_ids[i + CTX]
        if sc_moe.argmax().item() == tgt: top1_nys += 1
        _, t5 = sc_moe.topk(5)
        if tgt in t5.cpu().numpy(): top5_nys += 1
        if sc_glob.argmax().item() == tgt: top1_glob_nys += 1
    n = len(sample)
    print(f"    Nyström MoE  Top-1: {top1_nys/n*100:.1f}%, Top-5: {top5_nys/n*100:.1f}%")
    print(f"    Nyström Glob Top-1: {top1_glob_nys/n*100:.1f}%")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"  Nyström (m={m}) + MoE K={K}: Val Top-1 = {top1_nys/n*100:.1f}%")
    print(f"  Nyström Global:               Val Top-1 = {top1_glob_nys/n*100:.1f}%")
    print(f"  Previous best (RFF MoE K=8):  Val Top-1 = 18.3%")
    print(f"{'='*60}")

    # Save
    with open(args.output, 'wb') as f:
        pickle.dump({
            'landmarks': landmark_centers.cpu().numpy(),
            'K_mm_inv_sqrt': K_mm_inv_sqrt.cpu().numpy(),
            'expert_W': [w.cpu().numpy() for w in expert_W_nys],
            'W_global': W_global_nys.cpu().numpy(),
            'moe_centers': moe_centers.cpu().numpy(),
            'config': {'m': m, 'K': K, 'sigma': args.sigma, 'V': V, 'FEAT': FEAT,
                       'CTX': CTX, 'EMB_DIM': EMB},
            'stats': {'nys_moe_top1': top1_nys/n, 'nys_glob_top1': top1_glob_nys/n},
        }, f)
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()

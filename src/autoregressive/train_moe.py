#!/usr/bin/env python3
"""
Experiment G: Mixture of Experts (MoE) with k-means routing for KRR-LM.

Instead of one global W matrix for all contexts, we:
1. Encode all training contexts (attention features) — same as baseline
2. K-means cluster the feature vectors into K groups
3. For each cluster k: accumulate Z_k^T Z_k + Z_k^T Y_k from that cluster's samples
4. For each cluster k: solve W_k = (Z_k^TZ_k + λI)^{-1} Z_k^T Y_k
5. At inference: route to nearest cluster center, use its W_k

This multiplies effective model capacity by K without changing the feature
dimension. Each expert sees homogeneous data → easier regression.

Usage:
  python train_moe.py --experts 8 --D 12288
  python train_moe.py --experts 16 --D 12288
"""
import os, sys, time, math, argparse, pickle
import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser(description='MoE KRR-LM')
    p.add_argument('--corpus', default='data/autoregressive/corpus_15m.txt')
    p.add_argument('--tokenizer', default='data/autoregressive/bpe_50m.json')
    p.add_argument('--output', default='data/autoregressive/model_moe.pkl')
    p.add_argument('--D', type=int, default=12288)
    p.add_argument('--emb-dim', type=int, default=64)
    p.add_argument('--dk', type=int, default=32)
    p.add_argument('--dv', type=int, default=64)
    p.add_argument('--heads', type=int, default=4)
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
    # MoE-specific
    p.add_argument('--experts', type=int, default=8, help='Number of expert clusters')
    p.add_argument('--kmeans-iters', type=int, default=50, help='K-means iterations')
    p.add_argument('--kmeans-sample', type=int, default=500000, help='Subsample for k-means')
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
    diag_A = torch.diagonal(A).clone()
    diag_A[diag_A.abs() < 1e-30] = 1.0
    inv_diag = 1.0 / diag_A
    X = torch.zeros_like(B); R = B.clone()
    Z = R * inv_diag.unsqueeze(1); P = Z.clone()
    rz_old = (R * Z).sum(dim=0)
    init_res = R.norm(dim=0).max().item()
    for it in range(max_iter):
        AP = A @ P
        pAp = (P * AP).sum(dim=0)
        pAp[pAp.abs() < 1e-30] = 1.0
        alpha = rz_old / pAp
        X = X + P * alpha.unsqueeze(0)
        R = R - AP * alpha.unsqueeze(0)
        Z = R * inv_diag.unsqueeze(1)
        rz_new = (R * Z).sum(dim=0)
        rel_res = R.norm(dim=0).max().item() / init_res
        if rel_res < tol:
            return X, {'iterations': it+1, 'residual': rel_res, 'converged': True}
        rz_old_safe = rz_old.clone(); rz_old_safe[rz_old_safe.abs() < 1e-30] = 1.0
        beta = rz_new / rz_old_safe
        P = Z + P * beta.unsqueeze(0)
        rz_old = rz_new
    return X, {'iterations': max_iter, 'residual': rel_res, 'converged': False}


def gpu_kmeans(features, K, n_iters=50, seed=42):
    """Simple k-means on GPU. features: (N, d) tensor on GPU."""
    N, d = features.shape
    rng = np.random.default_rng(seed)
    # Init: random subset as centers
    idx = rng.choice(N, K, replace=False)
    centers = features[idx].clone()  # (K, d)

    for it in range(n_iters):
        # Assign: find nearest center for each point
        # (N, K) distance matrix via broadcasting
        dists = torch.cdist(features, centers)  # (N, K)
        assignments = dists.argmin(dim=1)  # (N,)

        # Update centers
        new_centers = torch.zeros_like(centers)
        counts = torch.zeros(K, device=features.device)
        for k in range(K):
            mask = (assignments == k)
            counts[k] = mask.sum()
            if counts[k] > 0:
                new_centers[k] = features[mask].mean(dim=0)
            else:
                new_centers[k] = centers[k]  # keep old if empty

        # Check convergence
        shift = (new_centers - centers).norm()
        centers = new_centers
        if it % 10 == 0:
            print(f"    k-means iter {it}: shift={shift:.4f}, sizes={counts.int().tolist()}")
        if shift < 1e-6:
            break

    return centers, assignments


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)
    K = args.experts

    print(f"MoE KRR-LM: K={K} experts, D={args.D}")
    if dev.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")

    # ---- Load corpus ----
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
    print(f"Corpus: {len(all_ids):,} tokens, V={V}, N={N:,}")

    CTX, EMB, DK, DV, H, DFF = args.ctx, args.emb_dim, args.dk, args.dv, args.heads, args.dff
    D = args.D
    FEAT = 2 * EMB

    # ---- Embeddings + attention ----
    print("Word2Vec...")
    from gensim.models import Word2Vec
    str_toks = [str(i) for i in train_ids]
    sents = [str_toks[i:i+100] for i in range(0, len(str_toks), 100)]
    w2v = Word2Vec(sents, vector_size=EMB, window=8, min_count=1,
                   workers=4, sg=0, epochs=args.w2v_epochs, seed=args.seed)
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
    attn_params = {
        'W_Q': torch.randn(EMB, H*DK, generator=g).to(dev) * si,
        'W_K': torch.randn(EMB, H*DK, generator=g).to(dev) * si,
        'W_V': torch.randn(EMB, H*DV, generator=g).to(dev) * si,
        'W_O': torch.randn(H*DV, EMB, generator=g).to(dev) / math.sqrt(H*DV),
        'W_in': torch.randn(EMB, DFF, generator=g).to(dev) / math.sqrt(EMB),
        'W_out': torch.randn(DFF, EMB, generator=g).to(dev) / math.sqrt(DFF),
    }
    causal_mask = torch.triu(torch.ones(CTX, CTX, device=dev), diagonal=1).bool()

    # RFF (shared across all experts)
    g_rff = torch.Generator(device='cpu').manual_seed(args.seed + 1000)
    omega_t = (torch.randn(FEAT, D, generator=g_rff) / args.sigma).to(dev)
    bias_t = (torch.rand(D, generator=g_rff) * 2 * math.pi).to(dev)
    rff_scale = math.sqrt(2.0 / D)

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
        batch_ctx = all_ctx[start:end].to(dev)
        B = batch_ctx.shape[0]
        E = emb_t[batch_ctx]; X = E + pe_t.unsqueeze(0)
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
        if start % (BS * 10) == 0:
            print(f"  {start:>9}/{N:,} ({100*start/N:.1f}%)")
    print(f"  Encoded in {time.time()-t0:.1f}s")

    # ---- K-means clustering on attention features ----
    print(f"\nK-means clustering (K={K})...")
    # Subsample for speed
    n_sub = min(args.kmeans_sample, N)
    sub_idx = np.random.choice(N, n_sub, replace=False)
    sub_feats = all_feats[sub_idx].to(dev)
    centers, _ = gpu_kmeans(sub_feats, K, n_iters=args.kmeans_iters, seed=args.seed)
    del sub_feats

    # Assign ALL samples to clusters
    print("  Assigning all samples to clusters...")
    all_assignments = torch.zeros(N, dtype=torch.long)
    for start in range(0, N, BS * 4):
        end = min(start + BS * 4, N)
        dists = torch.cdist(all_feats[start:end].to(dev), centers)
        all_assignments[start:end] = dists.argmin(dim=1).cpu()
    cluster_sizes = [(all_assignments == k).sum().item() for k in range(K)]
    print(f"  Cluster sizes: {cluster_sizes}")

    # ---- Train one W_k per cluster ----
    print(f"\nTraining {K} expert W matrices...")
    expert_W = []
    expert_cg_info = []

    for k in range(K):
        mask = (all_assignments == k)
        n_k = mask.sum().item()
        if n_k < 100:
            print(f"  Expert {k}: only {n_k} samples — using global fallback")
            expert_W.append(None)
            expert_cg_info.append(None)
            continue

        print(f"  Expert {k}: {n_k:,} samples")
        t0 = time.time()
        ZtZ_k = torch.zeros(D, D, dtype=torch.float64, device=dev)
        ZtY_k = torch.zeros(D, V, dtype=torch.float64, device=dev)

        # Get indices for this cluster
        cluster_indices = torch.where(mask)[0]

        # Process in batches
        for batch_start in range(0, n_k, BS):
            batch_end = min(batch_start + BS, n_k)
            idx = cluster_indices[batch_start:batch_end]
            feat_batch = all_feats[idx].to(dev)
            tgt_batch = all_targets[idx]

            with torch.no_grad():
                z = rff_scale * torch.cos(feat_batch @ omega_t + bias_t.unsqueeze(0))
                z64 = z.double()
                ZtZ_k += z64.T @ z64
                for j in range(len(idx)):
                    ZtY_k[:, tgt_batch[j].item()] += z64[j]

        ZtZ_k += args.lam * torch.eye(D, dtype=torch.float64, device=dev)
        W_k, info = block_cg_gpu(ZtZ_k.float(), ZtY_k.float(),
                                  tol=args.cg_tol, max_iter=args.cg_maxiter)
        t_k = time.time() - t0
        print(f"    Solve: {info['iterations']} iters, {t_k:.1f}s")
        expert_W.append(W_k)
        expert_cg_info.append(info)
        del ZtZ_k, ZtY_k

    # Global fallback W for empty/tiny clusters
    # (just use the baseline model if available, otherwise train one)
    print("\n  Training global fallback W...")
    t0 = time.time()
    ZtZ_g = torch.zeros(D, D, dtype=torch.float64, device=dev)
    ZtY_g = torch.zeros(D, V, dtype=torch.float64, device=dev)
    for start in range(0, N, BS):
        end = min(start + BS, N)
        feat_batch = all_feats[start:end].to(dev)
        tgt_batch = all_targets[start:end]
        with torch.no_grad():
            z = rff_scale * torch.cos(feat_batch @ omega_t + bias_t.unsqueeze(0))
            z64 = z.double()
            ZtZ_g += z64.T @ z64
            for j in range(end - start):
                ZtY_g[:, tgt_batch[j].item()] += z64[j]
    ZtZ_g += args.lam * torch.eye(D, dtype=torch.float64, device=dev)
    W_global, info_g = block_cg_gpu(ZtZ_g.float(), ZtY_g.float(),
                                     tol=args.cg_tol, max_iter=args.cg_maxiter)
    print(f"    Global solve: {info_g['iterations']} iters, {time.time()-t0:.1f}s")
    del ZtZ_g, ZtY_g

    # Replace None entries with global W
    for k in range(K):
        if expert_W[k] is None:
            expert_W[k] = W_global

    # ---- Evaluation ----
    print("\nEvaluation...")
    def eval_set(ids_np, label, n_eval=5000):
        ids_t = torch.from_numpy(ids_np).long()
        N_e = len(ids_t) - CTX
        if N_e <= 0: return 0.0, 0.0
        sample = np.random.choice(N_e, min(n_eval, N_e), replace=False)
        top1 = top5 = 0
        top1_global = 0  # compare with global-only
        for i in sample:
            ctx = ids_t[i:i+CTX].unsqueeze(0).to(dev)
            B = 1
            E = emb_t[ctx]; X = E + pe_t[-CTX:].unsqueeze(0)
            Q_ = (X @ attn_params['W_Q']).view(B, CTX, H, DK).permute(0, 2, 1, 3)
            K_ = (X @ attn_params['W_K']).view(B, CTX, H, DK).permute(0, 2, 3, 1)
            Vv_ = (X @ attn_params['W_V']).view(B, CTX, H, DV).permute(0, 2, 1, 3)
            sc = (Q_ @ K_) / math.sqrt(DK)
            sc.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            ao = (torch.softmax(sc, -1) @ Vv_).permute(0, 2, 1, 3).reshape(B, CTX, H*DV) @ attn_params['W_O']
            X1 = F.layer_norm(X + ao, [EMB])
            hh = torch.relu(X1 @ attn_params['W_in'])
            X2 = F.layer_norm(X1 + hh @ attn_params['W_out'], [EMB])
            feat = torch.cat([X2[0, -1], E[0, -1]])

            with torch.no_grad():
                z = rff_scale * torch.cos(feat @ omega_t + bias_t)

                # Route to nearest expert
                dists = torch.cdist(feat.unsqueeze(0), centers)
                expert_idx = dists.argmin().item()
                scores_moe = (z @ expert_W[expert_idx]).squeeze()

                # Global comparison
                scores_global = (z @ W_global).squeeze()

            tgt = ids_t[i+CTX].item()
            if scores_moe.argmax().item() == tgt: top1 += 1
            _, t5 = scores_moe.topk(5)
            if tgt in t5.cpu().numpy(): top5 += 1
            if scores_global.argmax().item() == tgt: top1_global += 1

        n = len(sample)
        t1 = top1/n; t5_ = top5/n; t1g = top1_global/n
        print(f"  {label} MoE  Top-1: {t1*100:.1f}%, Top-5: {t5_*100:.1f}%")
        print(f"  {label} Glob Top-1: {t1g*100:.1f}% (single-W baseline)")
        return t1, t5_

    train_t1, train_t5 = eval_set(train_ids, 'Train')
    val_t1, val_t5 = eval_set(val_ids, 'Val  ')

    # ---- Save ----
    print(f"\nSaving to {args.output}...")
    model = {
        'expert_W': [w.cpu().numpy() for w in expert_W],
        'centers': centers.cpu().numpy(),
        'W_global': W_global.cpu().numpy(),
        'omega': omega_t.cpu().numpy(),
        'bias': bias_t.cpu().numpy(),
        'emb': emb_t.cpu().numpy(),
        'pe': pe_t.cpu().numpy(),
        'attn': {k: v.cpu().numpy() for k, v in attn_params.items()},
        'config': {
            'CTX': CTX, 'EMB_DIM': EMB, 'D_K': DK, 'D_V': DV,
            'N_HEADS': H, 'D_FF': DFF, 'FEAT': FEAT, 'D': D,
            'SIGMA': args.sigma, 'LAMBDA': args.lam, 'V': V,
            'K': K,
        },
        'stats': {
            'train_top1': train_t1, 'val_top1': val_t1,
            'train_top5': train_t5, 'val_top5': val_t5,
            'cluster_sizes': cluster_sizes,
        },
    }
    with open(args.output, 'wb') as f:
        pickle.dump(model, f)

    print(f"\n{'='*60}")
    print(f"MoE DONE — K={K} experts")
    print(f"  Cluster sizes: {cluster_sizes}")
    print(f"  Val MoE  Top-1: {val_t1*100:.1f}%")
    print(f"  Val Glob Top-1: (printed above)")
    print(f"  (Baseline was 16.3%)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

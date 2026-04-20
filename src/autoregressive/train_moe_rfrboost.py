#!/usr/bin/env python3
"""
Experiment A+G: MoE K=8 + RFRBoost (3 rounds per expert).

Combines the two winners:
  - MoE: 8 specialized experts via k-means routing
  - RFRBoost: 3 boosting rounds per expert (residual correction)

Each expert gets its OWN boosting rounds with fresh random features.
Total: 8 experts × 3 rounds = 24 KRR solves.

This is the "everything we've got" configuration.
"""
import os, sys, time, math, argparse, pickle
import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser(description='MoE + RFRBoost')
    p.add_argument('--corpus', default='data/autoregressive/corpus_15m.txt')
    p.add_argument('--tokenizer', default='data/autoregressive/bpe_50m.json')
    p.add_argument('--output', default='data/autoregressive/model_moe_rfrboost.pkl')
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
    p.add_argument('--w2v-epochs', type=int, default=15)
    p.add_argument('--experts', type=int, default=8)
    p.add_argument('--rounds', type=int, default=3, help='Boosting rounds per expert')
    p.add_argument('--kmeans-iters', type=int, default=50)
    p.add_argument('--kmeans-sample', type=int, default=500000)
    return p.parse_args()


def sinusoidal_pe(max_len, d_emb, device):
    pe = torch.zeros(max_len, d_emb, device=device)
    pos = torch.arange(max_len, device=device).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_emb, 2, device=device).float() * (-math.log(10000.0) / d_emb))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def block_cg_gpu(A, B, tol=1e-5, max_iter=300, verbose=False):
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
        if rel_res < tol: return X, it+1
        rz_old_s = rz_old.clone(); rz_old_s[rz_old_s.abs() < 1e-30] = 1.0
        P = Z + P * (rz_new / rz_old_s).unsqueeze(0); rz_old = rz_new
    return X, max_iter


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


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device(args.device)
    K, R = args.experts, args.rounds
    D = args.D

    print(f"MoE K={K} + RFRBoost R={R} (= {K*R} total solves), D={D}")
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

    # ---- Shared components ----
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
    all_ctx = train_ids_t[indices]; all_targets = train_ids_t[args.ctx:args.ctx+N]
    BS = args.batch_size

    all_feats = torch.zeros(N, FEAT, dtype=torch.float32)
    t0 = time.time()
    for start in range(0, N, BS):
        end = min(start + BS, N); batch_ctx = all_ctx[start:end].to(dev); B = batch_ctx.shape[0]
        E = emb_t[batch_ctx]; X = E + pe_t.unsqueeze(0)
        Q_ = (X @ attn['W_Q']).view(B, CTX, H, DK).permute(0, 2, 1, 3)
        K_ = (X @ attn['W_K']).view(B, CTX, H, DK).permute(0, 2, 3, 1)
        Vv_ = (X @ attn['W_V']).view(B, CTX, H, DV).permute(0, 2, 1, 3)
        sc = (Q_ @ K_) / math.sqrt(DK)
        sc.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        ao = (torch.softmax(sc, -1) @ Vv_).permute(0, 2, 1, 3).reshape(B, CTX, H*DV) @ attn['W_O']
        X1 = F.layer_norm(X + ao, [EMB]); h = torch.relu(X1 @ attn['W_in'])
        X2 = F.layer_norm(X1 + h @ attn['W_out'], [EMB])
        all_feats[start:end] = torch.cat([X2[:, -1, :], E[:, -1, :]], dim=1).cpu()
    print(f"  Encoded in {time.time()-t0:.1f}s")

    # ---- MoE clustering ----
    print(f"\nK-means (K={K})...")
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

    # ---- Train each expert with RFRBoost ----
    rff_scale = math.sqrt(2.0 / D)
    expert_rounds = []  # expert_rounds[k] = [(omega, bias, W), ...]

    for k in range(K):
        mask = (all_assign == k); n_k = mask.sum().item()
        if n_k < 100:
            expert_rounds.append(None); continue

        print(f"\n  Expert {k}: {n_k:,} samples, {R} boosting rounds")
        cluster_idx = torch.where(mask)[0]
        rounds_k = []

        for r in range(R):
            g_rff = torch.Generator(device='cpu').manual_seed(args.seed + 5000 + k * 100 + r * 10)
            omega_r = (torch.randn(FEAT, D, generator=g_rff) / args.sigma).to(dev)
            bias_r = (torch.rand(D, generator=g_rff) * 2 * math.pi).to(dev)

            t0 = time.time()
            ZtZ = torch.zeros(D, D, dtype=torch.float64, device=dev)
            ZtY = torch.zeros(D, V, dtype=torch.float64, device=dev)
            # Cross-Grams for previous rounds of THIS expert
            cross_grams = [torch.zeros(D, D, dtype=torch.float64, device=dev) for _ in rounds_k]

            for bs in range(0, n_k, BS):
                be = min(bs + BS, n_k); idx = cluster_idx[bs:be]
                feat_batch = all_feats[idx].to(dev); tgt_batch = all_targets[idx]
                with torch.no_grad():
                    z_r = rff_scale * torch.cos(feat_batch @ omega_r + bias_r.unsqueeze(0))
                    z64 = z_r.double()
                    ZtZ += z64.T @ z64
                    for j in range(len(idx)):
                        ZtY[:, tgt_batch[j].item()] += z64[j]
                    for prev_idx, (om_p, bi_p, _) in enumerate(rounds_k):
                        z_p = rff_scale * torch.cos(feat_batch @ om_p + bi_p.unsqueeze(0))
                        cross_grams[prev_idx] += z64.T @ z_p.double()

            # Z_r^T R = Z_r^T Y - Σ (Z_r^T Z_p) W_p
            ZtR = ZtY.clone()
            for prev_idx, (_, _, W_p) in enumerate(rounds_k):
                ZtR -= cross_grams[prev_idx] @ W_p.double()
            del ZtY, cross_grams

            ZtZ += args.lam * torch.eye(D, dtype=torch.float64, device=dev)
            W_r, cg_iters = block_cg_gpu(ZtZ.float(), ZtR.float(), tol=args.cg_tol, max_iter=args.cg_maxiter)
            print(f"    Round {r+1}: {cg_iters} iters, {time.time()-t0:.0f}s")
            rounds_k.append((omega_r, bias_r, W_r))
            del ZtZ, ZtR

        expert_rounds.append(rounds_k)

    # ---- Evaluation ----
    print(f"\nEvaluation...")
    val_ids_t = torch.from_numpy(val_ids).long()
    N_val = len(val_ids_t) - CTX
    sample = np.random.choice(N_val, min(5000, N_val), replace=False)
    top1 = top5 = 0

    for i in sample:
        ctx = val_ids_t[i:i+CTX].to(dev)
        B = 1; E = emb_t[ctx].unsqueeze(0); X = E + pe_t[-CTX:].unsqueeze(0)
        Q_ = (X @ attn['W_Q']).view(B, CTX, H, DK).permute(0, 2, 1, 3)
        K_ = (X @ attn['W_K']).view(B, CTX, H, DK).permute(0, 2, 3, 1)
        Vv_ = (X @ attn['W_V']).view(B, CTX, H, DV).permute(0, 2, 1, 3)
        sc = (Q_ @ K_) / math.sqrt(DK)
        sc.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        ao = (torch.softmax(sc, -1) @ Vv_).permute(0, 2, 1, 3).reshape(B, CTX, H*DV) @ attn['W_O']
        X1 = F.layer_norm(X + ao, [EMB]); hh = torch.relu(X1 @ attn['W_in'])
        X2 = F.layer_norm(X1 + hh @ attn['W_out'], [EMB])
        feat = torch.cat([X2[0, -1], E[0, -1]])

        # Route to expert
        d = torch.cdist(feat.unsqueeze(0), moe_centers)
        eidx = d.argmin().item()
        rounds_k = expert_rounds[eidx]
        if rounds_k is None: continue

        with torch.no_grad():
            total_score = torch.zeros(V, device=dev)
            for om, bi, W in rounds_k:
                z = rff_scale * torch.cos(feat @ om + bi)
                total_score += z @ W

        tgt = val_ids[i + CTX]
        if total_score.argmax().item() == tgt: top1 += 1
        _, t5 = total_score.topk(5)
        if tgt in t5.cpu().numpy(): top5 += 1

    n = len(sample)
    print(f"\n{'='*60}")
    print(f"MoE K={K} + RFRBoost R={R} (= {K*R} solves)")
    print(f"  Val Top-1: {top1/n*100:.1f}%, Top-5: {top5/n*100:.1f}%")
    print(f"  Previous MoE K=8: 18.3%")
    print(f"  Previous RFRBoost 3R: 16.7%")
    print(f"{'='*60}")

    # Save
    with open(args.output, 'wb') as f:
        pickle.dump({
            'expert_rounds': [[(om.cpu().numpy(), bi.cpu().numpy(), W.cpu().numpy()) for om, bi, W in rk]
                              if rk else None for rk in expert_rounds],
            'moe_centers': moe_centers.cpu().numpy(),
            'config': {'K': K, 'R': R, 'D': D, 'V': V, 'FEAT': FEAT, 'sigma': args.sigma},
            'stats': {'val_top1': top1/n, 'val_top5': top5/n},
        }, f)
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()

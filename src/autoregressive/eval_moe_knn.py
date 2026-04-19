#!/usr/bin/env python3
"""
Evaluate MoE + kNN fusion on validation set.
Uses MoE model for KRR scores + kNN datastore for neighbor scores.
"""
import pickle, math, numpy as np, torch, torch.nn.functional as F, sys, time

def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load MoE model
    print("Loading MoE model...")
    with open('data/autoregressive/model_moe.pkl', 'rb') as f:
        moe = pickle.load(f)
    cfg = moe['config']
    CTX, EMB, H, DK, DV, D = cfg['CTX'], cfg['EMB_DIM'], cfg['N_HEADS'], cfg['D_K'], cfg['D_V'], cfg['D']
    FEAT, V = cfg['FEAT'], cfg['V']

    emb = torch.from_numpy(moe['emb']).to(dev)
    pe = torch.from_numpy(moe['pe']).to(dev)
    omega = torch.from_numpy(moe['omega']).to(dev)
    bias_t = torch.from_numpy(moe['bias']).to(dev)
    attn = {k: torch.from_numpy(v).to(dev) for k, v in moe['attn'].items()}
    expert_W = [torch.from_numpy(w).to(dev) for w in moe['expert_W']]
    centers = torch.from_numpy(moe['centers']).to(dev)
    W_global = torch.from_numpy(moe['W_global']).to(dev)
    causal = torch.triu(torch.ones(CTX, CTX, device=dev), diagonal=1).bool()
    rff_scale = math.sqrt(2.0 / D)

    # Load kNN datastore
    print("Loading kNN datastore...")
    ds = np.load('data/autoregressive/knn_datastore.npz')
    ds_feats = ds['features']  # (N, FEAT) float16
    ds_targets = ds['targets']  # (N,) int32
    print(f"  {len(ds_feats):,} entries")

    # Subsample datastore for speed
    N_ds = min(1000000, len(ds_feats))
    ds_idx = np.random.choice(len(ds_feats), N_ds, replace=False)
    ds_feats_sub = ds_feats[ds_idx].astype(np.float32)
    ds_targets_sub = ds_targets[ds_idx]
    print(f"  Using {N_ds:,} for kNN lookup")

    # Load corpus for val
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file('data/autoregressive/bpe_50m.json')
    with open('data/autoregressive/corpus_15m.txt') as f:
        text = f.read()
    all_ids = np.array(tokenizer.encode(text).ids, dtype=np.int64)
    n_val = int(len(all_ids) * 0.05)
    val_ids = torch.from_numpy(all_ids[len(all_ids) - n_val:]).long()
    N_val = len(val_ids) - CTX
    print(f"  Val samples: {N_val:,}")

    def encode_and_score(ctx_tensor):
        """Encode context → feat + MoE score."""
        ctx = ctx_tensor[-CTX:] if len(ctx_tensor) >= CTX else ctx_tensor
        n = len(ctx)
        E = emb[ctx].unsqueeze(0); X = E + pe[-n:].unsqueeze(0)
        Q_ = (X @ attn['W_Q']).view(1, n, H, DK).permute(0, 2, 1, 3)
        K_ = (X @ attn['W_K']).view(1, n, H, DK).permute(0, 2, 3, 1)
        Vv_ = (X @ attn['W_V']).view(1, n, H, DV).permute(0, 2, 1, 3)
        sc = (Q_ @ K_) / math.sqrt(DK)
        sc.masked_fill_(causal[:n, :n].unsqueeze(0).unsqueeze(0), float('-inf'))
        ao = (torch.softmax(sc, -1) @ Vv_).permute(0, 2, 1, 3).reshape(1, n, H*DV) @ attn['W_O']
        X1 = F.layer_norm(X + ao, [EMB])
        h = torch.relu(X1 @ attn['W_in'])
        X2 = F.layer_norm(X1 + h @ attn['W_out'], [EMB])
        feat = torch.cat([X2[0, -1], E[0, -1]])
        z = rff_scale * torch.cos(feat @ omega + bias_t)
        # MoE routing
        dists = torch.cdist(feat.unsqueeze(0), centers)
        expert_idx = dists.argmin().item()
        scores = (z @ expert_W[expert_idx]).squeeze()
        return feat.cpu().numpy(), scores.cpu().numpy()

    def knn_probs(query_feat, k=64, temp=10.0):
        dists = np.sum((ds_feats_sub - query_feat[None, :]) ** 2, axis=1)
        top_k = np.argpartition(dists, k)[:k]
        top_dists = dists[top_k]
        top_targets = ds_targets_sub[top_k]
        weights = np.exp(-top_dists / temp)
        weights /= weights.sum() + 1e-30
        probs = np.zeros(V, dtype=np.float32)
        for i in range(k):
            probs[top_targets[i]] += weights[i]
        return np.log(probs + 1e-30)

    # Evaluate with different lambda values
    n_eval = 3000
    sample = np.random.choice(N_val, min(n_eval, N_val), replace=False)

    for lam in [0.0, 0.1, 0.2, 0.3, 0.5]:
        top1 = top5 = 0
        t0 = time.time()
        for idx, i in enumerate(sample):
            ctx = val_ids[i:i+CTX].to(dev)
            with torch.no_grad():
                feat_np, moe_scores = encode_and_score(ctx)

            moe_lp = moe_scores - np.log(np.exp(moe_scores).sum() + 1e-30)

            if lam > 0:
                knn_lp = knn_probs(feat_np, k=64, temp=10.0)
                fused = (1 - lam) * moe_lp + lam * knn_lp
            else:
                fused = moe_lp

            tgt = val_ids[i + CTX].item()
            if np.argmax(fused) == tgt: top1 += 1
            if tgt in np.argpartition(-fused, 5)[:5]: top5 += 1

            if idx == 500 and idx > 0:
                print(f"  λ={lam}: {idx}/{n_eval} interim Top-1={top1/idx*100:.1f}%")

        n = len(sample)
        elapsed = time.time() - t0
        print(f"λ={lam}: Val Top-1={top1/n*100:.1f}%, Top-5={top5/n*100:.1f}% ({elapsed:.0f}s)")
    print()

if __name__ == '__main__':
    main()

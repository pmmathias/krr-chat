#!/usr/bin/env python3
"""
Experiment E: kNN-LM Hybrid — blend KRR predictions with k-nearest-neighbor lookup.

No retraining needed! Uses the existing trained model + a datastore of all
(feature_vector, next_token) pairs from training. At inference:

  P_final(w) = (1-λ) P_KRR(w) + λ P_kNN(w)

where P_kNN is computed from the k nearest training contexts.

Reference: Khandelwal et al. 2020, "Generalization through Memorization:
Nearest Neighbor Language Models" (ICLR 2020)

Usage:
  # Build datastore (run once, ~10 min)
  python knn_lm.py build --model model_L1_D12288_14M.pkl

  # Evaluate with kNN fusion
  python knn_lm.py eval --model model_L1_D12288_14M.pkl --k 64 --lam 0.3

  # Interactive generation
  python knn_lm.py generate --model model_L1_D12288_14M.pkl --k 64 --lam 0.3
"""
import os, sys, time, math, argparse, pickle
import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser(description='kNN-LM for KRR')
    sub = p.add_subparsers(dest='command')

    # Build datastore
    pb = sub.add_parser('build')
    pb.add_argument('--model', required=True)
    pb.add_argument('--corpus', default='data/autoregressive/corpus_15m.txt')
    pb.add_argument('--tokenizer', default='data/autoregressive/bpe_50m.json')
    pb.add_argument('--output', default='data/autoregressive/knn_datastore.npz')
    pb.add_argument('--batch-size', type=int, default=4096)
    pb.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    pb.add_argument('--val-frac', type=float, default=0.05)

    # Evaluate
    pe = sub.add_parser('eval')
    pe.add_argument('--model', required=True)
    pe.add_argument('--datastore', default='data/autoregressive/knn_datastore.npz')
    pe.add_argument('--tokenizer', default='data/autoregressive/bpe_50m.json')
    pe.add_argument('--corpus', default='data/autoregressive/corpus_15m.txt')
    pe.add_argument('--k', type=int, default=64, help='Number of nearest neighbors')
    pe.add_argument('--lam', type=float, default=0.3, help='kNN interpolation weight')
    pe.add_argument('--temp-knn', type=float, default=10.0, help='kNN temperature')
    pe.add_argument('--n-eval', type=int, default=5000)
    pe.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    pe.add_argument('--val-frac', type=float, default=0.05)

    # Generate
    pg = sub.add_parser('generate')
    pg.add_argument('--model', required=True)
    pg.add_argument('--datastore', default='data/autoregressive/knn_datastore.npz')
    pg.add_argument('--tokenizer', default='data/autoregressive/bpe_50m.json')
    pg.add_argument('--k', type=int, default=64)
    pg.add_argument('--lam', type=float, default=0.3)
    pg.add_argument('--temp-knn', type=float, default=10.0)
    pg.add_argument('--tokens', type=int, default=30)
    pg.add_argument('--temp', type=float, default=0.7)
    pg.add_argument('--topk', type=int, default=40)
    pg.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    return p.parse_args()


def load_model(path, device):
    with open(path, 'rb') as f:
        m = pickle.load(f)
    cfg = m['config']
    model = {
        'emb': torch.from_numpy(m['emb']).to(device),
        'pe': torch.from_numpy(m['pe']).to(device),
        'omega': torch.from_numpy(m['omega']).to(device),
        'bias': torch.from_numpy(m['bias']).to(device),
        'W': torch.from_numpy(m['W']).to(device),
        'layers': [{k: torch.from_numpy(v).to(device) for k, v in lp.items()}
                   for lp in m['layers']],
        'cfg': cfg,
    }
    return model


def encode_context(model, ctx_ids_tensor):
    """Encode a single context window → (FEAT,) feature vector + (D,) RFF vector."""
    cfg = model['cfg']
    CTX, EMB, H, DK, DV, D = cfg['CTX'], cfg['EMB_DIM'], cfg['N_HEADS'], cfg['D_K'], cfg['D_V'], cfg['D']
    FEAT = cfg['FEAT']
    dev = model['W'].device
    causal = torch.triu(torch.ones(CTX, CTX, device=dev), diagonal=1).bool()

    ctx = ctx_ids_tensor[-CTX:] if len(ctx_ids_tensor) >= CTX else ctx_ids_tensor
    n = len(ctx)
    E = model['emb'][ctx].unsqueeze(0)
    X = E + model['pe'][-n:].unsqueeze(0)
    for lp in model['layers']:
        Q_ = (X @ lp['W_Q']).view(1, n, H, DK).permute(0, 2, 1, 3)
        K_ = (X @ lp['W_K']).view(1, n, H, DK).permute(0, 2, 3, 1)
        Vv_ = (X @ lp['W_V']).view(1, n, H, DV).permute(0, 2, 1, 3)
        sc = (Q_ @ K_) / math.sqrt(DK)
        sc.masked_fill_(causal[:n, :n].unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = (torch.softmax(sc, -1) @ Vv_).permute(0, 2, 1, 3).reshape(1, n, H*DV) @ lp['W_O']
        X = F.layer_norm(X + attn, [EMB])
        h = torch.relu(X @ lp['W_in'])
        X = F.layer_norm(X + h @ lp['W_out'], [EMB])
    feat = torch.cat([X[0, -1], E[0, -1]])  # (FEAT,)
    rff_scale = math.sqrt(2.0 / D)
    z = rff_scale * torch.cos(feat @ model['omega'] + model['bias'])  # (D,)
    return feat, z


def build_datastore(args):
    """Build the kNN datastore: store all (feature, target) pairs from training."""
    dev = torch.device(args.device)
    model = load_model(args.model, dev)
    cfg = model['cfg']
    CTX, FEAT = cfg['CTX'], cfg['FEAT']

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer)
    with open(args.corpus) as f:
        text = f.read()
    all_ids = np.array(tokenizer.encode(text).ids, dtype=np.int64)
    n_val = int(len(all_ids) * args.val_frac)
    train_ids = all_ids[:len(all_ids) - n_val]
    N = len(train_ids) - CTX
    print(f"Building datastore: {N:,} entries, FEAT={FEAT}")

    # Store features (FEAT-dim, not D-dim — smaller and sufficient for kNN)
    all_feats = np.zeros((N, FEAT), dtype=np.float16)  # float16 to save memory
    all_targets = np.zeros(N, dtype=np.int32)

    train_ids_t = torch.from_numpy(train_ids).long()
    BS = args.batch_size

    t0 = time.time()
    for start in range(0, N, BS):
        end = min(start + BS, N)
        batch_feats = []
        for i in range(start, end):
            ctx = train_ids_t[i:i+CTX].to(dev)
            with torch.no_grad():
                feat, _ = encode_context(model, ctx)
            batch_feats.append(feat.cpu().numpy())
            all_targets[i] = train_ids[i + CTX]
        all_feats[start:end] = np.array(batch_feats, dtype=np.float16)
        if start % (BS * 5) == 0:
            elapsed = time.time() - t0
            print(f"  {start:>9}/{N:,} ({100*start/N:.1f}%)  {elapsed:.1f}s")

    np.savez_compressed(args.output, features=all_feats, targets=all_targets)
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"\nSaved datastore: {args.output} ({size_mb:.0f} MB)")
    print(f"  {N:,} entries × FEAT={FEAT} (float16)")


def knn_predict(query_feat, ds_feats, ds_targets, k=64, temp=10.0, V=32768):
    """Compute kNN distribution over vocabulary.

    query_feat: (FEAT,) numpy array
    ds_feats: (N, FEAT) numpy array (float16 datastore)
    ds_targets: (N,) numpy array (int32)
    Returns: (V,) log-probability array
    """
    # Compute distances (L2) — use float32 for precision
    query = query_feat.astype(np.float32)
    # For speed: subsample if datastore is too large
    N = len(ds_feats)
    if N > 1000000:
        idx = np.random.choice(N, 1000000, replace=False)
        feats = ds_feats[idx].astype(np.float32)
        targets = ds_targets[idx]
    else:
        feats = ds_feats.astype(np.float32)
        targets = ds_targets

    dists = np.sum((feats - query[None, :]) ** 2, axis=1)  # (N,)
    top_k_idx = np.argpartition(dists, k)[:k]
    top_k_dists = dists[top_k_idx]
    top_k_targets = targets[top_k_idx]

    # Softmax over negative distances
    weights = np.exp(-top_k_dists / temp)
    weights /= weights.sum() + 1e-30

    # Aggregate into vocabulary distribution
    knn_probs = np.zeros(V, dtype=np.float32)
    for i in range(k):
        knn_probs[top_k_targets[i]] += weights[i]

    return np.log(knn_probs + 1e-30)


def eval_knn(args):
    """Evaluate KRR + kNN fusion on validation set."""
    dev = torch.device(args.device)
    model = load_model(args.model, dev)
    cfg = model['cfg']
    CTX, D, V = cfg['CTX'], cfg['D'], cfg['V']

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer)
    with open(args.corpus) as f:
        text = f.read()
    all_ids = np.array(tokenizer.encode(text).ids, dtype=np.int64)
    n_val = int(len(all_ids) * args.val_frac)
    val_ids = all_ids[len(all_ids) - n_val:]

    # Load datastore
    print(f"Loading datastore from {args.datastore}...")
    ds = np.load(args.datastore)
    ds_feats = ds['features']
    ds_targets = ds['targets']
    print(f"  {len(ds_feats):,} entries loaded")

    N_val = len(val_ids) - CTX
    sample = np.random.choice(N_val, min(args.n_eval, N_val), replace=False)

    top1_krr = top1_knn = top1_fused = 0
    top5_fused = 0
    val_ids_t = torch.from_numpy(val_ids).long()

    print(f"\nEvaluating {len(sample)} val samples (k={args.k}, λ={args.lam})...")
    t0 = time.time()
    for idx, i in enumerate(sample):
        ctx = val_ids_t[i:i+CTX].to(dev)
        with torch.no_grad():
            feat, z = encode_context(model, ctx)
            krr_scores = (z @ model['W']).cpu().numpy()

        krr_lp = krr_scores - np.log(np.exp(krr_scores).sum() + 1e-30)  # log-softmax
        feat_np = feat.cpu().numpy()
        knn_lp = knn_predict(feat_np, ds_feats, ds_targets, k=args.k,
                             temp=args.temp_knn, V=V)

        # Fuse
        fused_lp = (1 - args.lam) * krr_lp + args.lam * knn_lp

        tgt = val_ids[i + CTX]
        if np.argmax(krr_lp) == tgt: top1_krr += 1
        if np.argmax(knn_lp) == tgt: top1_knn += 1
        if np.argmax(fused_lp) == tgt: top1_fused += 1
        if tgt in np.argpartition(-fused_lp, 5)[:5]: top5_fused += 1

        if idx % 500 == 0 and idx > 0:
            n = idx
            print(f"  {idx}/{len(sample)}: KRR={top1_krr/n*100:.1f}% "
                  f"kNN={top1_knn/n*100:.1f}% Fused={top1_fused/n*100:.1f}%")

    n = len(sample)
    print(f"\n{'='*60}")
    print(f"Results (k={args.k}, λ={args.lam}, temp_knn={args.temp_knn}):")
    print(f"  KRR only  Top-1: {top1_krr/n*100:.1f}%")
    print(f"  kNN only  Top-1: {top1_knn/n*100:.1f}%")
    print(f"  Fused     Top-1: {top1_fused/n*100:.1f}%, Top-5: {top5_fused/n*100:.1f}%")
    print(f"  Time: {time.time()-t0:.1f}s")
    print(f"{'='*60}")


def main():
    args = parse_args()
    if args.command == 'build':
        build_datastore(args)
    elif args.command == 'eval':
        eval_knn(args)
    elif args.command == 'generate':
        print("Interactive generation — use eval first to find best λ/k/temp")
    else:
        print("Usage: python knn_lm.py {build|eval|generate}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Experiment A: RFRBoost — Residual Random Feature Boosting for KRR-LM.

Instead of one solve W = (Z^TZ + λI)^{-1} Z^TY, we do K rounds:

  Round 1: W₁ = solve(Z₁, Y)              # standard KRR
           R₁ = Y - Z₁ · W₁              # compute residuals
  Round 2: W₂ = solve(Z₂, R₁)            # fit residuals with FRESH RFF
           R₂ = R₁ - Z₂ · W₂
  Round 3: W₃ = solve(Z₃, R₂)            # fit remaining residuals
           ...

  Prediction: score = z₁·W₁ + z₂·W₂ + z₃·W₃ + ...

Each round uses DIFFERENT random ω_k, b_k — so different RFF projections
see the data from different "angles". The residuals get smaller each round
(gradient boosting theory). All rounds are closed-form KRR solves.

Reference: Zozoulenko et al. 2025, "Random Feature Representation Boosting" (ICML)

Usage:
  python train_rfrboost.py --rounds 3 --D 12288
"""
import os, sys, time, math, argparse, pickle
import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser(description='RFRBoost for KRR-LM')
    p.add_argument('--corpus', default='data/autoregressive/corpus_15m.txt')
    p.add_argument('--tokenizer', default='data/autoregressive/bpe_50m.json')
    p.add_argument('--output', default='data/autoregressive/model_rfrboost.pkl')
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
    # Boosting-specific
    p.add_argument('--rounds', type=int, default=3, help='Number of boosting rounds')
    p.add_argument('--shrinkage', type=float, default=1.0, help='Learning rate for boosting (0<s<=1)')
    return p.parse_args()


def sinusoidal_pe(max_len, d_emb, device):
    pe = torch.zeros(max_len, d_emb, device=device)
    pos = torch.arange(max_len, device=device).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_emb, 2, device=device).float() * (-math.log(10000.0) / d_emb))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def block_cg_gpu(A, B, tol=1e-5, max_iter=500, verbose=True):
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
        if verbose and it % 100 == 0:
            print(f"      CG {it+1}: rel_res={rel_res:.2e}")
        if rel_res < tol:
            if verbose: print(f"      CG converged: {it+1} iters")
            return X, it+1
        rz_old_safe = rz_old.clone(); rz_old_safe[rz_old_safe.abs() < 1e-30] = 1.0
        beta = rz_new / rz_old_safe
        P = Z + P * beta.unsqueeze(0)
        rz_old = rz_new
    if verbose: print(f"      CG max_iter={max_iter}, rel_res={rel_res:.2e}")
    return X, max_iter


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)

    print(f"RFRBoost: {args.rounds} rounds, D={args.D}, shrinkage={args.shrinkage}")
    if dev.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")

    # ---- Load corpus + tokenizer ----
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
    FEAT = 2 * EMB  # attention output + last token emb

    # ---- Embeddings + attention (shared across all boosting rounds) ----
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

    # Attention params (1 layer, shared across all rounds)
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

    # ---- Encode all contexts ONCE (shared across rounds) ----
    print(f"\nEncoding all {N:,} contexts (attention features)...")
    train_ids_t = torch.from_numpy(train_ids).long()
    indices = torch.arange(CTX).unsqueeze(0) + torch.arange(N).unsqueeze(1)
    all_ctx = train_ids_t[indices]
    all_targets = train_ids_t[args.ctx:args.ctx+N]
    BS = args.batch_size

    # Compute attention features for all samples (stored on CPU to save GPU memory)
    all_feats = torch.zeros(N, FEAT, dtype=torch.float32)  # on CPU
    t0 = time.time()
    for start in range(0, N, BS):
        end = min(start + BS, N)
        batch_ctx = all_ctx[start:end].to(dev)
        B = batch_ctx.shape[0]
        E = emb_t[batch_ctx]
        X = E + pe_t.unsqueeze(0)
        # 1-layer attention
        Q_ = (X @ attn['W_Q']).view(B, CTX, H, DK).permute(0, 2, 1, 3)
        K_ = (X @ attn['W_K']).view(B, CTX, H, DK).permute(0, 2, 3, 1)
        Vv_ = (X @ attn['W_V']).view(B, CTX, H, DV).permute(0, 2, 1, 3)
        scores = (Q_ @ K_) / math.sqrt(DK)
        scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn_out = (torch.softmax(scores, -1) @ Vv_).permute(0, 2, 1, 3).reshape(B, CTX, H*DV) @ attn['W_O']
        X1 = F.layer_norm(X + attn_out, [EMB])
        h = torch.relu(X1 @ attn['W_in'])
        X2 = F.layer_norm(X1 + h @ attn['W_out'], [EMB])
        feat = torch.cat([X2[:, -1, :], E[:, -1, :]], dim=1)  # (B, FEAT)
        all_feats[start:end] = feat.cpu()
        if start % (BS * 10) == 0:
            print(f"  {start:>9}/{N:,} ({100*start/N:.1f}%)")
    t_encode = time.time() - t0
    print(f"  Encoded in {t_encode:.1f}s")

    # ---- Boosting rounds ----
    # Build one-hot Y on GPU (sparse would be better but dense is simpler)
    # We'll compute residuals as Y - sum(Z_k W_k) incrementally
    # Start: residual_target = Y (one-hot)

    rounds_data = []  # store (omega_k, bias_k, W_k) per round
    # We track residuals as a dense (D, V) matrix — but we can't store the
    # full N x V one-hot Y. Instead, we track the residual in the projected space.

    # Actually: the residual R = Y - Z·W is in the TOKEN space (N × V).
    # But we never materialize the full N×V matrix. Instead, for each round:
    #   ZtR = Z^T R = Z^T Y - Z^T Z W = ZtY_original - ZtZ W
    # And for the next round's accumulation:
    #   Z2^T R = Z2^T Y - Z2^T Z1 W1
    # This requires computing Z2^T Z1 which is D×D — tractable.

    # Simpler approach: just compute the residuals per-batch during accumulation.

    current_predictions = torch.zeros(V, dtype=torch.float32, device=dev)  # not needed globally

    print(f"\n{'='*60}")
    print(f"BOOSTING: {args.rounds} rounds")
    print(f"{'='*60}")

    for round_idx in range(args.rounds):
        print(f"\n--- Round {round_idx+1}/{args.rounds} ---")

        # Fresh RFF for this round
        g_rff = torch.Generator(device='cpu').manual_seed(args.seed + 5000 + round_idx * 100)
        omega_k = (torch.randn(FEAT, D, generator=g_rff) / args.sigma).to(dev)
        bias_k = (torch.rand(D, generator=g_rff) * 2 * math.pi).to(dev)
        rff_scale = math.sqrt(2.0 / D)

        # Accumulate Z_k^T Z_k and Z_k^T R_k (where R_k is the current residual target)
        print(f"  Accumulating ZtZ + ZtR (round {round_idx+1})...")
        t0 = time.time()
        ZtZ = torch.zeros(D, D, dtype=torch.float64, device=dev)
        ZtR = torch.zeros(D, V, dtype=torch.float64, device=dev)

        for start in range(0, N, BS):
            end = min(start + BS, N)
            feat_batch = all_feats[start:end].to(dev)
            tgt_batch = all_targets[start:end]

            with torch.no_grad():
                z = rff_scale * torch.cos(feat_batch @ omega_k + bias_k.unsqueeze(0))
                z64 = z.double()
                ZtZ += z64.T @ z64

                # Compute residual target for this batch
                # residual = one_hot(target) - sum of previous rounds' predictions
                for j in range(end - start):
                    tid = tgt_batch[j].item()
                    # One-hot contribution
                    residual_j = torch.zeros(V, dtype=torch.float64, device=dev)
                    residual_j[tid] = 1.0

                    # Subtract predictions from all previous rounds
                    for prev_omega, prev_bias, prev_W in rounds_data:
                        prev_z = rff_scale * torch.cos(feat_batch[j] @ prev_omega + prev_bias).double()
                        residual_j -= args.shrinkage * (prev_z @ prev_W.double())

                    ZtR[:, :] += z64[j].unsqueeze(1) * residual_j.unsqueeze(0)

            if start % (BS * 20) == 0:
                print(f"    {start:>9}/{N:,} ({100*start/N:.1f}%)  {time.time()-t0:.1f}s")

        ZtZ += args.lam * torch.eye(D, dtype=torch.float64, device=dev)
        t_accum = time.time() - t0
        print(f"  Accumulation: {t_accum:.1f}s")

        # Solve
        print(f"  Solving round {round_idx+1}...")
        t0 = time.time()
        W_k, cg_iters = block_cg_gpu(ZtZ.float(), ZtR.float(),
                                      tol=args.cg_tol, max_iter=args.cg_maxiter)
        t_solve = time.time() - t0
        print(f"  Solve: {t_solve:.1f}s ({cg_iters} CG iters)")
        del ZtZ, ZtR

        rounds_data.append((omega_k, bias_k, W_k))

        # Evaluate after this round
        print(f"  Evaluating round {round_idx+1}...")
        def eval_set(ids_np, label, n_eval=3000):
            ids_t = torch.from_numpy(ids_np).long()
            N_e = len(ids_t) - CTX
            if N_e <= 0: return 0.0, 0.0
            sample = np.random.choice(N_e, min(n_eval, N_e), replace=False)
            top1 = top5 = 0
            for i in sample:
                ctx = ids_t[i:i+CTX].unsqueeze(0).to(dev)
                B = 1
                E = emb_t[ctx]
                X = E + pe_t[-CTX:].unsqueeze(0)
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
                    total_score = torch.zeros(V, device=dev)
                    for om, bi, Wk in rounds_data:
                        zk = rff_scale * torch.cos(feat @ om + bi)
                        total_score += args.shrinkage * (zk @ Wk)

                pred = total_score.argmax().item()
                tgt = ids_t[i+CTX].item()
                if pred == tgt: top1 += 1
                _, t5 = total_score.topk(5)
                if tgt in t5.cpu().numpy(): top5 += 1
            t1 = top1/len(sample); t5_ = top5/len(sample)
            print(f"    {label} Top-1: {t1*100:.1f}%, Top-5: {t5_*100:.1f}%")
            return t1, t5_

        train_t1, train_t5 = eval_set(train_ids, f'Train (R{round_idx+1})')
        val_t1, val_t5 = eval_set(val_ids, f'Val   (R{round_idx+1})')
        print(f"    Gap: {(train_t1-val_t1)*100:.1f}pp")

    # ---- Save ----
    print(f"\nSaving to {args.output}...")
    model = {
        'rounds': [(om.cpu().numpy(), bi.cpu().numpy(), Wk.cpu().numpy())
                    for om, bi, Wk in rounds_data],
        'emb': emb_t.cpu().numpy(),
        'pe': pe_t.cpu().numpy(),
        'attn': {k: v.cpu().numpy() for k, v in attn.items()},
        'config': {
            'CTX': CTX, 'EMB_DIM': EMB, 'D_K': DK, 'D_V': DV,
            'N_HEADS': H, 'D_FF': DFF, 'FEAT': FEAT, 'D': D,
            'SIGMA': args.sigma, 'LAMBDA': args.lam, 'V': V,
            'n_rounds': len(rounds_data), 'shrinkage': args.shrinkage,
        },
        'stats': {
            'train_top1': train_t1, 'val_top1': val_t1,
            'train_top5': train_t5, 'val_top5': val_t5,
        },
    }
    with open(args.output, 'wb') as f:
        pickle.dump(model, f)

    print(f"\n{'='*60}")
    print(f"RFRBoost DONE — {args.rounds} rounds")
    print(f"  Val Top-1: {val_t1*100:.1f}%, Val Top-5: {val_t5*100:.1f}%")
    print(f"  (Baseline was 16.3%)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

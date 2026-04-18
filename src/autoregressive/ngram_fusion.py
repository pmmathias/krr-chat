#!/usr/bin/env python3
"""
Ansatz C: N-gram Shallow Fusion for inference.

Trains a simple n-gram model (trigram/5-gram with Kneser-Ney-like smoothing)
on the BPE token stream, then interpolates at inference time:

  log P(w|context) = log P_KRR(w|context) + lambda * log P_ngram(w|context)

This immediately improves local coherence because the n-gram model prevents
impossible word sequences. The KRR model provides global context awareness,
the n-gram model provides local fluency.

Usage:
  # Train n-gram model
  python ngram_fusion.py train --corpus corpus_15m.txt --tokenizer bpe_50m.json --order 5

  # Interactive generation with fusion
  python ngram_fusion.py generate --model model_L1_D12288_14M.pkl --ngram ngram_5.pkl --lambda 0.3
"""
import os, sys, pickle, math, argparse, collections
import numpy as np


def train_ngram(token_ids, order=5, min_count=2):
    """Train a simple n-gram model with add-k smoothing."""
    print(f"Training {order}-gram model on {len(token_ids):,} tokens...")
    V = int(token_ids.max()) + 1

    # Count n-grams
    counts = {}
    context_counts = {}
    for n in range(1, order + 1):
        print(f"  Counting {n}-grams...")
        counts[n] = collections.Counter()
        context_counts[n] = collections.Counter()
        for i in range(len(token_ids) - n):
            ngram = tuple(token_ids[i:i+n+1].tolist())
            context = ngram[:-1]
            counts[n][ngram] += 1
            context_counts[n][context] += 1

    # Prune rare n-grams
    for n in counts:
        before = len(counts[n])
        counts[n] = {k: v for k, v in counts[n].items() if v >= min_count}
        print(f"  {n}-gram: {before:,} → {len(counts[n]):,} after pruning (min_count={min_count})")

    return {
        'counts': counts,
        'context_counts': context_counts,
        'order': order,
        'V': V,
    }


def ngram_log_prob(ngram_model, context_ids, token_id, k=0.1):
    """Compute smoothed log probability of token_id given context_ids.
    Uses stupid backoff: try highest order first, back off to lower orders."""
    order = ngram_model['order']
    V = ngram_model['V']

    for n in range(min(order, len(context_ids)), 0, -1):
        ctx = tuple(context_ids[-n:])
        ngram = ctx + (token_id,)
        count = ngram_model['counts'].get(n+1 if n < order else order, {}).get(ngram, 0)
        ctx_count = ngram_model['context_counts'].get(n+1 if n < order else order, {}).get(ctx, 0)
        if ctx_count > 0:
            # Add-k smoothing
            prob = (count + k) / (ctx_count + k * V)
            return math.log(prob + 1e-30)

    # Unigram fallback
    unigrams = ngram_model['counts'].get(1, {})
    total = sum(unigrams.values())
    count = unigrams.get((token_id,), 0)
    prob = (count + k) / (total + k * V)
    return math.log(prob + 1e-30)


def ngram_log_probs_batch(ngram_model, context_ids, V, k=0.1):
    """Compute log probs for ALL tokens given context. Returns (V,) array."""
    log_probs = np.zeros(V, dtype=np.float32)
    for tid in range(V):
        log_probs[tid] = ngram_log_prob(ngram_model, context_ids, tid, k=k)
    return log_probs


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command')

    # Train subcommand
    p_train = sub.add_parser('train')
    p_train.add_argument('--corpus', required=True)
    p_train.add_argument('--tokenizer', required=True)
    p_train.add_argument('--order', type=int, default=5)
    p_train.add_argument('--output', default='data/autoregressive/ngram_5.pkl')

    # Generate subcommand
    p_gen = sub.add_parser('generate')
    p_gen.add_argument('--model', required=True, help='KRR model pickle')
    p_gen.add_argument('--ngram', required=True, help='N-gram model pickle')
    p_gen.add_argument('--tokenizer', default='data/autoregressive/bpe_50m.json')
    p_gen.add_argument('--lam', type=float, default=0.3, help='N-gram interpolation weight')
    p_gen.add_argument('--tokens', type=int, default=30)
    p_gen.add_argument('--temp', type=float, default=0.7)
    p_gen.add_argument('--topk', type=int, default=40)

    args = parser.parse_args()

    if args.command == 'train':
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(args.tokenizer)
        with open(args.corpus) as f:
            text = f.read()
        ids = np.array(tokenizer.encode(text).ids, dtype=np.int32)
        ngram = train_ngram(ids, order=args.order)
        with open(args.output, 'wb') as f:
            pickle.dump(ngram, f)
        print(f"\nSaved to {args.output}")

    elif args.command == 'generate':
        import torch
        import torch.nn.functional as F
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(args.tokenizer)
        with open(args.ngram, 'rb') as f:
            ngram = pickle.load(f)
        with open(args.model, 'rb') as f:
            model = pickle.load(f)

        cfg = model['config']
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        V = cfg['V']

        # Load model tensors
        emb = torch.from_numpy(model['emb']).to(dev)
        pe = torch.from_numpy(model['pe']).to(dev)
        omega = torch.from_numpy(model['omega']).to(dev)
        bias_t = torch.from_numpy(model['bias']).to(dev)
        W = torch.from_numpy(model['W']).to(dev)
        layers = [{k: torch.from_numpy(v).to(dev) for k, v in lp.items()}
                  for lp in model['layers']]
        CTX, EMB_DIM, H, DK, DV, D = cfg['CTX'], cfg['EMB_DIM'], cfg['N_HEADS'], cfg['D_K'], cfg['D_V'], cfg['D']
        FEAT = cfg['FEAT']
        causal = torch.triu(torch.ones(CTX, CTX, device=dev), diagonal=1).bool()

        def krr_scores(ids_tensor):
            ctx = ids_tensor[-CTX:] if len(ids_tensor) >= CTX else ids_tensor
            n = len(ctx)
            E = emb[ctx].unsqueeze(0)
            X = E + pe[-n:].unsqueeze(0)
            for lp in layers:
                Q_ = (X @ lp['W_Q']).view(1, n, H, DK).permute(0, 2, 1, 3)
                K_ = (X @ lp['W_K']).view(1, n, H, DK).permute(0, 2, 3, 1)
                Vv_ = (X @ lp['W_V']).view(1, n, H, DV).permute(0, 2, 1, 3)
                sc = (Q_ @ K_) / math.sqrt(DK)
                sc.masked_fill_(causal[:n, :n].unsqueeze(0).unsqueeze(0), float('-inf'))
                attn = (torch.softmax(sc, -1) @ Vv_).permute(0, 2, 1, 3).reshape(1, n, H*DV) @ lp['W_O']
                X = F.layer_norm(X + attn, [EMB_DIM])
                h = torch.relu(X @ lp['W_in'])
                X = F.layer_norm(X + h @ lp['W_out'], [EMB_DIM])
            vec = torch.cat([X[0, -1], E[0, -1]])
            z = math.sqrt(2.0 / D) * torch.cos(vec @ omega + bias_t)
            return (z @ W).cpu().numpy()

        print(f"Loaded KRR model + {ngram['order']}-gram. Lambda={args.lam}")
        print(f"Type a prompt (or 'quit'):\n")

        import readline
        rng = np.random.default_rng(42)
        while True:
            try:
                prompt = input("You> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if prompt.lower() in ('quit', 'exit', 'q'):
                break
            if not prompt:
                continue

            ids = list(tokenizer.encode(prompt).ids)
            print(f"KRR+{ngram['order']}gram> {prompt}", end='', flush=True)

            for _ in range(args.tokens):
                with torch.no_grad():
                    krr_sc = krr_scores(torch.tensor(ids, dtype=torch.long, device=dev))

                # N-gram log probs for top-k KRR candidates
                top_idx = np.argpartition(-krr_sc, args.topk)[:args.topk]

                # Fuse: log P = log P_KRR + lambda * log P_ngram
                fused = np.full(len(top_idx), -np.inf)
                for j, tid in enumerate(top_idx):
                    ng_lp = ngram_log_prob(ngram, ids[-ngram['order']:], int(tid))
                    # Normalize KRR scores to log-probs (softmax over top-k)
                    krr_lp = krr_sc[tid]
                    fused[j] = krr_lp + args.lam * ng_lp

                # Sample from fused distribution
                fused_shifted = fused / args.temp
                fused_shifted -= fused_shifted.max()
                probs = np.exp(fused_shifted)
                probs /= probs.sum()
                chosen = int(rng.choice(top_idx, p=probs))
                ids.append(chosen)
                print(tokenizer.decode([chosen]), end='', flush=True)

            print('\n')

    else:
        parser.print_help()


if __name__ == '__main__':
    main()

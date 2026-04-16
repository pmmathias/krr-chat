"""
T037 — Autoregressive generation from the trained KRR language model.

Loads the trained model and lets you predict the next token(s) for any prompt.

Usage:
  python3.11 src/autoregressive/generate.py
  python3.11 src/autoregressive/generate.py "Im Anfang war das" --n 5
"""
import os, sys, math, pickle, argparse
import numpy as np

DATA_DIR = '/Users/mathiasleonhardt/Dev/krr-chat/data/autoregressive'
TOKENIZER = f'{DATA_DIR}/bpe_tokenizer.json'
MODEL = f'{DATA_DIR}/model.pkl'


def load():
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(TOKENIZER)
    with open(MODEL, 'rb') as f:
        m = pickle.load(f)
    return tokenizer, m


def encode_context(ids, t, emb, cfg):
    """Encode the context preceding position t (same as training)."""
    CTX, EMB_DIM, FEAT, V = cfg['CTX'], cfg['EMB_DIM'], cfg['FEAT'], cfg['V']
    pos_weights = np.linspace(0.4, 1.0, CTX, dtype=np.float32)
    start = max(0, t - CTX)
    ctx = ids[start:t]
    ctx_len = len(ctx)
    vec = np.zeros(FEAT, dtype=np.float32)
    for i in range(ctx_len):
        slot = CTX - ctx_len + i
        tid = ctx[i]
        if 0 <= tid < V:
            vec[slot*EMB_DIM:(slot+1)*EMB_DIM] = pos_weights[slot] * emb[tid]
    return vec


def predict_next_logits(ids, emb, omega, bias, W, cfg):
    FEAT, D = cfg['FEAT'], cfg['D']
    scale = math.sqrt(2.0 / D)
    vec = encode_context(ids, len(ids), emb, cfg)
    z = scale * np.cos(vec @ omega + bias)
    scores = z @ W
    return scores


def top_k_for_prompt(tokenizer, m, prompt, k=10):
    ids = tokenizer.encode(prompt).ids
    scores = predict_next_logits(ids, m['emb'], m['omega'], m['bias'], m['W'], m['config'])
    top_idx = np.argpartition(-scores, k)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    results = []
    for idx in top_idx:
        tok = tokenizer.decode([int(idx)])
        results.append((int(idx), tok, float(scores[idx])))
    return results, ids


def generate(tokenizer, m, prompt, n_tokens=20, temperature=0.8, top_k=40, seed=42):
    rng = np.random.default_rng(seed)
    ids = list(tokenizer.encode(prompt).ids)
    for _ in range(n_tokens):
        scores = predict_next_logits(ids, m['emb'], m['omega'], m['bias'], m['W'], m['config'])
        # Top-k sampling with temperature
        top_idx = np.argpartition(-scores, top_k)[:top_k]
        top_scores = scores[top_idx] / temperature
        # Numerically stable softmax
        top_scores -= top_scores.max()
        probs = np.exp(top_scores)
        probs /= probs.sum()
        chosen = rng.choice(top_idx, p=probs)
        ids.append(int(chosen))
    return tokenizer.decode(ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', nargs='?', default=None)
    parser.add_argument('--n', type=int, default=15)
    parser.add_argument('--temp', type=float, default=0.8)
    parser.add_argument('--topk', type=int, default=40)
    args = parser.parse_args()

    print("Loading tokenizer + model...")
    tokenizer, m = load()
    print(f"Model: V={m['config']['V']}, D={m['config']['D']}, "
          f"CTX={m['config']['CTX']}")
    print(f"Training stats: Top-1 {m['stats']['train_top1']*100:.1f}%, "
          f"Top-5 {m['stats']['train_top5']*100:.1f}%\n")

    if args.prompt:
        # Single prompt
        print(f"Prompt: {args.prompt!r}")
        results, _ = top_k_for_prompt(tokenizer, m, args.prompt, k=10)
        print("\nTop-10 next-token predictions:")
        for i, (idx, tok, score) in enumerate(results, 1):
            print(f"  {i:>2}. {tok!r:<25}  score={score:>7.3f}")
        print(f"\nGreedy continuation ({args.n} tokens):")
        out = generate(tokenizer, m, args.prompt, n_tokens=args.n,
                       temperature=0.1, top_k=5)
        print(f"  {out!r}")
        print(f"\nSampled continuation (temp={args.temp}):")
        out = generate(tokenizer, m, args.prompt, n_tokens=args.n,
                       temperature=args.temp, top_k=args.topk)
        print(f"  {out!r}")
        return

    # Default: run through a suite of test prompts
    test_prompts = [
        "Heute scheint die",
        "Im Anfang war das",
        "Kernel ridge",
        "The eigenvalue of a",
        "Neuronale Netze sind",
        "Die wichtigste Gleichung in der Quantenmechanik",
        "PageRank computes the",
        "Emotionen sind",
        "Music is a form of",
        "Consciousness is",
    ]

    for prompt in test_prompts:
        print(f"=== Prompt: {prompt!r} ===")
        results, _ = top_k_for_prompt(tokenizer, m, prompt, k=5)
        top_toks = [f"{tok!r}" for _, tok, _ in results]
        print(f"  Top-5 next: {', '.join(top_toks)}")
        out = generate(tokenizer, m, prompt, n_tokens=12, temperature=0.6, top_k=20)
        print(f"  Greedy (10 tok): {out!r}")
        print()


if __name__ == '__main__':
    main()

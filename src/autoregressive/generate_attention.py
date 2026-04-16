"""Autoregressive generation from the attention-KRR model (T037 v2)."""
import os, sys, math, pickle, argparse
import numpy as np

DATA_DIR = '/Users/mathiasleonhardt/Dev/krr-chat/data/autoregressive'
TOKENIZER = f'{DATA_DIR}/bpe_tokenizer.json'
MODEL = f'{DATA_DIR}/model_attention.pkl'


def stable_softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def load():
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(TOKENIZER)
    with open(MODEL, 'rb') as f:
        m = pickle.load(f)
    return tokenizer, m


def encode_position(ids, t, m):
    cfg = m['config']
    CTX, EMB_DIM = cfg['CTX'], cfg['EMB_DIM']
    D_K, D_V, N_HEADS, FEAT, V = cfg['D_K'], cfg['D_V'], cfg['N_HEADS'], cfg['FEAT'], cfg['V']
    emb, pe, W_Q, W_K, W_V = m['emb'], m['pe'], m['W_Q'], m['W_K'], m['W_V']

    start = max(0, t - CTX)
    ctx = ids[start:t]
    ctx_len = len(ctx)
    if ctx_len == 0:
        return np.zeros(FEAT, dtype=np.float32)

    E = emb[ctx]
    X = E + pe[-ctx_len:]
    Q = (X @ W_Q).reshape(ctx_len, N_HEADS, D_K)
    K = (X @ W_K).reshape(ctx_len, N_HEADS, D_K)
    Vv = (X @ W_V).reshape(ctx_len, N_HEADS, D_V)
    q = Q[-1]
    scores = np.einsum('hk,nhk->hn', q, K) / math.sqrt(D_K)
    weights = stable_softmax(scores, axis=-1)
    c_heads = np.einsum('hn,nhv->hv', weights, Vv)
    c_attn = c_heads.reshape(-1)
    return np.concatenate([c_attn, E[-1]])


def predict_logits(ids, m):
    D = m['config']['D']
    scale = math.sqrt(2.0 / D)
    vec = encode_position(list(ids), len(ids), m)
    z = scale * np.cos(vec @ m['omega'] + m['bias'])
    return z @ m['W']


def generate(tokenizer, m, prompt, n=15, temperature=0.7, top_k=40, seed=42):
    rng = np.random.default_rng(seed)
    ids = list(tokenizer.encode(prompt).ids)
    for _ in range(n):
        scores = predict_logits(ids, m)
        top_idx = np.argpartition(-scores, top_k)[:top_k]
        top_scores = scores[top_idx] / temperature
        top_scores -= top_scores.max()
        probs = np.exp(top_scores); probs /= probs.sum()
        ids.append(int(rng.choice(top_idx, p=probs)))
    return tokenizer.decode(ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', nargs='?', default=None)
    parser.add_argument('--n', type=int, default=15)
    parser.add_argument('--temp', type=float, default=0.7)
    parser.add_argument('--topk', type=int, default=40)
    args = parser.parse_args()

    print("Loading tokenizer + attention model...")
    tokenizer, m = load()
    cfg = m['config']
    print(f"Model: V={cfg['V']}, D={cfg['D']}, CTX={cfg['CTX']}, "
          f"HEADS={cfg['N_HEADS']}, FEAT={cfg['FEAT']}")
    print(f"Training stats: Top-1 {m['stats']['train_top1']*100:.1f}%, "
          f"Top-5 {m['stats']['train_top5']*100:.1f}%\n")

    prompts = [args.prompt] if args.prompt else [
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

    for prompt in prompts:
        print(f"=== {prompt!r} ===")
        scores = predict_logits(tokenizer.encode(prompt).ids, m)
        top_idx = np.argpartition(-scores, 5)[:5]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        toks = [f"{tokenizer.decode([int(i)])!r}" for i in top_idx]
        print(f"  Top-5 next: {', '.join(toks)}")
        out = generate(tokenizer, m, prompt, n=args.n, temperature=args.temp, top_k=args.topk)
        print(f"  Sampled: {out!r}")
        print()


if __name__ == '__main__':
    main()

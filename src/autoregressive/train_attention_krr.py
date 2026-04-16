"""
T037 follow-up — Autoregressive KRR with random-feature self-attention.

This is the proper test of "can KRR do GPT-style LM if given content-adaptive
context aggregation?". We add causal self-attention with fixed random Q/K/V
projections (no learned parameters, no backprop) between the token embeddings
and the RFF projection.

Pipeline:
  tokens → embeddings + positional encoding
         → random Q/K/V projections
         → causal self-attention (softmax)
         → content-adaptive context vector c_t ∈ R^{d_v}
         → RFF projection to R^D
         → KRR solve: W = (Z^T Z + λI)^-1 Z^T Y
         → predict next token

Everything is a pure matrix operation. No neural network in the sense of
gradient descent / backprop. Only closed-form KRR at the end.

Usage:
  python3.11 src/autoregressive/train_attention_krr.py
"""
import os, sys, time, math, pickle
import numpy as np

sys.path.insert(0, '/Users/mathiasleonhardt/Dev/krr-chat/src')
from solvers import solve as krr_solve

# -------------------- Config --------------------
CTX       = 64          # attention window (causal, most recent 64 tokens)
EMB_DIM   = 64          # token embedding dim (bigger than non-attn variant)
D_K       = 32          # attention key/query dim
D_V       = 64          # attention value dim
N_HEADS   = 4           # multi-head attention (d_k and d_v are per-head)
FEAT      = D_V * N_HEADS + EMB_DIM  # attention output concat + last token emb
D         = 6144        # RFF projection dim
SIGMA     = 2.0
LAMBDA    = 1e-5
SEED      = 42
VALIDATION_FRAC = 0.05
REPEAT = int(os.environ.get('REPEAT', 1))

DATA_DIR = '/Users/mathiasleonhardt/Dev/krr-chat/data/autoregressive'
# Overridable via env vars so one script can run multiple experiments
CORPUS_TXT = os.environ.get('CORPUS',    f'{DATA_DIR}/corpus.txt')
TOKENIZER  = os.environ.get('TOKENIZER', f'{DATA_DIR}/bpe_tokenizer.json')
MODEL_OUT  = os.environ.get('OUTPUT',    f'{DATA_DIR}/model_attention.pkl')
# CG config overrideable (for T039 convergence experiments)
CG_MAX_ITER = int(os.environ.get('CG_MAX_ITER', 200))
CG_TOL      = float(os.environ.get('CG_TOL', 1e-5))

np.random.seed(SEED)


def sinusoidal_positional_encoding(max_len, d_emb):
    """Standard sinusoidal positional encoding (Vaswani et al. 2017)."""
    pe = np.zeros((max_len, d_emb), dtype=np.float32)
    pos = np.arange(max_len)[:, None]
    div = np.exp(np.arange(0, d_emb, 2) * (-math.log(10000.0) / d_emb))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)
    return pe


def stable_softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def main():
    # --- Tokenizer + corpus ---
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(TOKENIZER)
    V = tokenizer.get_vocab_size()
    print(f"Vocab size: {V}")

    with open(CORPUS_TXT) as f:
        corpus_text = f.read()
    token_ids = np.array(tokenizer.encode(corpus_text).ids, dtype=np.int32)
    print(f"Corpus tokens: {len(token_ids):,}")

    # --- Train/val split ---
    n_val = int(len(token_ids) * VALIDATION_FRAC)
    n_train_tokens = len(token_ids) - n_val
    train_ids = token_ids[:n_train_tokens]
    val_ids = token_ids[n_train_tokens:]

    if REPEAT > 1:
        train_ids = np.tile(train_ids, REPEAT)
    N_train = len(train_ids) - CTX
    print(f"Train samples: {N_train:,}, Val tokens: {len(val_ids):,}")
    print(f"Architecture: CTX={CTX} EMB_DIM={EMB_DIM} D_K={D_K} "
          f"D_V={D_V} HEADS={N_HEADS} FEAT={FEAT} D={D}")
    print(f"Expected W: {D * V * 2 / 1024 / 1024:.0f} MB Float16")

    # --- Token embeddings via Word2Vec on BPE stream ---
    print("\nTraining Word2Vec on BPE token stream...")
    from gensim.models import Word2Vec
    CHUNK_LINES = 100
    str_tokens = [str(i) for i in train_ids]
    sentences = [str_tokens[i:i+CHUNK_LINES]
                 for i in range(0, len(str_tokens), CHUNK_LINES)]
    t0 = time.time()
    w2v = Word2Vec(sentences, vector_size=EMB_DIM, window=8, min_count=1,
                   workers=4, sg=0, epochs=15, seed=SEED)
    print(f"  W2V done in {time.time()-t0:.1f}s")
    emb = np.zeros((V, EMB_DIM), dtype=np.float32)
    missing = 0
    for tid in range(V):
        key = str(tid)
        if key in w2v.wv:
            emb[tid] = w2v.wv[key]
        else:
            emb[tid] = np.random.randn(EMB_DIM).astype(np.float32) * 0.01
            missing += 1
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    print(f"  Embedded {V-missing}/{V}, {missing} rare tokens random-init")

    # --- Positional encoding ---
    pe = sinusoidal_positional_encoding(CTX, EMB_DIM)  # (CTX, EMB_DIM)

    # --- Random Q/K/V projections (fixed, not learned) ---
    # Multi-head: W_Q, W_K: (EMB_DIM, N_HEADS * D_K); W_V: (EMB_DIM, N_HEADS * D_V)
    scale_init = 1.0 / math.sqrt(EMB_DIM)
    W_Q = np.random.randn(EMB_DIM, N_HEADS * D_K).astype(np.float32) * scale_init
    W_K = np.random.randn(EMB_DIM, N_HEADS * D_K).astype(np.float32) * scale_init
    W_V = np.random.randn(EMB_DIM, N_HEADS * D_V).astype(np.float32) * scale_init

    # --- RFF projection ---
    omega = np.random.randn(FEAT, D).astype(np.float32) / SIGMA
    bias  = np.random.rand(D).astype(np.float32) * 2 * np.pi
    rff_scale = math.sqrt(2.0 / D)

    # --- Context encoder with attention ---
    def encode_position(ids, t):
        """Encode the context ids[t-CTX:t] with multi-head self-attention.
        Query = last position (the position just before t, which we're about
        to predict). Keys/values = all context positions. Result: single
        feature vector of dimension FEAT = N_HEADS*D_V + EMB_DIM.
        """
        start = max(0, t - CTX)
        ctx = ids[start:t]
        ctx_len = len(ctx)
        if ctx_len == 0:
            return np.zeros(FEAT, dtype=np.float32)

        # Embeddings + positional encoding (right-align in the CTX window)
        E = emb[ctx]                              # (ctx_len, EMB_DIM)
        X = E + pe[-ctx_len:]                     # (ctx_len, EMB_DIM)

        # Q/K/V projections (multi-head reshape)
        Q = (X @ W_Q).reshape(ctx_len, N_HEADS, D_K)   # (ctx_len, H, d_k)
        K = (X @ W_K).reshape(ctx_len, N_HEADS, D_K)
        V_ = (X @ W_V).reshape(ctx_len, N_HEADS, D_V)

        # Query from the last context position, across all heads
        q = Q[-1]                                  # (H, d_k)
        # Scores: for each head, (d_k)·(ctx_len, d_k)^T = (ctx_len,)
        # scores shape: (H, ctx_len)
        scores = np.einsum('hk,nhk->hn', q, K) / math.sqrt(D_K)
        weights = stable_softmax(scores, axis=-1)  # (H, ctx_len)
        # Weighted sum of values per head → (H, D_V)
        c_heads = np.einsum('hn,nhv->hv', weights, V_)
        c_attn = c_heads.reshape(-1)              # (N_HEADS * D_V,)

        # Feature = attention output ⊕ last token embedding
        return np.concatenate([c_attn, E[-1]])    # (FEAT,)

    # --- Streaming accumulation ---
    print("\nStreaming Z^T Z + Z^T Y accumulation...")
    t0 = time.time()
    ZtZ = np.zeros((D, D), dtype=np.float64)
    ZtY = np.zeros((D, V), dtype=np.float64)
    CHUNK = 2000  # smaller because encoding is more work per sample

    for start in range(0, N_train, CHUNK):
        end = min(start + CHUNK, N_train)
        size = end - start
        Fmat = np.zeros((size, FEAT), dtype=np.float32)
        targets = np.zeros(size, dtype=np.int32)
        for j in range(size):
            i = start + j
            Fmat[j] = encode_position(train_ids, i + CTX)
            targets[j] = train_ids[i + CTX]
        Z = rff_scale * np.cos(Fmat @ omega + bias[None, :])
        Z64 = Z.astype(np.float64)
        ZtZ += Z64.T @ Z64
        for j in range(size):
            ZtY[:, targets[j]] += Z64[j]
        del Fmat, Z, Z64
        if start % 10000 == 0:
            elapsed = time.time() - t0
            pct = 100 * start / N_train
            print(f"  {start:>7}/{N_train:,} ({pct:5.1f}%)  elapsed: {elapsed:5.1f}s")

    ZtZ += LAMBDA * np.eye(D)
    t_accum = time.time() - t0
    print(f"Accumulation: {t_accum:.1f}s")

    # --- Solve ---
    print("\nSolving via Block-PCG...")
    t0 = time.time()
    W64, info = krr_solve(ZtZ, ZtY, solver='cg',
                          tol=CG_TOL, max_iter=CG_MAX_ITER,
                          preconditioner='diagonal', verbose=False)
    t_solve = time.time() - t0
    W = W64.astype(np.float32)
    print(f"  {info}")
    print(f"  solve: {t_solve:.1f}s")
    del ZtZ, ZtY

    # --- Evaluation ---
    print("\nEvaluation...")
    sample_ids = np.random.choice(N_train, min(5000, N_train), replace=False)
    top1 = top5 = 0
    for i in sample_ids:
        fvec = encode_position(train_ids, i + CTX)
        z = rff_scale * np.cos(fvec @ omega + bias)
        scores = z @ W
        pred = np.argmax(scores)
        target = train_ids[i + CTX]
        if pred == target: top1 += 1
        if target in np.argpartition(-scores, 5)[:5]: top5 += 1
    print(f"  Train Top-1: {top1/len(sample_ids)*100:.1f}%, "
          f"Top-5: {top5/len(sample_ids)*100:.1f}%")

    # Validation
    N_val = len(val_ids) - CTX
    if N_val > 0:
        n_eval = min(5000, N_val)
        vtop1 = vtop5 = 0
        for i in range(n_eval):
            if i < CTX:
                ctx = np.concatenate([train_ids[-(CTX-i):], val_ids[:i]])
            else:
                ctx = val_ids[i-CTX:i]
            # Run encode_position on a synthetic stream
            tmp = np.concatenate([ctx, [0]])  # placeholder for t
            fvec = encode_position(tmp, CTX)
            z = rff_scale * np.cos(fvec @ omega + bias)
            scores = z @ W
            pred = np.argmax(scores)
            target = val_ids[i]
            if pred == target: vtop1 += 1
            if target in np.argpartition(-scores, 5)[:5]: vtop5 += 1
        print(f"  Val   Top-1: {vtop1/n_eval*100:.1f}%, "
              f"Top-5: {vtop5/n_eval*100:.1f}%")

    # --- Save ---
    with open(MODEL_OUT, 'wb') as f:
        pickle.dump({
            'W': W, 'omega': omega, 'bias': bias, 'emb': emb,
            'pe': pe, 'W_Q': W_Q, 'W_K': W_K, 'W_V': W_V,
            'config': {
                'CTX': CTX, 'EMB_DIM': EMB_DIM, 'D_K': D_K, 'D_V': D_V,
                'N_HEADS': N_HEADS, 'FEAT': FEAT, 'D': D,
                'SIGMA': SIGMA, 'LAMBDA': LAMBDA, 'V': V,
            },
            'stats': {
                'train_top1': top1/len(sample_ids),
                'train_top5': top5/len(sample_ids),
                't_accum': t_accum, 't_solve': t_solve,
            },
        }, f)
    print(f"\nSaved to {MODEL_OUT}")


if __name__ == '__main__':
    main()

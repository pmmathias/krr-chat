"""
T040 — Multi-layer attention with random (fixed) Q/K/V per layer.

Architecture:
  x₀ = embedding(tokens) + sinusoidal_PE
  for layer ℓ in 1..L:
      a_ℓ   = MultiHeadAttn_ℓ(x_{ℓ-1}, W_Q_ℓ, W_K_ℓ, W_V_ℓ)   (causal)
      x_ℓ'  = LayerNorm(x_{ℓ-1} + a_ℓ)                          (residual + norm)
      f_ℓ   = FixedFFN_ℓ(x_ℓ', W_in_ℓ, W_out_ℓ)                  (2-layer ReLU-MLP)
      x_ℓ   = LayerNorm(x_ℓ' + f_ℓ)
  c_t = x_L[-1]                 (final state at last context position)
  z_t = √(2/D) cos(c_t ω + b)
  predict next token = argmax(W · z_t)    (KRR, closed-form)

All W_Q, W_K, W_V, W_in, W_out are RANDOM, FIXED at init.
The only learned parameter is W (the final KRR readout).

Env vars:
  CORPUS, TOKENIZER, OUTPUT as before
  N_LAYERS = 2 or 3 (default 2)
  CG_MAX_ITER, CG_TOL (default 1000, 1e-5)
"""
import os, sys, time, math, pickle
import numpy as np

sys.path.insert(0, '/Users/mathiasleonhardt/Dev/krr-chat/src')
from solvers import solve as krr_solve

# -------------------- Config --------------------
CTX       = 64
EMB_DIM   = 64
D_K       = 32
D_V       = 64
N_HEADS   = 4
D_FF      = 256                       # FFN hidden dim
N_LAYERS  = int(os.environ.get('N_LAYERS', 2))

# Feature dim: final layer's output dimension = EMB_DIM (after residual)
# + final-token embedding → 2 * EMB_DIM
FEAT      = 2 * EMB_DIM
D         = 6144
SIGMA     = 2.0
LAMBDA    = 1e-5
SEED      = 42
VALIDATION_FRAC = 0.05

DATA_DIR   = '/Users/mathiasleonhardt/Dev/krr-chat/data/autoregressive'
CORPUS_TXT = os.environ.get('CORPUS',    f'{DATA_DIR}/corpus_clean.txt')
TOKENIZER  = os.environ.get('TOKENIZER', f'{DATA_DIR}/bpe_tokenizer_clean.json')
MODEL_OUT  = os.environ.get('OUTPUT',    f'{DATA_DIR}/model_multilayer.pkl')
CG_MAX_ITER = int(os.environ.get('CG_MAX_ITER', 1000))
CG_TOL      = float(os.environ.get('CG_TOL', 1e-5))

np.random.seed(SEED)


def sinusoidal_pe(max_len, d_emb):
    pe = np.zeros((max_len, d_emb), dtype=np.float32)
    pos = np.arange(max_len)[:, None]
    div = np.exp(np.arange(0, d_emb, 2) * (-math.log(10000.0) / d_emb))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)
    return pe


def layer_norm(x, eps=1e-5):
    """Per-row layer normalization (no learned affine)."""
    m = x.mean(axis=-1, keepdims=True)
    v = x.var(axis=-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps)


def stable_softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def make_layer_params(seed_offset=0):
    """Random Q/K/V + FFN matrices for one attention layer."""
    rng = np.random.default_rng(SEED + seed_offset)
    si = 1.0 / math.sqrt(EMB_DIM)
    W_Q  = rng.standard_normal((EMB_DIM, N_HEADS * D_K), dtype=np.float32) * si
    W_K  = rng.standard_normal((EMB_DIM, N_HEADS * D_K), dtype=np.float32) * si
    W_V  = rng.standard_normal((EMB_DIM, N_HEADS * D_V), dtype=np.float32) * si
    # Output projection: concat heads back to EMB_DIM
    W_O  = rng.standard_normal((N_HEADS * D_V, EMB_DIM), dtype=np.float32) / math.sqrt(N_HEADS * D_V)
    # FFN: (EMB_DIM → D_FF → EMB_DIM)
    W_in  = rng.standard_normal((EMB_DIM, D_FF), dtype=np.float32) / math.sqrt(EMB_DIM)
    W_out = rng.standard_normal((D_FF, EMB_DIM), dtype=np.float32) / math.sqrt(D_FF)
    return dict(W_Q=W_Q, W_K=W_K, W_V=W_V, W_O=W_O, W_in=W_in, W_out=W_out)


def attention_layer(X, params):
    """Causal self-attention + residual + layernorm, then FFN + residual + layernorm.
    X: (ctx_len, EMB_DIM) — the full context's layer input
    returns: (ctx_len, EMB_DIM)
    """
    ctx_len, _ = X.shape
    Q = (X @ params['W_Q']).reshape(ctx_len, N_HEADS, D_K)
    K = (X @ params['W_K']).reshape(ctx_len, N_HEADS, D_K)
    V = (X @ params['W_V']).reshape(ctx_len, N_HEADS, D_V)

    # Causal self-attention: position i attends only to positions ≤ i
    # scores: (ctx_len, H, ctx_len)
    scores = np.einsum('ihk,jhk->hij', Q, K) / math.sqrt(D_K)  # (H, ctx_len, ctx_len)
    # Causal mask
    mask = np.triu(np.ones((ctx_len, ctx_len), dtype=bool), k=1)
    scores[:, mask] = -np.inf
    weights = stable_softmax(scores, axis=-1)                  # (H, ctx_len, ctx_len)
    # Aggregate values per head
    out = np.einsum('hij,jhv->ihv', weights, V)                # (ctx_len, H, D_V)
    out = out.reshape(ctx_len, N_HEADS * D_V)                  # (ctx_len, H*D_V)
    out = out @ params['W_O']                                  # (ctx_len, EMB_DIM)

    X1 = layer_norm(X + out)                                   # residual + LN
    # FFN
    h = X1 @ params['W_in']                                    # (ctx_len, D_FF)
    h = np.maximum(h, 0.0)                                     # ReLU
    f = h @ params['W_out']                                    # (ctx_len, EMB_DIM)
    X2 = layer_norm(X1 + f)                                    # residual + LN
    return X2


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

    n_val = int(len(token_ids) * VALIDATION_FRAC)
    train_ids = token_ids[:len(token_ids) - n_val]
    val_ids = token_ids[len(token_ids) - n_val:]
    N_train = len(train_ids) - CTX

    print(f"Architecture: L={N_LAYERS} x (MHA H={N_HEADS} dk={D_K} dv={D_V} + FFN {EMB_DIM}->{D_FF}->{EMB_DIM})")
    print(f"FEAT (final x_t ⊕ final-token-emb) = {FEAT}")
    print(f"D (RFF) = {D}, N_train = {N_train:,}")

    # --- Token embeddings (Word2Vec on BPE stream) ---
    print("\nTraining Word2Vec on BPE token stream...")
    from gensim.models import Word2Vec
    CHUNK_LINES = 100
    str_tokens = [str(i) for i in train_ids]
    sentences = [str_tokens[i:i+CHUNK_LINES]
                 for i in range(0, len(str_tokens), CHUNK_LINES)]
    w2v = Word2Vec(sentences, vector_size=EMB_DIM, window=8, min_count=1,
                   workers=4, sg=0, epochs=15, seed=SEED)
    emb = np.zeros((V, EMB_DIM), dtype=np.float32)
    for tid in range(V):
        key = str(tid)
        if key in w2v.wv:
            emb[tid] = w2v.wv[key]
        else:
            emb[tid] = np.random.randn(EMB_DIM).astype(np.float32) * 0.01
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)

    # --- Layer parameters (random, fixed) ---
    layers = [make_layer_params(seed_offset=10*ℓ) for ℓ in range(N_LAYERS)]
    pe = sinusoidal_pe(CTX, EMB_DIM)

    # --- RFF projection ---
    omega = np.random.randn(FEAT, D).astype(np.float32) / SIGMA
    bias  = np.random.rand(D).astype(np.float32) * 2 * np.pi
    rff_scale = math.sqrt(2.0 / D)

    # --- Encoder: run through all layers, return final state at last position ---
    def encode_position(ids, t):
        start = max(0, t - CTX)
        ctx = ids[start:t]
        ctx_len = len(ctx)
        if ctx_len == 0:
            return np.zeros(FEAT, dtype=np.float32)

        # Initial: embeddings + PE (right-aligned in the CTX window)
        E = emb[ctx]                                # (ctx_len, EMB_DIM)
        X = E + pe[-ctx_len:]                       # (ctx_len, EMB_DIM)

        # Run through layers
        for params in layers:
            X = attention_layer(X, params)

        # Feature = final state at last position ⊕ last-token embedding
        return np.concatenate([X[-1], E[-1]])

    # --- Streaming accumulation ---
    print("\nAccumulating Z^T Z + Z^T Y (streaming)...")
    t0 = time.time()
    ZtZ = np.zeros((D, D), dtype=np.float64)
    ZtY = np.zeros((D, V), dtype=np.float64)
    CHUNK = 1000  # smaller chunk — each sample is heavier with multi-layer

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
        if start % 5000 == 0:
            pct = 100 * start / N_train
            print(f"  {start:>7}/{N_train:,} ({pct:5.1f}%)  elapsed: {time.time()-t0:5.1f}s")

    ZtZ += LAMBDA * np.eye(D)
    t_accum = time.time() - t0
    print(f"Accumulation: {t_accum:.1f}s")

    # --- Solve ---
    print("Solving via Block-PCG...")
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
    print("Evaluation...")
    sample_ids = np.random.choice(N_train, min(3000, N_train), replace=False)
    top1 = top5 = 0
    for i in sample_ids:
        fvec = encode_position(train_ids, i + CTX)
        z = rff_scale * np.cos(fvec @ omega + bias)
        scores = z @ W
        pred = np.argmax(scores)
        target = train_ids[i + CTX]
        if pred == target: top1 += 1
        if target in np.argpartition(-scores, 5)[:5]: top5 += 1
    train_t1 = top1/len(sample_ids); train_t5 = top5/len(sample_ids)
    print(f"  Train Top-1: {train_t1*100:.1f}%, Top-5: {train_t5*100:.1f}%")

    N_val = len(val_ids) - CTX
    val_t1 = val_t5 = 0.0
    if N_val > 0:
        vtop1 = vtop5 = 0
        n_eval = min(3000, N_val)
        for i in range(n_eval):
            if i < CTX:
                ctx = np.concatenate([train_ids[-(CTX-i):], val_ids[:i]])
            else:
                ctx = val_ids[i-CTX:i]
            tmp = np.concatenate([ctx, [0]])
            fvec = encode_position(tmp, CTX)
            z = rff_scale * np.cos(fvec @ omega + bias)
            scores = z @ W
            pred = np.argmax(scores)
            target = val_ids[i]
            if pred == target: vtop1 += 1
            if target in np.argpartition(-scores, 5)[:5]: vtop5 += 1
        val_t1 = vtop1/n_eval; val_t5 = vtop5/n_eval
        print(f"  Val   Top-1: {val_t1*100:.1f}%, Top-5: {val_t5*100:.1f}%")

    # --- Save ---
    with open(MODEL_OUT, 'wb') as f:
        pickle.dump({
            'W': W, 'omega': omega, 'bias': bias, 'emb': emb, 'pe': pe,
            'layers': layers,
            'config': {
                'CTX': CTX, 'EMB_DIM': EMB_DIM, 'D_K': D_K, 'D_V': D_V,
                'N_HEADS': N_HEADS, 'N_LAYERS': N_LAYERS, 'D_FF': D_FF,
                'FEAT': FEAT, 'D': D, 'SIGMA': SIGMA, 'LAMBDA': LAMBDA, 'V': V,
            },
            'stats': {
                'train_top1': train_t1, 'train_top5': train_t5,
                'val_top1': val_t1, 'val_top5': val_t5,
                't_accum': t_accum, 't_solve': t_solve,
                'cg_info': info,
            },
        }, f)
    print(f"\nSaved to {MODEL_OUT}")


if __name__ == '__main__':
    main()

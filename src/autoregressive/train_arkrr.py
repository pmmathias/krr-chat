"""
T037 — Train autoregressive KRR language model.

Architecture:
  - BPE tokenizer (V=8192) trained on the blog corpus
  - Fixed context window CTX=64 tokens (Kalle-style with BPE)
  - Token embeddings: randomly initialized (EMB_DIM=32, fixed, not learned)
  - Position-weighted context vector
  - RFF projection (D=6144, σ=1.5)
  - KRR solve (Block-PCG): W = (Z^T Z + λI)^-1 Z^T Y, Y = one-hot next-token

Next-token prediction: given a prompt, encode last CTX tokens → z → argmax(z·W).

Usage:
  python3.11 src/autoregressive/train_arkrr.py
"""
import os, sys, time, math, json, pickle
import numpy as np

sys.path.insert(0, '/Users/mathiasleonhardt/Dev/krr-chat/src')
from solvers import solve as krr_solve

# -------------------- Config --------------------
CTX = 64
EMB_DIM = 32
FEAT = CTX * EMB_DIM        # 2048
D = 6144
SIGMA = 2.0                 # slightly wider kernel than Kalle because FEAT is bigger
LAMBDA = 1e-5
SEED = 42
VALIDATION_FRAC = 0.05
REPEAT = int(os.environ.get('REPEAT', 1))   # 1 = true autoregressive, higher = Kalle-style memorization

DATA_DIR = '/Users/mathiasleonhardt/Dev/krr-chat/data/autoregressive'
CORPUS_TXT = f'{DATA_DIR}/corpus.txt'
TOKENIZER = f'{DATA_DIR}/bpe_tokenizer.json'
MODEL_OUT = f'{DATA_DIR}/model.pkl'

np.random.seed(SEED)


def main():
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(TOKENIZER)
    V = tokenizer.get_vocab_size()
    print(f"Vocab size: {V}")

    with open(CORPUS_TXT) as f:
        corpus_text = f.read()
    token_ids = np.array(tokenizer.encode(corpus_text).ids, dtype=np.int32)
    print(f"Corpus tokens: {len(token_ids):,}")

    # ---------- Train/val split ----------
    n_val = int(len(token_ids) * VALIDATION_FRAC)
    n_train_tokens = len(token_ids) - n_val
    train_ids = token_ids[:n_train_tokens]
    val_ids = token_ids[n_train_tokens:]
    print(f"Train: {len(train_ids):,} tokens, Val: {len(val_ids):,} tokens")

    # Apply REPEAT (replicate the training stream) — same lever as Kalle's REPEAT=5
    if REPEAT > 1:
        train_ids = np.tile(train_ids, REPEAT)
        print(f"REPEAT={REPEAT}, expanded train tokens: {len(train_ids):,}")
    N_train = len(train_ids) - CTX
    print(f"N_train samples: {N_train:,}")
    print(f"V (vocabulary): {V}")
    print(f"D (RFF dim):    {D}")
    print(f"FEAT (CTX×EMB): {FEAT}")
    print(f"Expected W size: {D * V * 2 / 1024 / 1024:.0f} MB Float16")

    # ---------- Word2Vec-style token embeddings (Kalle-style, not random) ----------
    # Random embeddings give the model zero semantic structure, so it just
    # learns the marginal (predict '.', ',', 'the'). Word2Vec on the BPE stream
    # places semantically similar tokens near each other in embedding space.
    print("\nTraining Word2Vec on BPE token stream (Kalle-style)...")
    from gensim.models import Word2Vec
    # Gensim expects lists of string tokens; we use the BPE ids as strings.
    # Chunk the stream into lines of ~100 tokens for W2V's window mechanism.
    CHUNK_LINES = 100
    str_tokens = [str(i) for i in train_ids]
    sentences = [str_tokens[i:i+CHUNK_LINES]
                 for i in range(0, len(str_tokens), CHUNK_LINES)]
    t_w2v = time.time()
    w2v = Word2Vec(sentences, vector_size=EMB_DIM, window=8, min_count=1,
                   workers=4, sg=0, epochs=15, seed=SEED)
    print(f"  Word2Vec trained in {time.time()-t_w2v:.1f}s")

    emb = np.zeros((V, EMB_DIM), dtype=np.float32)
    missing = 0
    for tid in range(V):
        key = str(tid)
        if key in w2v.wv:
            emb[tid] = w2v.wv[key]
        else:
            emb[tid] = np.random.randn(EMB_DIM).astype(np.float32) * 0.01
            missing += 1
    # Normalize each row
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / (norms + 1e-8)
    print(f"  Embedded {V-missing}/{V} tokens via W2V, {missing} rare/unseen got random init")

    # ---------- RFF projection ----------
    print(f"RFF projection (FEAT={FEAT}, D={D}, σ={SIGMA})...")
    omega = np.random.randn(FEAT, D).astype(np.float32) / SIGMA
    bias = np.random.rand(D).astype(np.float32) * 2 * np.pi
    scale = math.sqrt(2.0 / D)

    # ---------- Context encoding ----------
    # Position-weighted sum of last CTX embeddings (Kalle-style)
    pos_weights = np.linspace(0.4, 1.0, CTX, dtype=np.float32)  # 0.4..1.0

    def encode_position(ids, t):
        """Encode the context preceding position t."""
        # Last CTX tokens before position t
        start = max(0, t - CTX)
        ctx = ids[start:t]
        ctx_len = len(ctx)
        vec = np.zeros(FEAT, dtype=np.float32)
        # Place tokens right-aligned so the most recent is at the end
        for i in range(ctx_len):
            slot = CTX - ctx_len + i
            token_id = ctx[i]
            if 0 <= token_id < V:
                vec[slot*EMB_DIM:(slot+1)*EMB_DIM] = pos_weights[slot] * emb[token_id]
        return vec

    # ---------- Streaming Z^T Z + Z^T Y accumulation ----------
    print("\nStreaming Z^T Z + Z^T Y accumulation...")
    t0 = time.time()
    ZtZ = np.zeros((D, D), dtype=np.float64)
    ZtY = np.zeros((D, V), dtype=np.float64)
    CHUNK = 5000  # smaller chunk because each sample is bigger (CTX=64)

    for start in range(0, N_train, CHUNK):
        end = min(start + CHUNK, N_train)
        size = end - start
        F = np.zeros((size, FEAT), dtype=np.float32)
        targets = np.zeros(size, dtype=np.int32)
        for j in range(size):
            i = start + j
            F[j] = encode_position(train_ids, i + CTX)
            targets[j] = train_ids[i + CTX]
        Z = scale * np.cos(F @ omega + bias[None, :])
        Z64 = Z.astype(np.float64)
        ZtZ += Z64.T @ Z64
        for j in range(size):
            ZtY[:, targets[j]] += Z64[j]
        del F, Z, Z64
        if start % 25000 == 0:
            elapsed = time.time() - t0
            pct = 100 * start / N_train
            print(f"  {start:>7}/{N_train:,} ({pct:5.1f}%)  elapsed: {elapsed:5.1f}s")

    ZtZ += LAMBDA * np.eye(D)
    t_accum = time.time() - t0
    print(f"Accumulation: {t_accum:.1f}s, ZtZ mem: {ZtZ.nbytes/1024/1024:.0f} MB")

    # ---------- Solve ----------
    print(f"\nSolving via Block-PCG...")
    t0 = time.time()
    W64, solver_info = krr_solve(
        ZtZ, ZtY, solver='cg',
        tol=1e-5, max_iter=200, preconditioner='diagonal', verbose=False,
    )
    t_solve = time.time() - t0
    W = W64.astype(np.float32)
    print(f"  {solver_info}")
    print(f"  solve: {t_solve:.1f}s")
    del ZtZ, ZtY

    # ---------- Training accuracy on a sample ----------
    print("\nTop-1 / Top-5 on 5000 random training positions...")
    sample_ids = np.random.choice(N_train, min(5000, N_train), replace=False)
    top1 = top5 = 0
    for i in sample_ids:
        f = encode_position(train_ids, i + CTX)
        z = scale * np.cos(f @ omega + bias)
        scores = z @ W
        pred = np.argmax(scores)
        target = train_ids[i + CTX]
        if pred == target: top1 += 1
        if target in np.argpartition(-scores, 5)[:5]: top5 += 1
    top1_pct = top1 / len(sample_ids)
    top5_pct = top5 / len(sample_ids)
    print(f"  Top-1: {top1_pct*100:.1f}%, Top-5: {top5_pct*100:.1f}%")

    # ---------- Validation (held-out) ----------
    print("\nValidation (held-out tail)...")
    N_val = len(val_ids) - CTX
    if N_val > 0:
        val_top1 = val_top5 = 0
        for i in range(min(5000, N_val)):
            # Context can span the boundary between train and val
            if i < CTX:
                ctx = np.concatenate([train_ids[-(CTX-i):], val_ids[:i]])
            else:
                ctx = val_ids[i-CTX:i]
            f = np.zeros(FEAT, dtype=np.float32)
            for k, tid in enumerate(ctx):
                if 0 <= tid < V:
                    f[k*EMB_DIM:(k+1)*EMB_DIM] = pos_weights[k] * emb[tid]
            z = scale * np.cos(f @ omega + bias)
            scores = z @ W
            pred = np.argmax(scores)
            target = val_ids[i]
            if pred == target: val_top1 += 1
            if target in np.argpartition(-scores, 5)[:5]: val_top5 += 1
        n_eval = min(5000, N_val)
        print(f"  Val Top-1: {val_top1/n_eval*100:.1f}%, Top-5: {val_top5/n_eval*100:.1f}%")

    # ---------- Save the model ----------
    print(f"\nSaving model...")
    with open(MODEL_OUT, 'wb') as f:
        pickle.dump({
            'W': W,
            'omega': omega,
            'bias': bias,
            'emb': emb,
            'config': {
                'CTX': CTX, 'EMB_DIM': EMB_DIM, 'FEAT': FEAT,
                'D': D, 'SIGMA': SIGMA, 'LAMBDA': LAMBDA, 'V': V,
            },
            'stats': {
                'N_train': N_train,
                'train_top1': top1_pct,
                'train_top5': top5_pct,
                't_accum': t_accum,
                't_solve': t_solve,
            },
        }, f)
    print(f"  Saved to {MODEL_OUT}")


if __name__ == '__main__':
    main()

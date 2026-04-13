"""
Build kalle-chat.html — the complete KRR language model as a self-contained HTML file.

Pipeline:
  1. Parse corpus → list of (user, bot) pairs
  2. Build full token sequence (5x repeat) for KRR training
  3. Train Word2Vec (gensim) on tokens → 32-dim embeddings
  4. Generate ω/bias (Random Fourier Features, fixed seed)
  5. Encode contexts → Z (RFF features), streaming to avoid OOM
  6. Build Y (one-hot next-word targets)
  7. Solve W = (Z^TZ + λI)^-1 Z^T Y  (closed-form KRR, no gradient descent)
  8. Compute IDF over pairs
  9. Compute BoW+IDF pair embeddings (32-dim sentence vectors)
  10. Pack everything as Float16+gzip+base64
  11. Inject into HTML template

Hyperparameters:
  CTX=24, EMB_DIM=32, FEAT=768, D=6144, σ=1.5, λ=1e-6

Usage:
  python3 src/build.py [--corpus data/corpus.md] [--template data/template.html]
                       [--chunks data/chunk_index.json] [--output kalle-chat.html]
"""
import re, json, base64, gzip, math, time, os, argparse
import numpy as np
from gensim.models import Word2Vec

# --- CLI args with defaults pointing to repo structure ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parser = argparse.ArgumentParser(description='Build KRR Chat HTML')
parser.add_argument('--corpus', default=os.path.join(ROOT, 'data', 'corpus.md'))
parser.add_argument('--template', default=os.path.join(ROOT, 'data', 'template.html'))
parser.add_argument('--chunks', default=os.path.join(ROOT, 'data', 'chunk_index.json'))
parser.add_argument('--output', default=os.path.join(ROOT, 'kalle-chat.html'))
args = parser.parse_args()

# ==================================================================
# Config (must match kalle-tfidf.html JS)
# ==================================================================
CTX = 24
EMB_DIM = 32
FEAT = CTX * EMB_DIM   # 768
D = 6144
SIGMA = 1.5
LAMBDA = 1e-6
REPEAT = 5
SEED = 42
np.random.seed(SEED)

# ==================================================================
# Step 1: Parse corpus
# ==================================================================
print("Step 1: Parse corpus")
print(f"  source: {args.corpus}")
with open(args.corpus) as f:
    txt = f.read()

pairs = []
for line in txt.splitlines():
    line = line.strip()
    if not line.startswith('du:'): continue
    if ' bot: ' not in line: continue
    user, bot = line.split(' bot: ', 1)
    user = user.replace('du:', '').strip().rstrip(' .').strip()
    bot = bot.strip().rstrip(' .').strip()
    # Ensure bot response ends with sentence terminator
    if bot and bot.split()[-1] not in ['.','?','!']:
        bot = bot + ' .'
    if user and bot:
        pairs.append((user, bot))
print(f"  {len(pairs)} pairs")

# ==================================================================
# Step 2: Build full token sequence with turn markers
# ==================================================================
print("Step 2: Build training token sequence")
tokens = []
for u, b in pairs:
    tokens.append('du:')
    for w in u.split():
        tokens.append(w)
    tokens.append('.')
    tokens.append('bot:')
    for w in b.split():
        tokens.append(w)
    tokens.append('.')
print(f"  base tokens: {len(tokens)}")

# 5x repeat
tokens_full = tokens * REPEAT
print(f"  with {REPEAT}x repeat: {len(tokens_full)}")

# Vocab
vocab_set = set(tokens_full)
vocab = sorted(vocab_set, key=lambda w: -tokens_full.count(w))[:1000]  # top by freq
# Actually use all unique
vocab = sorted(vocab_set)
W2I = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
print(f"  vocab: {V}")

# ==================================================================
# Step 3: Train Word2Vec
# ==================================================================
print(f"Step 3: Train Word2Vec (dim={EMB_DIM})")
# Treat each pair as a "sentence" for word2vec
sentences = []
for u, b in pairs:
    sentences.append(['du:'] + u.split() + ['.', 'bot:'] + b.split() + ['.'])
# Repeat for stronger signal
sentences = sentences * 3

t0 = time.time()
w2v = Word2Vec(sentences, vector_size=EMB_DIM, window=8, min_count=1, workers=4, sg=0, epochs=20, seed=SEED)
print(f"  trained in {time.time()-t0:.1f}s")

# Build embedding matrix in vocab order
emb_matrix = np.zeros((V, EMB_DIM), dtype=np.float32)
for w, i in W2I.items():
    if w in w2v.wv:
        emb_matrix[i] = w2v.wv[w]
    else:
        emb_matrix[i] = np.random.randn(EMB_DIM).astype(np.float32) * 0.01
print(f"  emb shape: {emb_matrix.shape}")

# ==================================================================
# Step 4: Generate omega/bias (random Fourier features)
# ==================================================================
print(f"Step 4: Generate omega/bias (FEAT={FEAT}, D={D}, sigma={SIGMA})")
omega = np.random.randn(FEAT, D).astype(np.float32) / SIGMA
bias = (np.random.rand(D).astype(np.float32) * 2 * np.pi)
print(f"  omega: {omega.shape}, bias: {bias.shape}")

# ==================================================================
# Step 5+6+7: STREAMING encode → ZtZ + ZtY accumulator (no full Z stored)
# ==================================================================
print("Step 5+6+7: Streaming encode + accumulate ZtZ and ZtY (chunked)")

def encode_ctx(ctx_words):
    vec = np.zeros(FEAT, dtype=np.float32)
    for p in range(CTX):
        w = ctx_words[p] if p < len(ctx_words) else ''
        if not w or w not in W2I: continue
        wt = 0.4 + 0.6 * (p / (CTX - 1))
        vec[p*EMB_DIM:(p+1)*EMB_DIM] = wt * emb_matrix[W2I[w]]
    return vec

N_train = len(tokens_full) - CTX
print(f"  N_train: {N_train}")
scale = math.sqrt(2.0 / D)

# Build Y_idx (target class per sample) — small
Y_idx = np.zeros(N_train, dtype=np.int32)
for i in range(N_train):
    Y_idx[i] = W2I[tokens_full[i+CTX]]
print(f"  Y_idx ready: {Y_idx.shape}")

# Streaming chunks: never materialize full F or Z
ZtZ = np.zeros((D, D), dtype=np.float64)
ZtY = np.zeros((D, V), dtype=np.float64)
CHUNK = 10000
for start in range(0, N_train, CHUNK):
    end = min(start + CHUNK, N_train)
    size = end - start
    # Encode features for this chunk
    F_chunk = np.zeros((size, FEAT), dtype=np.float32)
    for j in range(size):
        i = start + j
        F_chunk[j] = encode_ctx(tokens_full[i:i+CTX])
    # RFF projection
    Z_chunk = scale * np.cos(F_chunk @ omega + bias[None, :])
    del F_chunk
    # Accumulate ZtZ (float64)
    Z64 = Z_chunk.astype(np.float64)
    ZtZ += Z64.T @ Z64
    # Accumulate ZtY
    for j in range(size):
        ZtY[:, Y_idx[start + j]] += Z64[j]
    del Z_chunk, Z64
    if start % 30000 == 0:
        print(f"    {start}/{N_train}")
print(f"  ZtZ: {ZtZ.shape}, mem ~{ZtZ.nbytes/1024/1024:.1f} MB")
print(f"  ZtY: {ZtY.shape}")

# Solve
ZtZ += LAMBDA * np.eye(D)
print("  solving linear system...")
t0 = time.time()
W = np.linalg.solve(ZtZ, ZtY).astype(np.float32)
print(f"  solved in {time.time()-t0:.1f}s, W: {W.shape}")
del ZtZ, ZtY

# ==================================================================
# Step 8: Top-1 accuracy on sample subset (avoid full Z recompute)
# ==================================================================
print("Step 8: Top-1 accuracy (subsampled to 10000 random positions)")
sample_ids = np.random.choice(N_train, min(10000, N_train), replace=False)
correct = 0
for idx, i in enumerate(sample_ids):
    f = encode_ctx(tokens_full[i:i+CTX])
    z = scale * np.cos(f @ omega + bias)
    pred = np.argmax(z @ W)
    if pred == Y_idx[i]: correct += 1
top1 = correct / len(sample_ids)
print(f"  Top-1 (sample): {top1*100:.1f}%")

# ==================================================================
# Step 9: Compute IDF over pairs (not over all tokens)
# ==================================================================
print("Step 9: Compute IDF")
df = np.zeros(V, dtype=np.float64)
for u, b in pairs:
    seen = set(u.split() + b.split())
    for w in seen:
        if w in W2I:
            df[W2I[w]] += 1
N_DOCS = len(pairs)
idf = np.log((N_DOCS + 1) / (df + 1)) + 1.0
idf = idf / idf.mean()
print(f"  IDF: range [{idf.min():.3f}, {idf.max():.3f}]")

# ==================================================================
# Step 10: BoW+IDF pair embeddings (32-dim)
# ==================================================================
print("Step 10: BoW+IDF pair embeddings")
def bow_emb(words):
    vec = np.zeros(EMB_DIM, dtype=np.float32)
    for w in words:
        if w in W2I:
            vec += idf[W2I[w]] * emb_matrix[W2I[w]]
    n = np.linalg.norm(vec) + 1e-10
    return vec / n

pair_embs = np.zeros((len(pairs), EMB_DIM), dtype=np.float32)
for i, (u, b) in enumerate(pairs):
    pair_embs[i] = bow_emb(u.split())  # Only user side
print(f"  pair_embs: {pair_embs.shape}")

# ==================================================================
# Step 11: Pack tensors as Float16+gzip+base64
# ==================================================================
print("Step 11: Pack tensors")

def encode_f16_gz_b64(arr):
    a = np.ascontiguousarray(arr.astype(np.float16))
    raw = a.tobytes()
    return base64.b64encode(gzip.compress(raw, compresslevel=9)).decode('ascii')

gom_b64 = encode_f16_gz_b64(omega)
gbi_b64 = encode_f16_gz_b64(bias)
gwt_b64 = encode_f16_gz_b64(W)
emb_b64 = encode_f16_gz_b64(emb_matrix)
pe_b64 = encode_f16_gz_b64(pair_embs)
idf_b64 = encode_f16_gz_b64(idf.astype(np.float32))

print(f"  gom: {len(gom_b64):,} chars")
print(f"  gbi: {len(gbi_b64):,} chars")
print(f"  gwt: {len(gwt_b64):,} chars")
print(f"  emb: {len(emb_b64):,} chars")
print(f"  pe:  {len(pe_b64):,} chars")
print(f"  idf: {len(idf_b64):,} chars")

# ==================================================================
# Step 12: Build M dict
# ==================================================================
print("Step 12: Build M dict")
# Sentences for retrieval (use bot side as searchable text)
sents = [b for u, b in pairs]
rw = [w for w in vocab]  # retrieval vocab (= regular vocab)
gw = tokens   # generation words (1x, not repeated — used for n-gram lookup only)

# Load RAG chunk index + compute BoW+IDF embeddings per chunk
import os
chunk_index = []
if os.path.exists(args.chunks):
    with open(args.chunks) as f:
        chunk_index = json.load(f)
    # Compute 32-dim BoW+IDF embedding per chunk
    # English keywords get 3× IDF boost to bridge the bilingual gap
    en_keywords_set = {'eigenvalue','eigenvalues','eigenvector','eigenvectors','kernel','pagerank',
                       'projection','regularization','convergence','overfitting','residual','iteration',
                       'fourier','radiosity','quantum','neural','matrix','vector','regression','ridge',
                       'stochastic','polynomial','diagonal','subspace','linear','nonlinear','feature'}
    for ci, chunk in enumerate(chunk_index):
        all_words = chunk['text'].split() + chunk.get('keywords', [])
        vec = np.zeros(EMB_DIM, dtype=np.float32)
        for w in all_words:
            if w in W2I:
                boost = 3.0 if w.lower() in en_keywords_set else 1.0
                vec += boost * idf[W2I[w]] * emb_matrix[W2I[w]]
        n = np.linalg.norm(vec) + 1e-10
        chunk['emb'] = (vec / n).tolist()
    print(f"  RAG chunk index: {len(chunk_index)} chunks loaded + BoW+IDF embeddings computed")

M = {
    'gc': {'CTX': CTX, 'EMB_DIM': EMB_DIM, 'FEAT': FEAT, 'D': D, 'SIGMA': SIGMA, 'V': V},
    'gv': vocab,
    'gw': gw,
    'gss': [],   # not used
    'pairs': [list(p) for p in pairs],
    'sents': sents,
    'rw': rw,
    'chunks': chunk_index,  # RAG blog chunks for auto-retrieval (with BoW+IDF embeddings)
    'gom': gom_b64,
    'gos': [FEAT, D],
    'gbi': gbi_b64,
    'gbs': [D],
    'gwt': gwt_b64,
    'gws': [D, V],
    'pe': pe_b64,
    'pes': [len(pairs), EMB_DIM],
    'emb': emb_b64,
    'embs': [V, EMB_DIM],
    'idf': idf_b64,
}

# ==================================================================
# Step 13: Inject into HTML template
# ==================================================================
print("Step 13: Inject into HTML")
print(f"  template: {args.template}")
with open(args.template) as f:
    html = f.read()

mstart = html.find('var M=')
mend_marker = '\nvar gc=M.gc'
mend = html.find(mend_marker, mstart)
old_M_str = html[mstart:mend]

new_M_json = json.dumps(M, separators=(',', ':'))
new_M_str = 'var M=' + new_M_json
html_new = html[:mstart] + new_M_str + html[mend:]

# Update header text
old_header = '<p>KRR Buddy-Chat ULTRA + TF-IDF BoW. 506 Woerter, 1281 Dialoge. GPU + 32-dim semantic match. Kein neuronales Netz.</p>'
new_header = f'<p>KRR Buddy-Chat Kalle. {V} Woerter, {len(pairs)} Dialoge. Top-1: {top1*100:.1f}%. Kuratiert + honest scope. Kein neuronales Netz.</p>'
if old_header in html_new:
    html_new = html_new.replace(old_header, new_header)
else:
    print('  WARN: header not found')

with open(args.output,'w') as f:
    f.write(html_new)
print(f"  saved {args.output} ({len(html_new):,} bytes = {len(html_new)/1024/1024:.1f} MB)")
print()
print("DONE")

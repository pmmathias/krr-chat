"""
Isolated solver benchmark on the REAL Kalle training matrices:
  A = Z^T Z + λI  (6144 × 6144)
  B = Z^T Y       (6144 × 2977)

Compares:
  - Direct (NumPy/LAPACK CPU)
  - Block-PCG (NumPy CPU)
  - Block-PCG (PyTorch CPU)
  - Block-PCG (PyTorch MPS = Apple GPU)

Reuses the trained matrices saved during build, so we only measure
solver cost (not the 90s of corpus preprocessing and Word2Vec).
"""
import os, sys, time, tracemalloc
import numpy as np

sys.path.insert(0, '/Users/mathiasleonhardt/Dev/krr-chat/src')
from solvers import direct_solve, block_cg

# ------------------------------------------------------------------
# Reconstruct the real Kalle A and B matrices by re-running the
# streaming Z^T Z accumulation (the same offline build does).
# This is the actual matrix the real model solves against.
# ------------------------------------------------------------------
import re, math
from gensim.models import Word2Vec

CTX, EMB_DIM, FEAT, D, SIGMA, LAMBDA, REPEAT, SEED = 24, 32, 768, 6144, 1.5, 1e-6, 5, 42
np.random.seed(SEED)

print("Reconstructing Kalle training matrices (this takes ~90s)...")
T0 = time.time()

with open('/Users/mathiasleonhardt/Dev/krr-chat/data/corpus.md') as f:
    txt = f.read()
pairs = []
for line in txt.splitlines():
    line = line.strip()
    if not line.startswith('du:'): continue
    if ' bot: ' not in line: continue
    user, bot = line.split(' bot: ', 1)
    user = user.replace('du:', '').strip().rstrip(' .').strip()
    bot = bot.strip().rstrip(' .').strip()
    if bot and bot.split()[-1] not in ['.', '?', '!']:
        bot = bot + ' .'
    if user and bot:
        pairs.append((user, bot))

tokens = []
for u, b in pairs:
    tokens += ['du:'] + u.split() + ['.', 'bot:'] + b.split() + ['.']
tokens_full = tokens * REPEAT
vocab = sorted(set(tokens_full))
W2I = {w: i for i, w in enumerate(vocab)}
V = len(vocab)

sentences = [['du:'] + u.split() + ['.', 'bot:'] + b.split() + ['.']
             for u, b in pairs] * 3
w2v = Word2Vec(sentences, vector_size=EMB_DIM, window=8, min_count=1,
               workers=4, sg=0, epochs=20, seed=SEED)
emb = np.zeros((V, EMB_DIM), dtype=np.float32)
for w, i in W2I.items():
    if w in w2v.wv:
        emb[i] = w2v.wv[w]
    else:
        emb[i] = np.random.randn(EMB_DIM).astype(np.float32) * 0.01

omega = np.random.randn(FEAT, D).astype(np.float32) / SIGMA
bias = np.random.rand(D).astype(np.float32) * 2 * np.pi
scale = math.sqrt(2.0 / D)

def encode_ctx(ctx_words):
    vec = np.zeros(FEAT, dtype=np.float32)
    for p in range(CTX):
        w = ctx_words[p] if p < len(ctx_words) else ''
        if not w or w not in W2I: continue
        wt = 0.4 + 0.6 * (p / (CTX - 1))
        vec[p*EMB_DIM:(p+1)*EMB_DIM] = wt * emb[W2I[w]]
    return vec

N = len(tokens_full) - CTX
Y_idx = np.array([W2I[tokens_full[i+CTX]] for i in range(N)], dtype=np.int32)

ZtZ = np.zeros((D, D), dtype=np.float64)
ZtY = np.zeros((D, V), dtype=np.float64)
CHUNK = 10000
for start in range(0, N, CHUNK):
    end = min(start + CHUNK, N)
    size = end - start
    F = np.zeros((size, FEAT), dtype=np.float32)
    for j in range(size):
        F[j] = encode_ctx(tokens_full[start+j:start+j+CTX])
    Z = scale * np.cos(F @ omega + bias[None, :])
    Z64 = Z.astype(np.float64)
    ZtZ += Z64.T @ Z64
    for j in range(size):
        ZtY[:, Y_idx[start + j]] += Z64[j]

A = ZtZ + LAMBDA * np.eye(D)
B = ZtY
print(f"  Setup: {time.time()-T0:.1f}s, A: {A.shape}, B: {B.shape}, V={V}")
print()

# ------------------------------------------------------------------
# Bench 1: Direct solve (NumPy)
# ------------------------------------------------------------------
print("=== Direct (NumPy/LAPACK CPU) ===")
tracemalloc.start()
t = time.time()
X_direct = np.linalg.solve(A, B).astype(np.float32)
t_direct = time.time() - t
_, peak_direct = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"  time: {t_direct*1000:.0f} ms, solve-only peak: {peak_direct/1024/1024:.0f} MB")

# ------------------------------------------------------------------
# Bench 2: Block-CG (NumPy CPU)
# ------------------------------------------------------------------
print("\n=== Block-PCG (NumPy CPU, diagonal) ===")
tracemalloc.start()
t = time.time()
X_cg_np, info = block_cg(A, B, tol=1e-6, max_iter=1000,
                         preconditioner='diagonal', verbose=False)
t_cg_np = time.time() - t
_, peak_cg_np = tracemalloc.get_traced_memory()
tracemalloc.stop()
err_np = np.linalg.norm(X_cg_np.astype(np.float32) - X_direct) / np.linalg.norm(X_direct)
print(f"  time: {t_cg_np*1000:.0f} ms, iters: {info['iterations']}, "
      f"peak: {peak_cg_np/1024/1024:.0f} MB, rel err vs direct: {err_np:.2e}")

# ------------------------------------------------------------------
# Bench 3+4: Block-CG (PyTorch CPU & MPS)
# ------------------------------------------------------------------
import torch

def block_cg_torch(A_t, B_t, tol=1e-6, max_iter=1000):
    diag_A = torch.diagonal(A_t).clone()
    diag_A[diag_A < 1e-30] = 1.0
    inv_diag = 1.0 / diag_A

    X = torch.zeros_like(B_t)
    R = B_t - A_t @ X
    Z = R * inv_diag.unsqueeze(1)
    P = Z.clone()
    rz_old = (R * Z).sum(dim=0)

    init_res = R.norm(dim=0).max().item()

    for it in range(max_iter):
        AP = A_t @ P
        pAp = (P * AP).sum(dim=0)
        pAp_safe = torch.where(pAp.abs() > 1e-30, pAp, torch.ones_like(pAp))
        alpha = rz_old / pAp_safe
        X = X + P * alpha.unsqueeze(0)
        R = R - AP * alpha.unsqueeze(0)
        Z = R * inv_diag.unsqueeze(1)
        rz_new = (R * Z).sum(dim=0)
        rel_res = R.norm(dim=0).max().item() / init_res
        if rel_res < tol:
            return X, it + 1, rel_res
        rz_old_safe = torch.where(rz_old.abs() > 1e-30, rz_old, torch.ones_like(rz_old))
        beta = rz_new / rz_old_safe
        P = Z + P * beta.unsqueeze(0)
        rz_old = rz_new
    return X, max_iter, rel_res


for device_name in ['cpu', 'mps']:
    print(f"\n=== Block-PCG (PyTorch {device_name.upper()}) ===")
    dev = torch.device(device_name)
    A_t = torch.from_numpy(A).to(dev, dtype=torch.float32)
    B_t = torch.from_numpy(B).to(dev, dtype=torch.float32)
    # Warmup
    _ = (A_t @ B_t[:, :10]).cpu()
    # Synchronize
    if device_name == 'mps':
        torch.mps.synchronize()
    t = time.time()
    X_t, iters, rel_res = block_cg_torch(A_t, B_t, tol=1e-5, max_iter=1000)
    if device_name == 'mps':
        torch.mps.synchronize()
    t_device = time.time() - t
    X_cpu = X_t.cpu().numpy()
    err = np.linalg.norm(X_cpu - X_direct) / np.linalg.norm(X_direct)
    print(f"  time: {t_device*1000:.0f} ms, iters: {iters}, "
          f"rel res: {rel_res:.2e}, err vs direct: {err:.2e}")

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
print("\n" + "="*72)
print("SUMMARY — solver cost on actual Kalle matrices")
print(f"  A: {A.shape} (6144 × 6144 SPD)")
print(f"  B: {B.shape} (6144 × {V})")
print("="*72)

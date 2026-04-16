"""
Training-pipeline scaling benchmark: how does the Z^T Z accumulation
scale vs. the solve step as we grow the corpus?

Scenarios (all using the same architecture: CTX=24, EMB_DIM=32, FEAT=768):
  S1: Kalle-only corpus (baseline)      → N ~ 322,636  V ~ 2,977
  S2: Kalle + full blog text             → N ~ 1,300,000 V ~ 18,000
  S3: Kalle + blog × 2                    → N ~ 2,300,000 V ~ 18,000

For each scenario, at D in {4096, 6144, 8192, 12288}, measure:
  - accumulation time (Z^T Z and Z^T Y streaming)
  - solve time (Direct via LAPACK, Block-PCG via PyTorch CPU, Block-PCG via MPS)
  - peak memory
  - resulting top-1 on sample
"""
import os, re, math, time, sys, gc
import numpy as np
from gensim.models import Word2Vec

sys.path.insert(0, '/Users/mathiasleonhardt/Dev/krr-chat/src')
from solvers import direct_solve, block_cg

# ---------- Architecture (fixed across scenarios) ----------
CTX = 24
EMB_DIM = 32
FEAT = CTX * EMB_DIM   # 768
LAMBDA = 1e-6
SEED = 42
np.random.seed(SEED)


def load_kalle_tokens():
    with open('/Users/mathiasleonhardt/Dev/krr-chat/data/corpus.md') as f:
        txt = f.read()
    tokens = []
    for line in txt.splitlines():
        line = line.strip()
        if not line.startswith('du:'): continue
        if ' bot: ' not in line: continue
        u, b = line.split(' bot: ', 1)
        u = u.replace('du:', '').strip().rstrip(' .').strip()
        b = b.strip().rstrip(' .').strip()
        if b and b.split()[-1] not in ['.', '?', '!']:
            b = b + ' .'
        if u and b:
            tokens += ['du:'] + u.split() + ['.', 'bot:'] + b.split() + ['.']
    return tokens


def load_blog_tokens():
    with open('/tmp/blog_tokens.txt') as f:
        return f.read().split()


def build_and_solve(tokens, D, solver='direct', repeat=5, device='cpu'):
    """Build Z^T Z, Z^T Y and solve. Returns timings and the W matrix."""
    # ----- Train Word2Vec on a mild approximation (chunk by 100 tokens) -----
    sentences = [tokens[i:i+100] for i in range(0, len(tokens), 100)]
    t0 = time.time()
    w2v = Word2Vec(sentences, vector_size=EMB_DIM, window=8, min_count=1,
                   workers=4, sg=0, epochs=10, seed=SEED)
    t_w2v = time.time() - t0

    vocab = sorted(set(tokens))
    W2I = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    emb = np.zeros((V, EMB_DIM), dtype=np.float32)
    for w, i in W2I.items():
        if w in w2v.wv:
            emb[i] = w2v.wv[w]
        else:
            emb[i] = np.random.randn(EMB_DIM).astype(np.float32) * 0.01

    # Repeat corpus
    tokens_full = tokens * repeat
    N = len(tokens_full) - CTX

    # Generate RFF omega/bias for this D
    rng = np.random.default_rng(SEED)
    omega = rng.standard_normal((FEAT, D)).astype(np.float32) / 1.5
    bias = rng.random(D, dtype=np.float32) * 2 * np.pi
    scale = math.sqrt(2.0 / D)

    def encode_ctx(ctx_words):
        vec = np.zeros(FEAT, dtype=np.float32)
        for p in range(CTX):
            w = ctx_words[p] if p < len(ctx_words) else ''
            if not w or w not in W2I: continue
            wt = 0.4 + 0.6 * (p / (CTX - 1))
            vec[p*EMB_DIM:(p+1)*EMB_DIM] = wt * emb[W2I[w]]
        return vec

    Y_idx = np.array([W2I[tokens_full[i+CTX]] for i in range(N)], dtype=np.int32)

    # ----- Streaming Z^T Z accumulation -----
    t0 = time.time()
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
        del F, Z, Z64
    ZtZ += LAMBDA * np.eye(D)
    t_accum = time.time() - t0

    # ----- Solve -----
    t0 = time.time()
    if solver == 'direct':
        W = np.linalg.solve(ZtZ, ZtY)
        iters = 1
    elif solver == 'cg_numpy':
        W, info = block_cg(ZtZ, ZtY, tol=1e-6, max_iter=1000,
                           preconditioner='diagonal', verbose=False)
        iters = info['iterations']
    elif solver == 'cg_torch':
        import torch
        dev = torch.device(device)
        A_t = torch.from_numpy(ZtZ).to(dev, dtype=torch.float32)
        B_t = torch.from_numpy(ZtY).to(dev, dtype=torch.float32)
        # Warmup
        _ = (A_t @ B_t[:, :10]).cpu()
        if device == 'mps':
            torch.mps.synchronize()
        t_solve_start = time.time()
        diag_A = torch.diagonal(A_t).clone()
        diag_A[diag_A < 1e-30] = 1.0
        inv_diag = 1.0 / diag_A
        X = torch.zeros_like(B_t)
        R = B_t - A_t @ X
        Z_ = R * inv_diag.unsqueeze(1)
        P_ = Z_.clone()
        rz_old = (R * Z_).sum(dim=0)
        init_res = R.norm(dim=0).max().item()
        iters = 0
        for it in range(1000):
            AP = A_t @ P_
            pAp = (P_ * AP).sum(dim=0)
            pAp_safe = torch.where(pAp.abs() > 1e-30, pAp, torch.ones_like(pAp))
            alpha = rz_old / pAp_safe
            X = X + P_ * alpha.unsqueeze(0)
            R = R - AP * alpha.unsqueeze(0)
            Z_ = R * inv_diag.unsqueeze(1)
            rz_new = (R * Z_).sum(dim=0)
            rel_res = R.norm(dim=0).max().item() / init_res
            if rel_res < 1e-5:
                iters = it + 1
                break
            rz_old_safe = torch.where(rz_old.abs() > 1e-30, rz_old, torch.ones_like(rz_old))
            beta = rz_new / rz_old_safe
            P_ = Z_ + P_ * beta.unsqueeze(0)
            rz_old = rz_new
        iters = iters or 1000
        if device == 'mps':
            torch.mps.synchronize()
        W = X.cpu().numpy().astype(np.float64)
        del A_t, B_t, X, R, Z_, P_, AP
    t_solve = time.time() - t0

    del ZtZ, ZtY
    gc.collect()

    return {
        'N': N,
        'V': V,
        'D': D,
        'solver': solver,
        'device': device if solver == 'cg_torch' else None,
        't_w2v': t_w2v,
        't_accum': t_accum,
        't_solve': t_solve,
        'iters': iters,
    }


# ---------- Run scenarios ----------
kalle = load_kalle_tokens()
blog = load_blog_tokens()

print(f"Kalle tokens:       {len(kalle):>9,}")
print(f"Blog tokens:        {len(blog):>9,}")
print(f"Kalle + blog:       {len(kalle) + len(blog):>9,}")
print(f"Kalle + blog × 2:   {len(kalle) + 2*len(blog):>9,}")
print()

scenarios = {
    'kalle':             kalle,
    'kalle+blog':        kalle + blog,
    'kalle+blog×2':      kalle + blog + blog,
}

# Only test at D=6144 to keep runtime manageable; we can add more later
D = 6144

results = []
for name, toks in scenarios.items():
    for solver, dev in [('direct', None), ('cg_torch', 'cpu'), ('cg_torch', 'mps')]:
        print(f"\n=== {name} | D={D} | {solver}{'-'+dev if dev else ''} ===")
        print(f"  tokens: {len(toks):,}")
        try:
            r = build_and_solve(toks, D=D, solver=solver, device=dev or 'cpu', repeat=5)
            r['scenario'] = name
            results.append(r)
            total = r['t_w2v'] + r['t_accum'] + r['t_solve']
            solve_pct = 100 * r['t_solve'] / total
            print(f"  N={r['N']:,}, V={r['V']:,}")
            print(f"  Word2Vec:    {r['t_w2v']:>6.1f}s")
            print(f"  Accumulate:  {r['t_accum']:>6.1f}s  ({100*r['t_accum']/total:.0f}%)")
            print(f"  Solve:       {r['t_solve']:>6.2f}s  ({solve_pct:.1f}%)  iters={r['iters']}")
            print(f"  Total:       {total:>6.1f}s")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            results.append({'scenario': name, 'solver': solver, 'failed': str(e)})
        gc.collect()

# Summary
print("\n" + "=" * 90)
print("SUMMARY: how does training time decompose as corpus grows?")
print("=" * 90)
print(f"{'Scenario':<16} {'Solver':<12} {'N':>10} {'V':>6} {'Accum':>8} {'Solve':>8} {'Solve%':>7} {'Total':>7}")
print("-" * 90)
for r in results:
    if r.get('failed'):
        print(f"{r['scenario']:<16} {r['solver']:<12} FAILED: {r['failed'][:50]}")
        continue
    total = r['t_w2v'] + r['t_accum'] + r['t_solve']
    dev_suffix = f"-{r['device']}" if r['device'] else ""
    print(f"{r['scenario']:<16} {r['solver']+dev_suffix:<12} "
          f"{r['N']:>10,} {r['V']:>6,} "
          f"{r['t_accum']:>7.1f}s {r['t_solve']:>7.2f}s "
          f"{100*r['t_solve']/total:>6.1f}% {total:>6.1f}s")

import json
with open('/tmp/bench_scaling_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to /tmp/bench_scaling_results.json")

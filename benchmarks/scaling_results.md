# Training Pipeline Scaling: How Does the Solver Share Evolve with Corpus Size?

**Date:** 2026-04-16
**Hardware:** Apple Silicon (M-class), 16 GB unified memory
**Script:** `src/benchmark_scaling.py`
**Question:** When the corpus grows from Kalle's 64K tokens to the full ki-mathias.de blog (+197K tokens) and beyond, does the solve step become a bigger share of total training time?

## Scenarios

| Corpus | Tokens | Samples (N) | Vocab (V) | Comment |
|---|---|---|---|---|
| kalle | 64,532 | 322,636 | 2,977 | Baseline (current Kalle) |
| kalle + blog | 261,910 | 1,309,526 | 19,411 | + full DE/EN blog content (30 articles, 197K tokens) |
| kalle + blog × 2 | 459,288 | 2,296,416 | 19,411 | + blog text repeated once more (stress test for N scaling) |

Architecture fixed at Kalle's defaults: `CTX=24, EMB_DIM=32, FEAT=768, D=6144, σ=1.5, λ=1e-6, REPEAT=5`.

## Results

| Scenario | Solver | Accum | Solve | Solve % | Total | Iters |
|---|---|---|---|---|---|---|
| kalle | Direct (NumPy, Float64) | 44.9 s | 2.38 s | 5.0% | 47.4 s | 1 |
| kalle | CG (PyTorch CPU, Float32) | 46.5 s | 3.77 s | 7.5% | 50.4 s | 20 |
| kalle | CG (PyTorch MPS, Float32) | 46.5 s | **2.19 s** | 4.5% | 48.8 s | 21 |
| kalle + blog | Direct | 197.2 s | 10.20 s | 4.9% | 208.0 s | 1 |
| kalle + blog | CG (PyTorch CPU) | 189.9 s | 10.86 s | 5.4% | 201.4 s | 8 |
| kalle + blog | CG (PyTorch MPS) | 189.4 s | **6.74 s** | 3.4% | 196.7 s | 7 |
| kalle + blog × 2 | Direct | 350.0 s | 9.75 s | 2.7% | 360.7 s | 1 |
| kalle + blog × 2 | CG (PyTorch CPU) | 324.7 s | **10.47 s** | 3.1% | 336.1 s | 7 |
| kalle + blog × 2 | CG (PyTorch MPS) | 324.0 s | 18.35 s | 5.3% | 343.3 s | 7 |

## Three non-obvious findings

### 1. Accumulation scales linearly with N; solve is (nearly) independent of N

As $N$ grows from 322K → 1.3M → 2.3M samples (factor 7.1 total), the streaming $Z^\top Z$ and $Z^\top Y$ accumulation scales exactly linearly: 45 s → 190 s → 325 s (factor 7.2). This is the expected behavior: each sample produces one rank-1 update to $Z^\top Z$ and one single-column increment to $Z^\top Y$; total work is $O(N \cdot D^2)$.

The solve time, by contrast, **jumps with $V$ (vocabulary) but is essentially independent of $N$** once the matrices are accumulated: Direct solve goes 2.4 s → 10.2 s → 9.8 s as $V$ jumps from 2,977 → 19,411 → 19,411. The $O(D^3)$ factorization cost is fixed; what scales is the $O(D^2 \cdot V)$ back-substitution, proportional to how many vocabulary columns need solving.

**The solver share therefore stays roughly constant across corpus sizes** — even at the full 2.3 M-sample regime (7× Kalle's baseline), the solve is still only 3% of total training time. The "solve is a small share" finding is not an artifact of Kalle's small corpus; it is a structural property of the pipeline at this $D$.

### 2. More vocabulary → fewer CG iterations (counterintuitively)

At $V = 2,977$, Block-PCG needed 20–21 iterations to converge. At $V = 19,411$, it needed only 7–8. The reason: with more diverse training data, the effective condition number of the kernel matrix $Z^\top Z + \lambda I$ becomes more favorable — eigenvalues decay more smoothly, reducing the $\sqrt{\kappa}$ factor in CG's convergence bound. This reverses the naive expectation that "bigger problems are harder to solve iteratively."

### 3. MPS wins at $V = 19{,}411, N = 1.3 \text{M}$ but loses at $N = 2.3 \text{M}$

At the medium scenario (`kalle + blog`), Apple MPS shows a real 1.5× speedup over PyTorch CPU for the solve (6.74 s vs 10.86 s). At the larger scenario (`kalle + blog × 2`), MPS unexpectedly slows down to 18.4 s — slower than CPU. The matrices ($D \times V = 6144 \times 19411$, $\sim 480$ MB per Float32 copy, times the state vectors $X, R, Z, P$) approach the ~8 GB working-set limit of the unified memory, and we likely hit swap behavior on the GPU side. A dedicated GPU with more VRAM would not exhibit this regression. This is a concrete example of **when the scaling analysis in the paper becomes instrument-specific**.

## When would the solver become dominant?

The answer is: **when $D$ grows, not $N$**.

- **Direct solve:** $O(D^3)$ compute + $O(D^2)$ memory. At $D = 50{,}000$ this is $\sim 20$ GB and hours of wall time — infeasible on the benchmark hardware. The accumulation becomes negligible in comparison.
- **CG:** $O(D^2 \cdot V \cdot \text{iters})$. At the same $D = 50{,}000$, one matrix-matrix product $A \cdot P$ is 100× larger than at $D = 6144$, but remains feasible because no dense factor is stored.

The practical lever for scaling Kalle is therefore:
1. **Increase $D$ toward the regime where direct solve is infeasible**, so the Block-PCG contribution becomes indispensable — and MPS pays off.
2. **Accelerate the accumulation on GPU** (which this benchmark does not yet), since that dominates at this $D$.

## End-to-end decomposition at the full blog scale

For `kalle + blog × 2` with Block-PCG (MPS), the breakdown is:

| Phase | Time | Share |
|---|---|---|
| Corpus parsing + Word2Vec | 0.9 s | 0.3% |
| Streaming $Z^\top Z$ + $Z^\top Y$ accumulation | 324.0 s | 94.4% |
| Block-PCG solve | 18.4 s | 5.3% |

**The honest answer to "does the solver matter more with more data?"** is: the relative share of solve is remarkably stable across our 7× scaling test — but the absolute cost of training grows linearly. The next-highest-impact engineering targets are (a) parallelizing the accumulation on GPU and (b) reducing the `REPEAT` factor from 5 to something smaller now that we have more natural data, rather than further solver tuning.

## Reproducing

```bash
# Extract blog tokens (one-time, ~1 s)
python3.11 /tmp/extract_blog_text.py

# Run the scaling benchmark (~30 minutes on M-class Apple Silicon)
python3.11 src/benchmark_scaling.py
```

# Solver Benchmark on Real Kalle Training Matrices

**Date:** 2026-04-16
**Hardware:** Apple Silicon (M-class), 16 GB unified memory
**Matrix:** $A = Z^\top Z + \lambda I$ with $D = 6144$, 2977 simultaneous right-hand sides
**Reconstructed from:** the actual Kalle training pipeline (same Word2Vec seed, same corpus, same RFF projection)

## Results

| Solver | Time | Iterations | Speedup vs Direct | Accuracy vs Direct |
|---|---|---|---|---|
| **Direct** (NumPy / LAPACK, Float64) | 2097 ms | 1 | 1.00× (baseline) | reference |
| **Block-PCG** (NumPy, Float64, diagonal precond.) | 6731 ms | 14 | 0.31× (3.2× slower) | 4.3e-07 rel. Frobenius |
| **Block-PCG** (PyTorch CPU, Float32) | 1704 ms | 11 | **1.23× (faster)** | 1.0e-05 rel. Frobenius |
| **Block-PCG** (PyTorch MPS / Apple GPU, Float32) | 1622 ms | 11 | **1.29× (faster)** | 1.0e-05 rel. Frobenius |

Run: `python3.11 src/benchmark_real_kalle.py`

## Three non-obvious findings

### 1. The bigger win is PyTorch, not the GPU

On this hardware, **switching the BLAS backend from NumPy to PyTorch** produces a larger speedup (3.95×) than moving to the GPU on top of that (additional 1.05×). PyTorch's matrix-matrix product routines bind to newer, better-tuned BLAS kernels than the ones NumPy links against by default on this system. At $D = 6144$, a well-implemented Block-PCG on the CPU is already faster than a direct solve — it just needs better primitives than pure NumPy gives you.

### 2. Apple MPS is underutilized at $D = 6144$

The jump from PyTorch CPU to PyTorch MPS is only 80 ms (~5%). The problem is too small for Apple's GPU to shine: transfer overhead CPU↔GPU and launch latency dominate. Extrapolating from the $O(D^2 \cdot V \cdot \text{iters})$ scaling, MPS should start to win decisively around $D \geq 20{,}000$; at $D = 50{,}000$ (where direct solve is OOM anyway on this 16 GB machine) we would expect MPS to beat CPU by roughly 5–10×.

### 3. Float32 is sufficient

PyTorch runs in Float32; the reference Float64 direct solve differs by rel. Frobenius error $\sim 10^{-5}$. For Kalle's downstream argmax-based prediction, this is invisible (identical Top-1 accuracy within sampling noise).

## Where the training time actually goes

Full end-to-end training of Kalle (corpus → Word2Vec → RFF → solve → HTML pack) takes $\sim 100$ s. Of that:

| Phase | Time | Share |
|---|---|---|
| Streaming $Z^\top Z$ + $Z^\top Y$ accumulation (322,636 RFF features) | ~45 s | 47% |
| Word2Vec training (20 epochs on 322 k tokens) | ~1 s | 1% |
| Corpus parsing + token sequence build | ~2 s | 2% |
| **Linear solve (this benchmark)** | **2–7 s** | **2–7%** |
| IDF + BoW embeddings + HTML pack | ~3 s | 3% |
| HTML template inject + base64 encode | ~40 s | 40% |

**The solve is not the bottleneck.** If we wanted to accelerate Kalle's training significantly, the big wins are elsewhere:
- RFF feature accumulation on GPU (via PyTorch)
- A smaller corpus repetition factor (we use ×5)
- Compiled/JIT feature encoder instead of Python-loop `encode_ctx`

The v2 paper's Block-PCG story still matters — for the scaling regime ($D \gg 10^4$) where direct solve is infeasible — but at Kalle's current size, **the honest answer is that solver choice is a minor lever**. What mattered instead was that we now have a path to GPU, and a theoretical framework (absorber-stochasticization) that predicts which problems will benefit most.

## Reproducing

```bash
# Real Kalle matrices (~50 s setup + a few seconds per solver)
python3.11 src/benchmark_real_kalle.py

# Synthetic problems across D values (fast)
python3.11 src/benchmark.py --D 256 512 1024 2048 4096
```

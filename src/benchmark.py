"""
Benchmark v1 (direct solve) vs v2 (Block-PCG) across various D values.

Generates synthetic SPD problems with realistic structure (Z^T Z + λI) and
measures peak memory, compute time, and solver accuracy at multiple scales.

Usage:
  python3 src/benchmark.py --D 1024 2048 4096 6144 8192 --output benchmarks/results.json
"""
import argparse
import json
import os
import sys
import time
import tracemalloc
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solvers import direct_solve, block_cg


def generate_problem(D, V=1000, N=None, seed=42, lambda_reg=1e-6):
    """Generate a realistic SPD problem (A, B) where A = Z^T Z + λI."""
    if N is None:
        N = max(2 * D, 5000)
    rng = np.random.default_rng(seed)
    Z = (rng.standard_normal((N, D)) * 0.1).astype(np.float64)
    A = Z.T @ Z + lambda_reg * np.eye(D)
    Y_idx = rng.integers(0, V, N)
    B = np.zeros((D, V), dtype=np.float64)
    for j in range(N):
        B[:, Y_idx[j]] += Z[j]
    return A, B


def benchmark_solver(A, B, solver_name, **solver_kwargs):
    """Run a solver and measure memory + time."""
    tracemalloc.start()
    t0 = time.time()

    if solver_name == 'direct':
        X = direct_solve(A, B)
        info = {'solver': 'direct', 'iterations': 1}
    elif solver_name == 'cg':
        X, info = block_cg(A, B, verbose=False, **solver_kwargs)
        info['solver'] = 'cg'
    else:
        raise ValueError(f"Unknown solver: {solver_name}")

    elapsed = time.time() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return X, {
        **info,
        'elapsed_sec': elapsed,
        'peak_mem_MB': peak / 1024 / 1024,
        'current_mem_MB': current / 1024 / 1024,
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark KRR solvers')
    parser.add_argument('--D', type=int, nargs='+',
                        default=[512, 1024, 2048, 4096],
                        help='RFF dimensions to benchmark (default: 512 1024 2048 4096)')
    parser.add_argument('--V', type=int, default=1000,
                        help='Vocabulary size (default: 1000)')
    parser.add_argument('--lambda-reg', type=float, default=1e-6,
                        help='Ridge regularization (default: 1e-6)')
    parser.add_argument('--cg-tol', type=float, default=1e-6,
                        help='CG tolerance (default: 1e-6)')
    parser.add_argument('--cg-maxiter', type=int, default=500,
                        help='CG max iterations (default: 500)')
    parser.add_argument('--output', default='benchmarks/results.json',
                        help='Output JSON file (default: benchmarks/results.json)')
    parser.add_argument('--repeats', type=int, default=2,
                        help='Repetitions per config (median taken, default: 2)')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    results = []

    for D in args.D:
        print(f"\n{'='*70}")
        print(f"D = {D}, V = {args.V}, λ = {args.lambda_reg}")
        print(f"{'='*70}")

        # Generate the problem once per D
        print(f"  generating problem...")
        A, B = generate_problem(D, V=args.V, lambda_reg=args.lambda_reg)
        cond_est = float(np.linalg.cond(A)) if D <= 4096 else None
        print(f"  A shape: {A.shape}, cond ≈ {cond_est}")

        # Benchmark direct
        print(f"\n  --- Direct solve ---")
        direct_results = []
        X_ref = None
        for r in range(args.repeats):
            X, info = benchmark_solver(A, B, 'direct')
            print(f"    run {r+1}: {info['elapsed_sec']:.2f}s, peak={info['peak_mem_MB']:.0f} MB")
            direct_results.append(info)
            if X_ref is None:
                X_ref = X
        direct_med = {
            'elapsed_sec': float(np.median([r['elapsed_sec'] for r in direct_results])),
            'peak_mem_MB': float(np.median([r['peak_mem_MB'] for r in direct_results])),
        }

        # Benchmark CG (with diagonal preconditioner)
        print(f"\n  --- Block-CG (diagonal) ---")
        cg_results = []
        for r in range(args.repeats):
            X, info = benchmark_solver(
                A, B, 'cg',
                tol=args.cg_tol, max_iter=args.cg_maxiter, preconditioner='diagonal',
            )
            err = float(np.linalg.norm(X - X_ref) / np.linalg.norm(X_ref))
            print(f"    run {r+1}: {info['iterations']} iters, "
                  f"{info['elapsed_sec']:.2f}s, peak={info['peak_mem_MB']:.0f} MB, err={err:.2e}")
            info['accuracy_vs_direct'] = err
            cg_results.append(info)
        cg_med = {
            'iterations': int(np.median([r['iterations'] for r in cg_results])),
            'elapsed_sec': float(np.median([r['elapsed_sec'] for r in cg_results])),
            'peak_mem_MB': float(np.median([r['peak_mem_MB'] for r in cg_results])),
            'accuracy_vs_direct': float(np.median([r['accuracy_vs_direct'] for r in cg_results])),
        }

        results.append({
            'D': D,
            'V': args.V,
            'lambda_reg': args.lambda_reg,
            'condition_number': cond_est,
            'direct': direct_med,
            'cg_diagonal': cg_med,
        })

        # Free memory before next D
        del A, B, X_ref

    # Print summary table
    print(f"\n{'='*90}")
    print("SUMMARY")
    print(f"{'='*90}")
    print(f"{'D':>6} | {'Direct (s)':>12} | {'Direct mem':>11} | "
          f"{'CG iters':>10} | {'CG (s)':>9} | {'CG mem':>9} | {'CG err':>10}")
    print("-" * 90)
    for r in results:
        print(f"{r['D']:>6} | {r['direct']['elapsed_sec']:>12.3f} | "
              f"{r['direct']['peak_mem_MB']:>9.1f} MB | "
              f"{r['cg_diagonal']['iterations']:>10d} | "
              f"{r['cg_diagonal']['elapsed_sec']:>9.3f} | "
              f"{r['cg_diagonal']['peak_mem_MB']:>7.1f} MB | "
              f"{r['cg_diagonal']['accuracy_vs_direct']:>10.2e}")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()

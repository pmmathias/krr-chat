"""
Linear system solvers for Kernel Ridge Regression: A·W = B
where A = (Z^T Z + λI) ∈ R^{D×D} (SPD), B = Z^T Y ∈ R^{D×V}, W ∈ R^{D×V}.

Three solvers, all returning the same mathematical result up to numerical
tolerance. They differ in memory usage, GPU-friendliness, and scalability.

Implements the algorithms derived in paper/theory/absorber_stochastisierung.md.

References:
  - Hestenes & Stiefel (1952): Methods of Conjugate Gradients
  - Saad (2003): Iterative Methods for Sparse Linear Systems
  - Yao, Rosasco, Caponnetto (2007): On Early Stopping in Gradient Descent Learning
  - Boldi, Santini, Vigna (2005): PageRank as a Function of the Damping Factor
"""
import time
import numpy as np


def direct_solve(A, B):
    """
    Direct Gaussian elimination via numpy.linalg.solve.
    Memory: O(D^2) (A is materialized as a dense D×D matrix)
    Compute: O(D^3) for the solve, O(D^2 · V) for back-substitution

    This is the v1 solver — closed-form, exact, but does not scale to large D.
    """
    return np.linalg.solve(A, B)


def block_cg(A, B, X0=None, tol=1e-6, max_iter=1000, preconditioner='diagonal',
             verbose=True):
    """
    Block Preconditioned Conjugate Gradient (Block-PCG).

    Solves A·X = B for X ∈ R^{D×V} where A is symmetric positive definite.
    Each column of X is treated as an independent CG run; the matrix-matrix
    product A·P (per iteration) parallelizes the work across all V columns.

    Memory: O(D · V) for X, R, Z, P (no A^{-1} or LU factors stored)
    Compute per iteration: O(D^2 · V) for A·P (the dominant cost)
    Convergence: O(sqrt(κ) · log(1/tol)) iterations where κ = κ(A)

    Args:
        A: D×D SPD matrix
        B: D×V right-hand side
        X0: D×V initial guess (default: zeros)
        tol: convergence tolerance on max column residual norm (relative)
        max_iter: maximum iterations
        preconditioner: 'none' | 'diagonal' (Jacobi)
        verbose: print convergence info

    Returns:
        X: D×V solution matrix
        info: dict with 'iterations', 'final_residual', 'converged'
    """
    D, V = B.shape
    assert A.shape == (D, D), f"A shape {A.shape} != ({D}, {D})"

    if X0 is None:
        X = np.zeros_like(B)
    else:
        X = X0.copy()

    # Set up preconditioner
    if preconditioner == 'none':
        apply_precond = lambda r: r
    elif preconditioner == 'diagonal':
        diag_A = np.diag(A).copy()
        # Avoid division by zero (shouldn't happen for SPD, but safety)
        diag_A[diag_A < 1e-30] = 1.0
        inv_diag = 1.0 / diag_A
        apply_precond = lambda r: r * inv_diag[:, None]
    else:
        raise ValueError(f"Unknown preconditioner: {preconditioner}")

    # Initial residual and direction
    R = B - A @ X            # D×V
    Z = apply_precond(R)     # D×V
    P = Z.copy()             # D×V (search direction)

    # Per-column inner products (V scalars each)
    rz_old = np.sum(R * Z, axis=0)  # V

    # Initial residual norm (for relative tolerance)
    init_res_norm = np.linalg.norm(R, axis=0).max()
    if init_res_norm == 0:
        if verbose:
            print(f"  Block-PCG: trivial case, B is zero")
        return X, {'iterations': 0, 'final_residual': 0.0, 'converged': True}

    t0 = time.time()
    last_print = 0

    for it in range(max_iter):
        AP = A @ P                       # D×V (the heavy lifting)
        pAp = np.sum(P * AP, axis=0)     # V

        # Avoid division by zero (column already converged)
        pAp_safe = np.where(np.abs(pAp) > 1e-30, pAp, 1.0)
        alpha = rz_old / pAp_safe        # V

        X = X + P * alpha[None, :]
        R = R - AP * alpha[None, :]
        Z = apply_precond(R)

        rz_new = np.sum(R * Z, axis=0)

        # Convergence check (relative max column residual)
        res_norm = np.linalg.norm(R, axis=0).max()
        rel_res = res_norm / init_res_norm

        if verbose and (it - last_print >= 50 or it < 5):
            print(f"  iter {it+1}: rel_res = {rel_res:.2e}")
            last_print = it

        if rel_res < tol:
            elapsed = time.time() - t0
            if verbose:
                print(f"  Block-PCG converged: {it+1} iterations, "
                      f"rel_res={rel_res:.2e}, {elapsed:.1f}s")
            return X, {
                'iterations': it + 1,
                'final_residual': float(rel_res),
                'converged': True,
                'elapsed_sec': elapsed,
            }

        # Avoid division by zero in beta
        rz_old_safe = np.where(np.abs(rz_old) > 1e-30, rz_old, 1.0)
        beta = rz_new / rz_old_safe
        P = Z + P * beta[None, :]
        rz_old = rz_new

    elapsed = time.time() - t0
    if verbose:
        print(f"  Block-PCG: max_iter={max_iter} reached, "
              f"final rel_res={rel_res:.2e}, {elapsed:.1f}s")
    return X, {
        'iterations': max_iter,
        'final_residual': float(rel_res),
        'converged': False,
        'elapsed_sec': elapsed,
    }


def power_iteration_stochastic(A_unreg, B, lambda_reg, max_iter=10000,
                                tol=1e-6, verbose=True):
    """
    Power iteration on the absorber-stochasticized matrix (didactic implementation).

    Constructs the augmented stochastic matrix as derived in
    paper/theory/absorber_stochastisierung.md §3.3, then iterates.

    NOTE: This is included for didactic comparison with the PageRank analogy.
    For practical use, prefer block_cg (faster convergence: O(sqrt(κ)) vs O(1/λ)).

    Args:
        A_unreg: D×D matrix (Z^T Z, NOT including λI yet)
        B: D×V right-hand side (Z^T Y)
        lambda_reg: ridge regularization parameter
        max_iter: maximum iterations
        tol: convergence tolerance

    Returns:
        X: D×V solution matrix
        info: convergence info dict
    """
    D, V = B.shape

    # Normalize A so mu_max(A_unreg) <= 1 (per theory §3.3)
    # Use spectral radius estimate via power method on A_unreg
    print(f"  Estimating spectral radius of A...")
    v = np.random.randn(D).astype(np.float64)
    v /= np.linalg.norm(v)
    for _ in range(50):
        v = A_unreg @ v
        v_norm = np.linalg.norm(v)
        if v_norm > 0:
            v /= v_norm
    spec_radius = float(np.linalg.norm(A_unreg @ v))
    print(f"  Estimated mu_max(A_unreg) ≈ {spec_radius:.3f}")

    A_norm = A_unreg / spec_radius
    omega_step = 1.0 / (1.0 + lambda_reg)

    # Iteration matrix M = I - omega * (A_norm + lambda * I)
    # Effective right-hand side b = omega * B / spec_radius (scaling adjustment)
    # X_new = M @ X + b

    X = np.zeros_like(B)
    b_eff = omega_step * (B / spec_radius)
    coef_for_X = 1.0 - omega_step * lambda_reg  # diagonal scalar

    t0 = time.time()
    init_res = np.linalg.norm(B, axis=0).max()

    for it in range(max_iter):
        # X_new = (I - omega * A_norm) X - omega*lambda*X + b_eff
        #       = X - omega * (A_norm @ X) - omega*lambda*X + b_eff
        AX = A_norm @ X
        X_new = X * coef_for_X - omega_step * AX + b_eff

        diff = np.linalg.norm(X_new - X, axis=0).max()
        X = X_new

        if it % 100 == 0 and verbose:
            print(f"  iter {it}: max diff = {diff:.2e}")

        if diff < tol * init_res:
            elapsed = time.time() - t0
            if verbose:
                print(f"  Power iteration converged: {it+1} iterations, {elapsed:.1f}s")
            return X * spec_radius, {
                'iterations': it + 1,
                'final_residual': float(diff),
                'converged': True,
                'elapsed_sec': elapsed,
            }

    elapsed = time.time() - t0
    if verbose:
        print(f"  Power iteration: max_iter={max_iter} reached, {elapsed:.1f}s")
    return X * spec_radius, {
        'iterations': max_iter,
        'final_residual': float(diff),
        'converged': False,
        'elapsed_sec': elapsed,
    }


def solve(A, B, solver='direct', **kwargs):
    """
    Unified entry point: solve A·X = B with the chosen solver.

    Args:
        A: D×D SPD matrix (= Z^T Z + λI for KRR)
        B: D×V right-hand side
        solver: 'direct' | 'cg' | 'power'
        **kwargs: solver-specific args (tol, max_iter, preconditioner, ...)

    Returns:
        X: D×V solution matrix
        info: dict (empty for direct, populated for iterative solvers)
    """
    if solver == 'direct':
        X = direct_solve(A, B)
        return X, {'solver': 'direct'}
    elif solver == 'cg':
        X, info = block_cg(A, B, **kwargs)
        info['solver'] = 'cg'
        return X, info
    elif solver == 'power':
        # power_iteration_stochastic expects A_unreg (without λI), so we need to extract λ
        lambda_reg = kwargs.pop('lambda_reg', None)
        if lambda_reg is None:
            raise ValueError("Power solver requires lambda_reg kwarg")
        # A here is (Z^T Z + λI), so A_unreg = A - λI
        D = A.shape[0]
        A_unreg = A - lambda_reg * np.eye(D)
        X, info = power_iteration_stochastic(A_unreg, B, lambda_reg, **kwargs)
        info['solver'] = 'power'
        return X, info
    else:
        raise ValueError(f"Unknown solver: {solver!r}. Choose from: direct, cg, power")

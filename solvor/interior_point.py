r"""
Interior Point Method for Linear Programming.

While simplex walks along edges of the feasible polytope, interior point
cuts straight through the middle. Imagine simplex as taking the stairs,
interior point as taking the elevator - different path, same destination.

    from solvor import solve_lp_interior

    # minimize c @ x, subject to A @ x <= b, x >= 0
    result = solve_lp_interior(c, A, b)

How it works: implements primal-dual interior point with Mehrotra's
predictor-corrector. Each iteration solves a Newton system to find a
direction toward optimality while staying strictly inside the feasible
region. The barrier parameter mu drives complementarity to zero.

Use this when:

- Understanding how modern LP solvers work (HiGHS, CPLEX, Gurobi)
- When simplex is cycling or slow on degenerate problems
- Learning the "other" way to solve LP

Parameters:

    c: objective coefficients
    A: constraint matrix
    b: constraint bounds
    minimize: if True minimize, else maximize
    eps: convergence tolerance

Simplex is often faster for small/medium LPs. Interior point shines on
large, sparse problems, but that advantage disappears without numpy/scipy.
This implementation is for learning, not production.
"""

from collections.abc import Sequence
from math import sqrt

from solvor.types import Result, Status
from solvor.utils import check_matrix_dims, warn_large_coefficients

__all__ = ["solve_lp_interior"]


def solve_lp_interior(
    c: Sequence[float],
    A: Sequence[Sequence[float]],
    b: Sequence[float],
    *,
    minimize: bool = True,
    eps: float = 1e-8,
    max_iter: int = 100,
) -> Result[tuple[float, ...]]:
    """Solve LP using primal-dual interior point method.

    minimize c @ x  subject to  A @ x <= b, x >= 0

    Converts to standard form by adding slacks, then solves using
    Mehrotra's predictor-corrector with infeasible start.
    """
    check_matrix_dims(c, A, b)
    warn_large_coefficients(A)

    m, n = len(b), len(c)
    if m == 0 or n == 0:
        return Result(tuple([0.0] * n), 0.0, 0, 0, Status.OPTIMAL)

    # Flip objective for maximization
    obj = list(c) if minimize else [-ci for ci in c]

    c_ext = obj + [0.0] * m
    n_total = n + m

    # Build augmented constraint matrix [A | I]
    A_aug = []
    for i in range(m):
        row = list(A[i]) + [0.0] * m
        row[n + i] = 1.0  # slack variable coefficient
        A_aug.append(row)

    # Initialize with strictly positive starting point
    # Use infeasible-start method - don't require Ax = b initially
    x, y, z = _initialize(c_ext, m, n_total)

    for iteration in range(max_iter):
        # Primal residual: rb = Ax - b
        rb = [sum(A_aug[i][j] * x[j] for j in range(n_total)) - b[i] for i in range(m)]

        # Dual residual: rc = A'y + z - c
        rc = [sum(A_aug[i][j] * y[i] for i in range(m)) + z[j] - c_ext[j] for j in range(n_total)]

        # Complementarity products
        xz = [x[j] * z[j] for j in range(n_total)]
        mu = sum(xz) / n_total

        # Convergence check
        primal_inf = sqrt(sum(r * r for r in rb))
        dual_inf = sqrt(sum(r * r for r in rc))

        if primal_inf < eps and dual_inf < eps and mu < eps:
            solution = tuple(x[j] for j in range(n))
            objective = sum(obj[j] * x[j] for j in range(n))
            if not minimize:
                objective = -objective
            return Result(solution, objective, iteration, iteration, Status.OPTIMAL)

        # Predictor step (affine scaling direction, sigma=0)
        dx_aff, dy_aff, dz_aff = _solve_newton(A_aug, x, z, rb, rc, xz, m, n_total, eps)

        if dx_aff is None:
            break

        # Step lengths to boundary
        alpha_p = _step_length(x, dx_aff, n_total)
        alpha_d = _step_length(z, dz_aff, n_total)

        # Affine gap after step
        mu_aff = sum((x[j] + alpha_p * dx_aff[j]) * (z[j] + alpha_d * dz_aff[j])
                     for j in range(n_total)) / n_total

        # Centering parameter (Mehrotra heuristic)
        if mu > 1e-12:
            ratio = mu_aff / mu
            sigma = min(1.0, ratio * ratio * ratio)
        else:
            sigma = 0.1

        # Corrector step with centering
        # Modify complementarity RHS: xz - sigma*mu + dx_aff*dz_aff
        xz_mod = [xz[j] - sigma * mu + dx_aff[j] * dz_aff[j] for j in range(n_total)]

        dx, dy, dz = _solve_newton(A_aug, x, z, rb, rc, xz_mod, m, n_total, eps)

        if dx is None:
            # Fall back to affine direction
            dx, dy, dz = dx_aff, dy_aff, dz_aff

        # Step lengths with safety margin
        tau = max(0.9, 1.0 - mu)  # more aggressive as we converge
        alpha_p = tau * _step_length(x, dx, n_total)
        alpha_d = tau * _step_length(z, dz, n_total)

        # Take step
        for j in range(n_total):
            x[j] = max(eps, x[j] + alpha_p * dx[j])
            z[j] = max(eps, z[j] + alpha_d * dz[j])
        for i in range(m):
            y[i] += alpha_d * dy[i]

    # Extract solution after max iterations
    solution = tuple(max(0.0, x[j]) for j in range(n))
    objective = sum(obj[j] * solution[j] for j in range(n))
    if not minimize:
        objective = -objective

    # Check final feasibility
    primal_inf = sqrt(sum((sum(A_aug[i][j] * x[j] for j in range(n_total)) - b[i]) ** 2
                          for i in range(m)))
    if primal_inf < 0.01:
        return Result(solution, objective, max_iter, max_iter, Status.FEASIBLE)

    return Result(solution, objective, max_iter, max_iter, Status.MAX_ITER)


def _initialize(c_ext, m, n_total):
    """Initialize with strictly positive x, z and zero y."""
    # Start with x = 1, z = max(1, |c|) to be "centered"
    x = [1.0] * n_total
    y = [0.0] * m
    z = [max(1.0, abs(c_ext[j]) + 1.0) for j in range(n_total)]
    return x, y, z


def _solve_newton(A_aug, x, z, rb, rc, xz, m, n_total, eps):
    """Solve the Newton system for (dx, dy, dz).

    The KKT system is:
    [A   0   0 ] [dx]   [-rb     ]
    [0   A'  I ] [dy] = [-rc     ]
    [Z   0   X ] [dz]   [-xz_mod ]

    Eliminate dz = -z - (X/Z)*dx + correction term from 3rd equation.
    Substitute into 2nd: A'dy + dz = -rc
      -> A'dy - z - (X/Z)dx + ... = -rc

    Then eliminate dx from 1st: dx = -D*(A'dy + rc - correction)
    where D = X/Z (diagonal scaling).

    Normal equations: (A D A') dy = A D (rc - correction) - rb
    """
    # Scaling: D = X / Z
    d = [x[j] / max(z[j], eps) for j in range(n_total)]

    # RHS for dz from complementarity: dz = -(xz + Z*dx)/X = -z - (Z/X)dx - xz/x
    # But we need the inverse direction. From XZ + Zdx + Xdz = -xz_mod
    # dz = (-xz_mod - Z*dx) / X

    # Modified rc for reduced system: rc_mod = rc - xz/x (absorbing complementarity)
    # For predictor: xz = x*z, so rc_mod = rc - z
    # For corrector: xz = x*z - sigma*mu + dx_aff*dz_aff, so rc_mod = rc - z + sigma*mu/x - ...
    rc_mod = [rc[j] - xz[j] / max(x[j], eps) for j in range(n_total)]

    # Build A D A' (m x m)
    ADA = [[sum(A_aug[i][j] * d[j] * A_aug[k][j] for j in range(n_total))
            for k in range(m)] for i in range(m)]

    # RHS: -rb - A D rc_mod
    rhs = [-rb[i] - sum(A_aug[i][j] * d[j] * rc_mod[j] for j in range(n_total))
           for i in range(m)]

    # Solve ADA @ dy = rhs
    dy = _solve_cholesky(ADA, rhs, m, eps)
    if dy is None:
        return None, None, None

    # Back-substitute: dx = D (A'dy + rc_mod)
    dx = [d[j] * (sum(A_aug[i][j] * dy[i] for i in range(m)) + rc_mod[j])
          for j in range(n_total)]

    # Back-substitute: dz = (-xz - Z*dx) / X
    dz = [(-xz[j] - z[j] * dx[j]) / max(x[j], eps) for j in range(n_total)]

    return dx, dy, dz


def _solve_cholesky(M, b, n, eps):
    """Solve M @ x = b for symmetric positive definite M."""
    if n == 0:
        return []

    # Copy and regularize
    A = [[M[i][j] for j in range(n)] for i in range(n)]
    for i in range(n):
        A[i][i] += eps  # regularization

    # Cholesky: A = L L'
    L = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1):
            s = A[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                if s <= 0:
                    s = eps  # matrix not positive definite
                L[i][j] = sqrt(s)
            else:
                L[i][j] = s / max(L[j][j], eps)

    # Forward: L y = b
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(L[i][j] * y[j] for j in range(i))) / max(L[i][i], eps)

    # Backward: L' x = y
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(L[j][i] * x[j] for j in range(i + 1, n))) / max(L[i][i], eps)

    return x


def _step_length(v, dv, n):
    """Max alpha such that v + alpha*dv >= 0."""
    alpha = 1.0
    for j in range(n):
        if dv[j] < -1e-12:
            alpha = min(alpha, -v[j] / dv[j])
    return max(0.0, alpha)

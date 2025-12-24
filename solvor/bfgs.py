"""
BFGS Quasi-Newton Method for smooth optimization.

BFGS approximates the Hessian (second derivatives) using gradient history,
achieving superlinear convergence on smooth convex problems. It's faster than
gradient descent because it uses curvature information to take better steps.

    from solvor.bfgs import bfgs, lbfgs

    result = bfgs(grad_fn, x0)
    result = bfgs(grad_fn, x0, objective_fn=f)  # enables line search
    result = lbfgs(grad_fn, x0, m=10)  # memory-limited for large problems

How it works: maintains an approximation H of the inverse Hessian. Each step
computes direction d = -H @ grad, does line search, then updates H using the
BFGS formula. The approximation improves as it learns the local curvature.

Use this when:
- You have access to gradients (or can compute them)
- The objective is smooth (twice differentiable)
- Problem dimension is moderate (< 1000 for BFGS, use L-BFGS for larger)

Don't use for: non-smooth functions, very large dimensions (use L-BFGS),
or when gradients are expensive (consider derivative-free methods).
"""

from collections.abc import Callable, Sequence
from math import sqrt

from solvor.types import Progress, ProgressCallback, Result, Status

__all__ = ["bfgs", "lbfgs"]


def _dot(a: list[float], b: list[float]) -> float:
    """Dot product of two vectors."""
    return sum(ai * bi for ai, bi in zip(a, b))


def _mat_vec(H: list[list[float]], v: list[float]) -> list[float]:
    """Matrix-vector product H @ v."""
    n = len(v)
    return [sum(H[i][j] * v[j] for j in range(n)) for i in range(n)]


def _outer_update(
    H: list[list[float]],
    s: list[float],
    y: list[float],
    rho: float,
) -> None:
    """In-place BFGS update: H = (I - rho*s*y^T) @ H @ (I - rho*y*s^T) + rho*s*s^T."""
    n = len(s)

    # Compute H @ y
    Hy = _mat_vec(H, y)

    # Compute y^T @ H @ y
    yHy = _dot(y, Hy)

    # Compute s^T @ y (should equal 1/rho, but recompute for numerical stability)
    sy = _dot(s, y)

    # Update formula (Sherman-Morrison style):
    # H_new = H - (s @ Hy^T + Hy @ s^T) / sy + (1 + yHy/sy) * (s @ s^T) / sy
    factor = (1.0 + yHy / sy) / sy

    for i in range(n):
        for j in range(n):
            H[i][j] = H[i][j] - (s[i] * Hy[j] + Hy[i] * s[j]) / sy + factor * s[i] * s[j]


def _backtracking_line_search(
    x: list[float],
    direction: list[float],
    grad: list[float],
    objective_fn: Callable[[Sequence[float]], float],
    sign: int,
    c1: float = 1e-4,
    rho: float = 0.5,
    max_backtracks: int = 30,
) -> tuple[float, int]:
    """Backtracking line search with Armijo condition."""
    f_x = sign * objective_fn(x)  # For maximize, negate so we're minimizing -f
    # Slope is derivative in direction d: grad^T @ d
    # For minimize: we go in -grad direction, slope should be negative
    # For maximize: we go in +grad direction, but negate objective, so also negative
    slope = _dot(grad, direction)
    evals = 1

    alpha = 1.0
    for _ in range(max_backtracks):
        x_new = [x[i] + alpha * direction[i] for i in range(len(x))]
        f_new = sign * objective_fn(x_new)
        evals += 1

        if f_new <= f_x + c1 * alpha * slope:
            return alpha, evals

        alpha *= rho

    return alpha, evals


def bfgs(
    grad_fn: Callable[[Sequence[float]], Sequence[float]],
    x0: Sequence[float],
    *,
    minimize: bool = True,
    objective_fn: Callable[[Sequence[float]], float] | None = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
    on_progress: ProgressCallback | None = None,
    progress_interval: int = 0,
) -> Result:
    """
    BFGS quasi-Newton optimization.

    Args:
        grad_fn: Function computing gradient at a point
        x0: Initial guess
        minimize: If True minimize, else maximize
        objective_fn: Optional objective for line search (recommended)
        max_iter: Maximum iterations
        tol: Convergence tolerance on gradient norm
        on_progress: Optional progress callback
        progress_interval: How often to call progress callback

    Returns:
        Result with solution, final gradient norm, iterations, evaluations
    """
    sign = 1 if minimize else -1
    x = list(x0)
    n = len(x)
    evals = 0

    # Initialize inverse Hessian approximation as identity
    H = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    # Initial gradient
    grad = list(grad_fn(x))
    evals += 1

    for iteration in range(max_iter):
        grad_norm = sqrt(_dot(grad, grad))

        if grad_norm < tol:
            obj = objective_fn(x) if objective_fn else grad_norm
            return Result(x, obj, iteration, evals)

        # Compute search direction: d = -H @ grad (with sign adjustment)
        Hg = _mat_vec(H, grad)
        direction = [-sign * Hg[i] for i in range(n)]

        # Line search
        if objective_fn is not None:
            alpha, ls_evals = _backtracking_line_search(x, direction, grad, objective_fn, sign)
            evals += ls_evals
        else:
            # Without objective, use fixed step with gradient-based scaling
            alpha = 1.0 / (1.0 + grad_norm)

        # Update position
        s = [alpha * direction[i] for i in range(n)]
        x_new = [x[i] + s[i] for i in range(n)]

        # Compute new gradient
        grad_new = list(grad_fn(x_new))
        evals += 1

        # Compute y = grad_new - grad
        y = [sign * (grad_new[i] - grad[i]) for i in range(n)]

        # Update inverse Hessian if curvature condition is satisfied
        sy = _dot(s, y)
        if sy > 1e-10:  # Positive curvature
            _outer_update(H, s, y, 1.0 / sy)

        x = x_new
        grad = grad_new

        if on_progress and progress_interval > 0 and (iteration + 1) % progress_interval == 0:
            obj = objective_fn(x) if objective_fn else grad_norm
            progress = Progress(iteration + 1, obj, None, evals)
            if on_progress(progress) is True:
                return Result(x, obj, iteration + 1, evals, Status.FEASIBLE)

    grad_norm = sqrt(_dot(grad, grad))
    obj = objective_fn(x) if objective_fn else grad_norm
    return Result(x, obj, max_iter, evals, Status.MAX_ITER)


def lbfgs(
    grad_fn: Callable[[Sequence[float]], Sequence[float]],
    x0: Sequence[float],
    *,
    minimize: bool = True,
    objective_fn: Callable[[Sequence[float]], float] | None = None,
    m: int = 10,
    max_iter: int = 1000,
    tol: float = 1e-6,
    on_progress: ProgressCallback | None = None,
    progress_interval: int = 0,
) -> Result:
    """
    L-BFGS (Limited-memory BFGS) for large-scale optimization.

    Instead of storing the full n×n inverse Hessian approximation, L-BFGS
    stores only the last m gradient differences. This reduces memory from
    O(n²) to O(mn), making it suitable for high-dimensional problems.

    Args:
        grad_fn: Function computing gradient at a point
        x0: Initial guess
        minimize: If True minimize, else maximize
        objective_fn: Optional objective for line search (recommended)
        m: Number of correction pairs to store (default 10)
        max_iter: Maximum iterations
        tol: Convergence tolerance on gradient norm
        on_progress: Optional progress callback
        progress_interval: How often to call progress callback

    Returns:
        Result with solution, final gradient norm, iterations, evaluations
    """
    sign = 1 if minimize else -1
    x = list(x0)
    n = len(x)
    evals = 0

    # Storage for correction pairs
    s_history: list[list[float]] = []
    y_history: list[list[float]] = []
    rho_history: list[float] = []

    # Initial gradient
    grad = list(grad_fn(x))
    evals += 1

    for iteration in range(max_iter):
        grad_norm = sqrt(_dot(grad, grad))

        if grad_norm < tol:
            obj = objective_fn(x) if objective_fn else grad_norm
            return Result(x, obj, iteration, evals)

        # Compute search direction using two-loop recursion
        q = [sign * g for g in grad]
        k = len(s_history)
        alpha_list = [0.0] * k

        # First loop (backward)
        for i in range(k - 1, -1, -1):
            alpha_list[i] = rho_history[i] * _dot(s_history[i], q)
            q = [q[j] - alpha_list[i] * y_history[i][j] for j in range(n)]

        # Initial Hessian scaling
        if k > 0:
            sy = _dot(s_history[-1], y_history[-1])
            yy = _dot(y_history[-1], y_history[-1])
            gamma = sy / yy if yy > 1e-10 else 1.0
        else:
            gamma = 1.0 / (1.0 + grad_norm)

        r = [gamma * q[i] for i in range(n)]

        # Second loop (forward)
        for i in range(k):
            beta = rho_history[i] * _dot(y_history[i], r)
            r = [r[j] + (alpha_list[i] - beta) * s_history[i][j] for j in range(n)]

        direction = [-r[i] for i in range(n)]

        # Line search
        if objective_fn is not None:
            alpha, ls_evals = _backtracking_line_search(x, direction, grad, objective_fn, sign)
            evals += ls_evals
        else:
            alpha = 1.0

        # Update position
        s = [alpha * direction[i] for i in range(n)]
        x_new = [x[i] + s[i] for i in range(n)]

        # Compute new gradient
        grad_new = list(grad_fn(x_new))
        evals += 1

        # Compute y = grad_new - grad
        y = [sign * (grad_new[i] - grad[i]) for i in range(n)]

        # Store correction pair if curvature condition satisfied
        sy = _dot(s, y)
        if sy > 1e-10:
            if len(s_history) >= m:
                s_history.pop(0)
                y_history.pop(0)
                rho_history.pop(0)
            s_history.append(s)
            y_history.append(y)
            rho_history.append(1.0 / sy)

        x = x_new
        grad = grad_new

        if on_progress and progress_interval > 0 and (iteration + 1) % progress_interval == 0:
            obj = objective_fn(x) if objective_fn else grad_norm
            progress = Progress(iteration + 1, obj, None, evals)
            if on_progress(progress) is True:
                return Result(x, obj, iteration + 1, evals, Status.FEASIBLE)

    grad_norm = sqrt(_dot(grad, grad))
    obj = objective_fn(x) if objective_fn else grad_norm
    return Result(x, obj, max_iter, evals, Status.MAX_ITER)

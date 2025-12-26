"""
Bayesian Optimization, for when each evaluation is expensive.

Use this for hyperparameter tuning, A/B testing, simulation optimization, or any
black-box problem where you can only afford 20-100 evaluations. Works well in
low dimensions (5-15 parameters), builds a surrogate model to guess where to
sample next instead of brute-forcing the space.

Don't use this for: functions that are cheap to evaluate (just use anneal or genetic with
more iterations), high-dimensional problems (>20 dims, the surrogate struggles),
or discrete/categorical parameters without encoding tricks.

    from solvor.bayesian import bayesian_opt

    result = bayesian_opt(objective_fn, bounds=[(0, 1), (-5, 5)])
    result = bayesian_opt(objective_fn, bounds, minimize=False)  # maximize
    result = bayesian_opt(objective_fn, bounds, acquisition="ucb", kappa=2.5)  # UCB

If you're doing serious ML hyperparameter tuning, consider scikit-optimize or
Optuna, they handle the edge cases and integrations this implementation doesn't.
"""

from collections.abc import Callable, Sequence
from math import erf, exp, pi, sqrt
from random import Random

from solvor.nelder_mead import nelder_mead
from solvor.types import ProgressCallback, Result, Status
from solvor.utils.helpers import Evaluator, report_progress

__all__ = ["bayesian_opt"]


def bayesian_opt(
    objective_fn: Callable[[Sequence[float]], float],
    bounds: Sequence[tuple[float, float]],
    *,
    minimize: bool = True,
    max_iter: int = 50,
    n_initial: int = 5,
    acquisition: str = "ei",
    kappa: float = 2.0,
    acq_restarts: int = 6,
    seed: int | None = None,
    on_progress: ProgressCallback | None = None,
    progress_interval: int = 0,
) -> Result:
    """
    Bayesian optimization using Gaussian Process surrogate.

    Args:
        objective_fn: Function to optimize
        bounds: List of (lower, upper) bounds for each dimension
        minimize: If True minimize, else maximize
        max_iter: Total evaluations including initial points
        n_initial: Number of random initial points
        acquisition: Acquisition function - "ei" (Expected Improvement) or "ucb"
        kappa: UCB exploration parameter (only used if acquisition="ucb")
        acq_restarts: Number of random restarts for acquisition optimization
        seed: Random seed for reproducibility
        on_progress: Optional progress callback
        progress_interval: How often to call progress callback

    Returns:
        Result with best solution found
    """
    if acquisition not in ("ei", "ucb"):
        raise ValueError(f"acquisition must be 'ei' or 'ucb', got '{acquisition}'")

    rng = Random(seed)
    evaluate = Evaluator(objective_fn, minimize)

    def random_point():
        return [rng.uniform(lo, hi) for lo, hi in bounds]

    def clip_to_bounds(x):
        return [max(lo, min(hi, xi)) for xi, (lo, hi) in zip(x, bounds)]

    # Initial random samples
    xs = [random_point() for _ in range(n_initial)]
    ys = [evaluate(x) for x in xs]

    best_idx = min(range(len(ys)), key=lambda i: ys[i])
    best_solution, best_obj = xs[best_idx][:], ys[best_idx]

    length_scales = [(hi - lo) / 2 for lo, hi in bounds]

    def kernel(x1, x2):
        sq_dist = sum(((a - b) / ls) ** 2 for a, b, ls in zip(x1, x2, length_scales))
        return exp(-0.5 * sq_dist)

    def gp_predict(x_new, X, Y, L, noise=1e-6):
        n = len(X)
        if n == 0:
            return 0.0, 1.0

        cross_kernel = [kernel(x_new, X[i]) for i in range(n)]

        # Solve L @ alpha = y, then L.T @ mean_weights = alpha
        alpha = _forward_solve(L, Y)
        mean_weights = _backward_solve(L, alpha)

        # Solve L @ v = cross_kernel
        v = _forward_solve(L, cross_kernel)

        mu = sum(cross_kernel[i] * mean_weights[i] for i in range(n))
        var = kernel(x_new, x_new) - sum(v[i] * v[i] for i in range(n))
        var = max(var, 1e-10)

        return mu, sqrt(var)

    def expected_improvement(x, X, Y, L, best_y, xi=0.01):
        mu, sigma = gp_predict(x, X, Y, L)
        if sigma < 1e-10:
            return 0.0

        z = (best_y - mu - xi) / sigma
        ei = (best_y - mu - xi) * _norm_cdf(z) + sigma * _norm_pdf(z)
        return ei

    def ucb(x, X, Y, L, kappa_val):
        mu, sigma = gp_predict(x, X, Y, L)
        # For minimization, we want low mu and high uncertainty
        # UCB = -mu + kappa * sigma (we maximize this)
        return -mu + kappa_val * sigma

    def optimize_acquisition(X, Y, L, best_y):
        """Find point that maximizes acquisition using multi-start Nelder-Mead."""
        best_acq = -float("inf")
        best_candidate = None

        if acquisition == "ei":

            def neg_acq(x):
                return -expected_improvement(x, X, Y, L, best_y)
        else:

            def neg_acq(x):
                return -ucb(x, X, Y, L, kappa)

        for _ in range(acq_restarts):
            x0 = random_point()
            result = nelder_mead(neg_acq, x0, max_iter=50, tol=1e-4)

            if result.ok:
                candidate = clip_to_bounds(result.solution)
                acq_val = -neg_acq(candidate)
                if acq_val > best_acq:
                    best_acq = acq_val
                    best_candidate = candidate

        # Fallback to random if optimization failed
        if best_candidate is None:
            best_candidate = random_point()

        return best_candidate

    iteration = n_initial
    for iteration in range(n_initial, max_iter):
        # Build kernel matrix and compute Cholesky
        n = len(xs)
        noise = 1e-6
        K = [[kernel(xs[i], xs[j]) + (noise if i == j else 0) for j in range(n)] for i in range(n)]
        L = _cholesky(K)

        if L is None:
            # Cholesky failed, fall back to random sampling
            top_candidate = random_point()
        else:
            top_candidate = optimize_acquisition(xs, ys, L, best_obj)

        y_new = evaluate(top_candidate)
        xs.append(top_candidate)
        ys.append(y_new)

        if y_new < best_obj:
            best_solution, best_obj = top_candidate[:], y_new

        if report_progress(on_progress, progress_interval, iteration + 1,
                          evaluate.to_user(best_obj), evaluate.to_user(best_obj), evaluate.evals):
            return Result(best_solution, evaluate.to_user(best_obj), iteration + 1, evaluate.evals, Status.FEASIBLE)

    final_obj = evaluate.to_user(best_obj)
    status = Status.MAX_ITER if iteration >= max_iter - 1 else Status.OPTIMAL
    return Result(best_solution, final_obj, iteration + 1, evaluate.evals, status)


def _cholesky(A):
    """Cholesky decomposition: A = L @ L.T. Returns None if not positive definite."""
    n = len(A)
    L = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))

            if i == j:
                val = A[i][i] - s
                if val <= 0:
                    return None
                L[i][j] = sqrt(val)
            else:
                if abs(L[j][j]) < 1e-12:
                    return None
                L[i][j] = (A[i][j] - s) / L[j][j]

    return L


def _forward_solve(L, b):
    """Solve L @ x = b where L is lower triangular."""
    n = len(b)
    x = [0.0] * n
    for i in range(n):
        s = sum(L[i][j] * x[j] for j in range(i))
        if abs(L[i][i]) < 1e-12:
            x[i] = 0.0
        else:
            x[i] = (b[i] - s) / L[i][i]
    return x


def _backward_solve(L, b):
    """Solve L.T @ x = b where L is lower triangular (so L.T is upper)."""
    n = len(b)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(L[j][i] * x[j] for j in range(i + 1, n))
        if abs(L[i][i]) < 1e-12:
            x[i] = 0.0
        else:
            x[i] = (b[i] - s) / L[i][i]
    return x


def _norm_pdf(x):
    return exp(-0.5 * x * x) / sqrt(2 * pi)


def _norm_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))

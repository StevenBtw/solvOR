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

If you're doing serious ML hyperparameter tuning, consider scikit-optimize or
Optuna, they handle the edge cases and integrations this implementation doesn't.
"""

from collections.abc import Callable, Sequence
from math import sqrt, exp, pi, erf
from random import Random
from solvor.types import Status, Result

__all__ = ["bayesian_opt"]

def bayesian_opt(
    objective_fn: Callable[[Sequence[float]], float],
    bounds: Sequence[tuple[float, float]],
    *,
    minimize: bool = True,
    max_iter: int = 50,
    n_initial: int = 5,
    seed: int | None = None,
) -> Result:

    rng = Random(seed)
    sign = 1 if minimize else -1
    n_dims = len(bounds)
    evals = 0

    def evaluate(x):
        nonlocal evals
        evals += 1
        return sign * objective_fn(x)

    def random_point():
        return [rng.uniform(lo, hi) for lo, hi in bounds]

    xs = [random_point() for _ in range(n_initial)]
    ys = [evaluate(x) for x in xs]

    best_idx = min(range(len(ys)), key=lambda i: ys[i])
    best_x, best_y = xs[best_idx], ys[best_idx]

    length_scales = [(hi - lo) / 2 for lo, hi in bounds]

    def kernel(x1, x2):
        sq_dist = sum(((a - b) / ls) ** 2 for a, b, ls in zip(x1, x2, length_scales))
        return exp(-0.5 * sq_dist)

    def gp_predict(x_new, X, Y, noise=1e-6):
        n = len(X)
        if n == 0:
            return 0.0, 1.0

        K = [[kernel(X[i], X[j]) + (noise if i == j else 0) for j in range(n)] for i in range(n)]
        k_star = [kernel(x_new, X[i]) for i in range(n)]

        K_inv_y = _solve_linear(K, Y)
        K_inv_k = _solve_linear(K, k_star)

        mu = sum(k_star[i] * K_inv_y[i] for i in range(n))
        var = kernel(x_new, x_new) - sum(k_star[i] * K_inv_k[i] for i in range(n))
        var = max(var, 1e-10)

        return mu, sqrt(var)

    def expected_improvement(x, X, Y, best_y, xi=0.01):
        mu, sigma = gp_predict(x, X, Y)
        if sigma < 1e-10:
            return 0.0

        z = (best_y - mu - xi) / sigma
        ei = (best_y - mu - xi) * _norm_cdf(z) + sigma * _norm_pdf(z)
        return ei

    for iteration in range(n_initial, max_iter):
        best_ei, best_candidate = -float('inf'), None

        for _ in range(100 * n_dims):
            candidate = random_point()
            ei = expected_improvement(candidate, xs, ys, best_y)
            if ei > best_ei:
                best_ei, best_candidate = ei, candidate

        if best_candidate is None:
            best_candidate = random_point()

        y_new = evaluate(best_candidate)
        xs.append(best_candidate)
        ys.append(y_new)

        if y_new < best_y:
            best_x, best_y = best_candidate, y_new

    final_obj = best_y * sign
    return Result(best_x, final_obj, max_iter, evals, Status.FEASIBLE)

def _solve_linear(A, b):
    n = len(A)
    aug = [row[:] + [b[i]] for i, row in enumerate(A)]

    for col in range(n):
        max_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[max_row] = aug[max_row], aug[col]

        pivot = aug[col][col]
        if abs(pivot) < 1e-12:
            continue

        for j in range(col, n + 1):
            aug[col][j] /= pivot

        for row in range(n):
            if row != col:
                factor = aug[row][col]
                for j in range(col, n + 1):
                    aug[row][j] -= factor * aug[col][j]

    return [aug[i][n] for i in range(n)]

def _norm_pdf(x):
    return exp(-0.5 * x * x) / sqrt(2 * pi)

def _norm_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))
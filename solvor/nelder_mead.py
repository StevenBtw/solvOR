"""
Nelder-Mead simplex method for derivative-free optimization.

When you can't compute gradients (noisy simulations, black-box functions,
non-differentiable objectives), Nelder-Mead is your go-to. It maintains a
simplex of n+1 points in n dimensions, iteratively reflecting, expanding,
contracting, and shrinking to find the minimum.

    from solvor.nelder_mead import nelder_mead

    result = nelder_mead(objective_fn, [1.0, 2.0])
    result = nelder_mead(objective_fn, x0, adaptive=True)  # better for high dimensions

The algorithm doesn't use gradients, so it works on:
- Simulation outputs where derivatives aren't available
- Functions with discontinuities or noise
- Black-box optimization (hyperparameter tuning, etc.)

Limitations: slower than gradient methods for smooth functions, can stall
on high-dimensional problems (>20 variables). For global optimization,
combine with random restarts or use differential_evolution.
"""

from collections.abc import Callable, Sequence

from solvor.types import Progress, ProgressCallback, Result, Status

__all__ = ["nelder_mead"]


def nelder_mead(
    objective_fn: Callable[[Sequence[float]], float],
    x0: Sequence[float],
    *,
    minimize: bool = True,
    max_iter: int = 1000,
    tol: float = 1e-6,
    adaptive: bool = False,
    initial_step: float = 0.05,
    on_progress: ProgressCallback | None = None,
    progress_interval: int = 0,
) -> Result:
    """Derivative-free optimization using simplex reflections."""
    n = len(x0)
    sign = 1 if minimize else -1
    evals = 0

    # Nelder-Mead parameters
    if adaptive:
        # Adaptive parameters for high dimensions (Gao & Han, 2012)
        alpha = 1.0
        gamma = 1.0 + 2.0 / n
        rho = 0.75 - 1.0 / (2.0 * n)
        sigma = 1.0 - 1.0 / n
    else:
        # Standard parameters
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5  # contraction
        sigma = 0.5  # shrink

    def evaluate(x: list[float]) -> float:
        nonlocal evals
        evals += 1
        return sign * objective_fn(x)

    # Build initial simplex: x0 plus n vertices offset in each direction
    simplex: list[list[float]] = [list(x0)]
    for i in range(n):
        vertex = list(x0)
        step = initial_step if x0[i] == 0 else initial_step * abs(x0[i])
        vertex[i] += step
        simplex.append(vertex)

    # Evaluate all vertices
    values = [evaluate(v) for v in simplex]

    for iteration in range(1, max_iter + 1):
        # Sort vertices by objective value (best first)
        order = sorted(range(n + 1), key=lambda i: values[i])
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]

        best_val = values[0]
        worst_val = values[n]
        second_worst_val = values[n - 1]

        # Check convergence: spread of values
        spread = abs(worst_val - best_val)
        if spread < tol:
            break

        # Centroid of all but worst vertex
        centroid = [0.0] * n
        for i in range(n):
            for j in range(n):
                centroid[j] += simplex[i][j]
        for j in range(n):
            centroid[j] /= n

        # Reflect worst vertex through centroid
        worst = simplex[n]
        reflected = [centroid[j] + alpha * (centroid[j] - worst[j]) for j in range(n)]
        reflected_val = evaluate(reflected)

        if best_val <= reflected_val < second_worst_val:
            # Accept reflection
            simplex[n] = reflected
            values[n] = reflected_val

        elif reflected_val < best_val:
            # Try expansion
            expanded = [centroid[j] + gamma * (reflected[j] - centroid[j]) for j in range(n)]
            expanded_val = evaluate(expanded)

            if expanded_val < reflected_val:
                simplex[n] = expanded
                values[n] = expanded_val
            else:
                simplex[n] = reflected
                values[n] = reflected_val

        else:
            # Reflection worse than second-worst, try contraction
            if reflected_val < worst_val:
                # Outside contraction
                contracted = [centroid[j] + rho * (reflected[j] - centroid[j]) for j in range(n)]
                contracted_val = evaluate(contracted)

                if contracted_val <= reflected_val:
                    simplex[n] = contracted
                    values[n] = contracted_val
                else:
                    # Shrink toward best
                    _shrink(simplex, values, sigma, evaluate)
            else:
                # Inside contraction
                contracted = [centroid[j] + rho * (worst[j] - centroid[j]) for j in range(n)]
                contracted_val = evaluate(contracted)

                if contracted_val < worst_val:
                    simplex[n] = contracted
                    values[n] = contracted_val
                else:
                    # Shrink toward best
                    _shrink(simplex, values, sigma, evaluate)

        if on_progress and progress_interval > 0 and iteration % progress_interval == 0:
            best = values[0] * sign
            progress = Progress(iteration, best, None, evals)
            if on_progress(progress) is True:
                return Result(simplex[0], best, iteration, evals, Status.FEASIBLE)

    # Find best vertex
    best_idx = min(range(n + 1), key=lambda i: values[i])
    best_solution = simplex[best_idx]
    best_objective = values[best_idx] * sign

    status = Status.OPTIMAL if iteration < max_iter else Status.MAX_ITER
    return Result(best_solution, best_objective, iteration, evals, status)


def _shrink(
    simplex: list[list[float]],
    values: list[float],
    sigma: float,
    evaluate: Callable[[list[float]], float],
) -> None:
    """Shrink all vertices toward the best vertex."""
    n = len(simplex) - 1
    best = simplex[0]

    for i in range(1, n + 1):
        for j in range(n):
            simplex[i][j] = best[j] + sigma * (simplex[i][j] - best[j])
        values[i] = evaluate(simplex[i])

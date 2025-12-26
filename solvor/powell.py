"""
Powell's Conjugate Direction Method for derivative-free optimization.

Powell's method finds minima without computing gradients by doing sequential
line searches along a set of directions. It's often faster than Nelder-Mead
for smooth functions while still being derivative-free.

    from solvor.powell import powell

    result = powell(objective_fn, x0)
    result = powell(objective_fn, x0, bounds=[(-5, 5), (-5, 5)])

How it works: start with coordinate directions, do line search along each,
then replace one direction with the overall displacement. Over iterations,
the directions become conjugate (orthogonal in the Hessian metric), which
gives quadratic convergence for quadratic functions.

Use this when:
- You can't compute gradients (black-box function)
- The objective is smooth and continuous
- Nelder-Mead is too slow

Don't use for: noisy functions (try Nelder-Mead), non-smooth objectives,
or very high dimensions (the method needs O(n) line searches per iteration).
"""

from collections.abc import Callable, Sequence
from math import sqrt

from solvor.types import ProgressCallback, Result, Status
from solvor.utils.helpers import report_progress

__all__ = ["powell"]


def _bracket_minimum(
    f: Callable[[float], float],
    a: float = 0.0,
    b: float = 1.0,
    grow_factor: float = 1.618,
    max_iter: int = 50,
) -> tuple[float, float, float, int]:
    """Find bracket [a, b, c] containing a minimum. Returns (a, b, c, evals)."""
    fa = f(a)
    fb = f(b)
    evals = 2

    if fa < fb:
        a, b = b, a
        fa, fb = fb, fa

    c = b + grow_factor * (b - a)
    fc = f(c)
    evals += 1

    while fb > fc and evals < max_iter:
        a, b, c = b, c, c + grow_factor * (c - b)
        fa, fb = fb, fc
        fc = f(c)
        evals += 1

    if a > c:
        a, c = c, a

    return a, b, c, evals


def _golden_section_search(
    f: Callable[[float], float],
    a: float,
    c: float,
    tol: float = 1e-5,
    max_iter: int = 100,
) -> tuple[float, float, int]:
    """Golden section search for minimum in [a, c]. Returns (x_min, f_min, evals)."""
    phi = (1 + sqrt(5)) / 2
    resphi = 2 - phi

    b = a + resphi * (c - a)
    d = c - resphi * (c - a)

    fb = f(b)
    fd = f(d)
    evals = 2

    for _ in range(max_iter):
        if c - a < tol:
            break

        if fb < fd:
            c = d
            d = b
            fd = fb
            b = a + resphi * (c - a)
            fb = f(b)
        else:
            a = b
            b = d
            fb = fd
            d = c - resphi * (c - a)
            fd = f(d)
        evals += 1

    x_min = (a + c) / 2
    f_min = f(x_min)
    return x_min, f_min, evals + 1


def _line_search(
    objective_fn: Callable[[Sequence[float]], float],
    x: list[float],
    direction: list[float],
    sign: int,
    bounds: list[tuple[float, float]] | None = None,
) -> tuple[list[float], float, int]:
    """Line search along direction from x. Returns (x_new, f_new, evals)."""
    n = len(x)

    # Determine search range considering bounds
    alpha_min = -1e10
    alpha_max = 1e10

    if bounds is not None:
        for i in range(n):
            if abs(direction[i]) > 1e-12:
                # x[i] + alpha * direction[i] must be in [bounds[i][0], bounds[i][1]]
                lo, hi = bounds[i]
                if direction[i] > 0:
                    alpha_max = min(alpha_max, (hi - x[i]) / direction[i])
                    alpha_min = max(alpha_min, (lo - x[i]) / direction[i])
                else:
                    alpha_min = max(alpha_min, (hi - x[i]) / direction[i])
                    alpha_max = min(alpha_max, (lo - x[i]) / direction[i])

    # Clamp search range
    alpha_min = max(alpha_min, -1000.0)
    alpha_max = min(alpha_max, 1000.0)

    if alpha_min >= alpha_max:
        return x, objective_fn(x), 1

    def f_alpha(alpha: float) -> float:
        x_new = [x[i] + alpha * direction[i] for i in range(n)]
        return sign * objective_fn(x_new)

    # Bracket the minimum
    mid = (alpha_min + alpha_max) / 2
    a, _, c, bracket_evals = _bracket_minimum(f_alpha, mid, mid + 0.1 * (alpha_max - mid))

    # Clamp bracket to valid range
    a = max(a, alpha_min)
    c = min(c, alpha_max)

    # Golden section search
    alpha_opt, f_opt, search_evals = _golden_section_search(f_alpha, a, c)

    x_new = [x[i] + alpha_opt * direction[i] for i in range(n)]
    return x_new, sign * f_opt, bracket_evals + search_evals


def powell(
    objective_fn: Callable[[Sequence[float]], float],
    x0: Sequence[float],
    *,
    minimize: bool = True,
    bounds: Sequence[tuple[float, float]] | None = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
    on_progress: ProgressCallback | None = None,
    progress_interval: int = 0,
) -> Result:
    """
    Powell's conjugate direction method.

    Args:
        objective_fn: Function to optimize
        x0: Initial guess
        minimize: If True minimize, else maximize
        bounds: Optional bounds [(lo, hi), ...] for each dimension
        max_iter: Maximum iterations (each is n line searches)
        tol: Convergence tolerance on function value change
        on_progress: Optional progress callback
        progress_interval: How often to call progress callback

    Returns:
        Result with solution, objective value, iterations, function evaluations
    """
    sign = 1 if minimize else -1
    x = list(x0)
    n = len(x)
    evals = 0

    # Apply bounds to initial point
    if bounds is not None:
        bounds_list = [tuple(b) for b in bounds]
        for i in range(n):
            x[i] = max(bounds_list[i][0], min(bounds_list[i][1], x[i]))
    else:
        bounds_list = None

    # Initialize directions as coordinate axes
    directions = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    f_x = objective_fn(x)
    evals += 1

    for iteration in range(max_iter):
        f_start = f_x
        x_start = x.copy()

        # Track the direction with largest decrease
        max_decrease = 0.0
        max_decrease_idx = 0

        # Line search along each direction
        for i in range(n):
            f_before = f_x
            x, f_x, ls_evals = _line_search(objective_fn, x, directions[i], sign, bounds_list)
            evals += ls_evals

            decrease = f_before - f_x
            if decrease > max_decrease:
                max_decrease = decrease
                max_decrease_idx = i

        # Check convergence
        if abs(f_start - f_x) < tol * (1 + abs(f_x)):
            return Result(x, f_x, iteration, evals)

        # Compute overall displacement
        displacement = [x[i] - x_start[i] for i in range(n)]
        disp_norm = sqrt(sum(d * d for d in displacement))

        if disp_norm > 1e-12:
            # Normalize displacement
            displacement = [d / disp_norm for d in displacement]

            # Replace direction with largest decrease
            directions[max_decrease_idx] = displacement

            # Extra line search along new direction
            x, f_x, ls_evals = _line_search(objective_fn, x, displacement, sign, bounds_list)
            evals += ls_evals

        if report_progress(on_progress, progress_interval, iteration + 1, f_x, f_x, evals):
            return Result(x, f_x, iteration + 1, evals, Status.FEASIBLE)

    return Result(x, f_x, max_iter, evals, Status.MAX_ITER)

"""
Helper functions for optimization tasks, debugging, and evaluation.

Small, stable helpers used across solvers for common operations like objective
function wrapping, progress reporting, and solution manipulation.

    from solvor.utils import debug, Evaluator, report_progress
    from solvor.utils import random_permutation, pairwise_swap_neighbors
"""

from collections.abc import Callable, Iterator
from os import environ
from random import randint
from time import perf_counter

from solvor.types import Progress, ProgressCallback

__all__ = [
    "debug",
    "assignment_cost",
    "is_feasible",
    "random_permutation",
    "pairwise_swap_neighbors",
    "timed_progress",
    "default_progress",
    "Evaluator",
    "report_progress",
]

_DEBUG = bool(environ.get("DEBUG"))


def debug(*args, **kwargs) -> None:
    """Print only when DEBUG=1. Same signature as print()."""
    if _DEBUG:
        print(*args, **kwargs)


def assignment_cost(matrix: list[list[float]], assignment: list[int]) -> float:
    """Compute total cost of an assignment."""
    total = 0.0
    for i, j in enumerate(assignment):
        if j != -1 and i < len(matrix) and 0 <= j < len(matrix[i]):
            total += matrix[i][j]
    return total


def is_feasible(
    A: list[list[float]],
    b: list[float],
    x: list[float],
    tol: float = 1e-9,
) -> bool:
    """Check if x satisfies A @ x <= b within tolerance."""
    for i, row in enumerate(A):
        lhs = sum(row[j] * x[j] for j in range(min(len(row), len(x))))
        if lhs > b[i] + tol:
            return False
    return True


def random_permutation(n: int) -> list[int]:
    """Generate a random permutation of [0, 1, ..., n-1]."""
    perm = list(range(n))
    for i in range(n - 1, 0, -1):
        j = randint(0, i)
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def pairwise_swap_neighbors(perm: list[int]) -> Iterator[list[int]]:
    """Generate all neighbors by swapping pairs of elements."""
    n = len(perm)
    for i in range(n):
        for j in range(i + 1, n):
            neighbor = perm.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            yield neighbor


def timed_progress(
    callback: Callable[[Progress, float], bool | None],
) -> ProgressCallback:
    """Wrap a callback to receive elapsed time as second argument.

    Use this to add time tracking without modifying solver code.

    Example:
        def my_callback(progress, elapsed):
            print(f"iter {progress.iteration}, time {elapsed:.2f}s")
            return elapsed > 60  # Stop after 60 seconds

        result = solver(func, bounds, on_progress=timed_progress(my_callback))
    """
    start = perf_counter()

    def wrapper(progress: Progress) -> bool | None:
        elapsed = perf_counter() - start
        return callback(progress, elapsed)

    return wrapper


def default_progress(name: str = "", *, interval: int = 100, time_limit: float | None = None) -> ProgressCallback:
    """Create a default progress callback with formatted output.

    Args:
        name: Solver name prefix for output (optional)
        interval: Print every N iterations (default 100)
        time_limit: Stop after this many seconds (optional)

    Example:
        result = solver(func, bounds, on_progress=default_progress("PSO"))
        # Output: PSO iter=100 obj=1.234 best=0.567 time=0.42s
    """
    start = perf_counter()
    prefix = f"{name} " if name else ""

    def callback(progress: Progress) -> bool | None:
        elapsed = perf_counter() - start
        if progress.iteration % interval == 0:
            best = progress.best if progress.best is not None else progress.objective
            print(f"{prefix}iter={progress.iteration} obj={progress.objective:.6g} best={best:.6g} time={elapsed:.2f}s")
        if time_limit is not None and elapsed > time_limit:
            return True
        return None

    return callback


class Evaluator[T]:
    """Wraps objective function to track evaluations and handle minimize/maximize.

    The internal objective values are sign-adjusted so that minimization always
    means finding smaller values. Use `to_user()` to convert back for reporting.

    Example:
        evaluator = Evaluator(objective_fn, minimize=True)
        obj = evaluator(solution)  # Internal (signed) value
        user_obj = evaluator.to_user(obj)  # User-facing value
        print(f"Evaluations: {evaluator.evals}")
    """

    __slots__ = ("objective_fn", "sign", "evals")

    def __init__(self, objective_fn: Callable[[T], float], minimize: bool = True):
        self.objective_fn = objective_fn
        self.sign = 1 if minimize else -1
        self.evals = 0

    def __call__(self, sol: T) -> float:
        """Evaluate solution, returning internal (signed) value."""
        self.evals += 1
        return self.sign * self.objective_fn(sol)

    def to_user(self, internal_obj: float) -> float:
        """Convert internal objective to user-facing value."""
        return internal_obj * self.sign


def report_progress(
    on_progress: ProgressCallback | None,
    progress_interval: int,
    iteration: int,
    current_obj: float,
    best_obj: float,
    evals: int,
) -> bool:
    """Report progress if interval reached. Returns True if callback requested stop.

    Args:
        on_progress: Progress callback or None
        progress_interval: Report every N iterations (0 = disabled)
        iteration: Current iteration number
        current_obj: Current objective value (user-facing)
        best_obj: Best objective found so far (user-facing)
        evals: Number of objective evaluations

    Returns:
        True if callback returned True (stop requested), False otherwise.

    Example:
        if report_progress(on_progress, progress_interval, iteration,
                          evaluator.to_user(obj), evaluator.to_user(best), evaluator.evals):
            return Result(solution, evaluator.to_user(best), iteration, evaluator.evals, Status.FEASIBLE)
    """
    if not (on_progress and progress_interval > 0 and iteration % progress_interval == 0):
        return False

    progress = Progress(
        iteration,
        current_obj,
        best_obj if best_obj != current_obj else None,
        evals,
    )
    return on_progress(progress) is True

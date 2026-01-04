"""
Shared types for all solvers.

Core data structures returned by solvers and used for progress tracking.

    from solvor.types import Result, Status, Progress

    result = solve_lp(c, A, b)
    if result.ok:
        print(f"Solution: {result.solution}, cost: {result.objective}")
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum, auto
from os import environ

__all__ = ["Status", "Result", "Progress", "ProgressCallback"]

_DEBUG = bool(environ.get("DEBUG"))


class Status(IntEnum):
    """Solver outcome status."""

    OPTIMAL = auto()    # Proven optimal (exact solvers)
    FEASIBLE = auto()   # Feasible but not proven optimal (heuristics)
    INFEASIBLE = auto() # No feasible solution exists
    UNBOUNDED = auto()  # Objective can improve infinitely
    MAX_ITER = auto()   # Iteration limit reached


@dataclass(frozen=True, slots=True)
class Result[T]:
    """Solver result containing solution, objective, and metadata.

    Attributes:
        solution: The solution found (type varies by solver)
        objective: Objective value (cost, distance, makespan, etc.)
        iterations: Number of iterations performed
        evaluations: Number of objective function evaluations
        status: Outcome status (OPTIMAL, FEASIBLE, INFEASIBLE, etc.)
        error: Error message if status indicates failure
        solutions: Multiple solutions when solution_limit > 1
    """

    solution: T
    objective: float
    iterations: int = 0
    evaluations: int = 0
    status: Status = Status.OPTIMAL
    error: str | None = None
    solutions: tuple[T, ...] | None = None

    @property
    def ok(self) -> bool:
        """True if solution is usable (OPTIMAL or FEASIBLE)."""
        return self.status in (Status.OPTIMAL, Status.FEASIBLE)

    def log(self, prefix: str = "") -> "Result":
        """Print debug info if DEBUG=1. Returns self for chaining."""
        if _DEBUG:
            msg = f"{prefix}{self.status.name}: obj={self.objective}, iter={self.iterations}"
            if self.error:
                msg += f" - {self.error}"
            print(msg)
        return self


@dataclass(frozen=True, slots=True)
class Progress:
    """Solver progress info passed to callbacks.

    Attributes:
        iteration: Current iteration number
        objective: Current objective value
        best: Best objective found so far (None if same as objective)
        evaluations: Number of objective function evaluations
    """

    iteration: int
    objective: float
    best: float | None = None
    evaluations: int = 0


# Callback type for progress reporting. Return True to stop early, None/False to continue.
ProgressCallback = Callable[["Progress"], bool | None]

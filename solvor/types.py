"""Shared types for all solvers."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum, auto
from os import environ

__all__ = ["Status", "Result", "Progress", "ProgressCallback"]

_DEBUG = bool(environ.get("DEBUG"))

class Status(IntEnum):
    OPTIMAL = auto()
    FEASIBLE = auto()
    INFEASIBLE = auto()
    UNBOUNDED = auto()
    MAX_ITER = auto()

@dataclass(frozen=True, slots=True)
class Result:
    solution: object
    objective: float
    iterations: int = 0
    evaluations: int = 0
    status: Status = Status.OPTIMAL
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status in (Status.OPTIMAL, Status.FEASIBLE)

    def log(self, prefix: str = "") -> 'Result':
        if _DEBUG:
            msg = f"{prefix}{self.status.name}: obj={self.objective}, iter={self.iterations}"
            if self.error:
                msg += f" - {self.error}"
            print(msg)
        return self


@dataclass(frozen=True, slots=True)
class Progress:
    """Solver progress info passed to callbacks.

    iteration: Current iteration number
    objective: Current objective value
    best: Best objective found so far (None if same as objective)
    evaluations: Number of objective function evaluations
    """
    iteration: int
    objective: float
    best: float | None = None
    evaluations: int = 0


ProgressCallback = Callable[['Progress'], bool | None]
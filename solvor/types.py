"""Shared types for all solvers."""

from collections import namedtuple
from enum import IntEnum, auto

__all__ = ["Status", "Result"]

class Status(IntEnum):
    OPTIMAL = auto()
    FEASIBLE = auto()
    INFEASIBLE = auto()
    UNBOUNDED = auto()
    MAX_ITER = auto()

Result = namedtuple('Result', ['solution', 'objective', 'iterations', 'evaluations', 'status'])

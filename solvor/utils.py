"""
Utility functions for optimization tasks, debugging, and evaluation.

Small, stable helpers that don't change often. Use these for common operations
across solvers or for debugging and evaluation.

    from solvor.utils import debug, assignment_cost, is_feasible
"""

from collections.abc import Iterator
from os import environ
from random import randint

__all__ = [
    "debug",
    "assignment_cost",
    "is_feasible",
    "random_permutation",
    "pairwise_swap_neighbors",
    "fenwick_build",
    "fenwick_update",
    "fenwick_prefix",
]

_DEBUG = bool(environ.get("DEBUG"))


def debug(*args, **kwargs) -> None:
    """Print only when DEBUG=1. Same signature as print()."""
    if _DEBUG:
        print(*args, **kwargs)


def assignment_cost(matrix: list[list[float]], assignment: list[int]) -> float:
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
    for i, row in enumerate(A):
        lhs = sum(row[j] * x[j] for j in range(min(len(row), len(x))))
        if lhs > b[i] + tol:
            return False
    return True


def random_permutation(n: int) -> list[int]:
    perm = list(range(n))
    for i in range(n - 1, 0, -1):
        j = randint(0, i)
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def pairwise_swap_neighbors(perm: list[int]) -> Iterator[list[int]]:
    n = len(perm)
    for i in range(n):
        for j in range(i + 1, n):
            neighbor = perm.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            yield neighbor


def fenwick_build(values: list[float]) -> list[float]:
    n = len(values)
    tree = values.copy()
    for i in range(n):
        j = i | (i + 1)
        if j < n:
            tree[j] += tree[i]
    return tree


def fenwick_update(tree: list[float], i: int, delta: float) -> None:
    n = len(tree)
    while i < n:
        tree[i] += delta
        i |= i + 1


def fenwick_prefix(tree: list[float], i: int) -> float:
    total = 0.0
    while i >= 0:
        total += tree[i]
        i = (i & (i + 1)) - 1
    return total

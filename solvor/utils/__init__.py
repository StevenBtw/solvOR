"""
Utility functions and data structures for optimization.

    from solvor.utils import FenwickTree, UnionFind, debug
    from solvor.utils import check_matrix_dims, check_bounds
"""

from solvor.utils.data_structures import FenwickTree, UnionFind
from solvor.utils.helpers import (
    assignment_cost,
    debug,
    default_progress,
    is_feasible,
    pairwise_swap_neighbors,
    random_permutation,
    timed_progress,
)
from solvor.utils.validate import (
    check_bounds,
    check_edge_nodes,
    check_graph_nodes,
    check_in_range,
    check_integers_valid,
    check_matrix_dims,
    check_non_negative,
    check_positive,
    check_sequence_lengths,
    warn_large_coefficients,
)


# Backward-compatible function-based Fenwick Tree API
def fenwick_build(values: list[float]) -> list[float]:
    """Build a Fenwick tree from values. Returns the internal tree list."""
    n = len(values)
    tree = values.copy()
    for i in range(n):
        j = i | (i + 1)
        if j < n:
            tree[j] += tree[i]
    return tree


def fenwick_update(tree: list[float], i: int, delta: float) -> None:
    """Add delta to element at index i in the Fenwick tree."""
    n = len(tree)
    while i < n:
        tree[i] += delta
        i |= i + 1


def fenwick_prefix(tree: list[float], i: int) -> float:
    """Return sum of elements from index 0 to i (inclusive) in the Fenwick tree."""
    total = 0.0
    while i >= 0:
        total += tree[i]
        i = (i & (i + 1)) - 1
    return total


__all__ = [
    "FenwickTree",
    "UnionFind",
    "debug",
    "assignment_cost",
    "is_feasible",
    "random_permutation",
    "pairwise_swap_neighbors",
    "timed_progress",
    "default_progress",
    "check_matrix_dims",
    "check_sequence_lengths",
    "check_bounds",
    "check_positive",
    "check_non_negative",
    "check_in_range",
    "check_edge_nodes",
    "check_graph_nodes",
    "check_integers_valid",
    "warn_large_coefficients",
    # Backward-compatible function-based Fenwick API
    "fenwick_build",
    "fenwick_update",
    "fenwick_prefix",
]

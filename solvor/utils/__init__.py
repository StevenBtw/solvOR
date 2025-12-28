"""
Utility functions and data structures for optimization.

Data structures, validation helpers, and debugging utilities used internally
by solvers. Also available for custom implementations.

    from solvor.utils import UnionFind, FenwickTree
    from solvor.utils import check_matrix_dims, check_positive
    from solvor.utils import Evaluator, report_progress
"""

from solvor.utils.data_structures import FenwickTree, UnionFind
from solvor.utils.helpers import (
    Evaluator,
    assignment_cost,
    debug,
    default_progress,
    is_feasible,
    pairwise_swap_neighbors,
    random_permutation,
    report_progress,
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

__all__ = [
    "FenwickTree",
    "UnionFind",
    "Evaluator",
    "debug",
    "assignment_cost",
    "is_feasible",
    "random_permutation",
    "pairwise_swap_neighbors",
    "timed_progress",
    "default_progress",
    "report_progress",
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
]

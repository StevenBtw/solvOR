"""
Input validation utilities for solvers.

Common validation functions to ensure inputs are well-formed before solving.
Provides clear error messages for dimension mismatches, invalid values, and
other common mistakes.

    from solvor.utils import check_matrix_dims, check_bounds, check_positive

Used internally by solvers but also available for custom validation.
"""

from collections.abc import Sequence
from warnings import warn

__all__ = [
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


def check_matrix_dims(
    c: Sequence,
    A: Sequence[Sequence],
    b: Sequence,
    *,
    name_c: str = "c",
    name_A: str = "A",
    name_b: str = "b",
) -> None:
    """Validate dimensions of LP/MILP inputs: c (n,), A (m, n), b (m,)."""
    if not A:
        raise ValueError(f"Constraint matrix {name_A} cannot be empty")

    n = len(c)
    m = len(b)

    if len(A) != m:
        raise ValueError(f"Dimension mismatch: {name_b} has {m} constraints but {name_A} has {len(A)} rows")

    for i, row in enumerate(A):
        if len(row) != n:
            raise ValueError(
                f"Dimension mismatch: {name_c} has {n} variables but {name_A} row {i} has {len(row)} columns"
            )


def check_sequence_lengths(
    *seqs: tuple[Sequence, str],
    expected: int | None = None,
) -> int:
    """Check that all sequences have the same length. Returns the length."""
    if not seqs:
        return 0

    if expected is None:
        expected = len(seqs[0][0])

    for seq, name in seqs:
        if len(seq) != expected:
            raise ValueError(f"Length mismatch: expected {expected} elements in {name}, got {len(seq)}")

    return expected


def check_bounds(
    bounds: Sequence[tuple[float, float]],
    *,
    name: str = "bounds",
) -> int:
    """Validate bounds are (low, high) pairs with low <= high. Returns dimension."""
    n = len(bounds)
    for i, (lo, hi) in enumerate(bounds):
        if lo > hi:
            raise ValueError(f"Invalid {name}[{i}]: lower bound {lo} > upper bound {hi}")
    return n


def check_positive(
    value: float,
    *,
    name: str = "value",
) -> None:
    """Check that value is strictly positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def check_non_negative(
    value: float,
    *,
    name: str = "value",
) -> None:
    """Check that value is non-negative."""
    if value < 0:
        raise ValueError(f"{name} cannot be negative, got {value}")


def check_in_range(
    value: float,
    low: float,
    high: float,
    *,
    name: str = "value",
    inclusive: bool = True,
) -> None:
    """Check that value is within [low, high] or (low, high)."""
    if inclusive:
        if not (low <= value <= high):
            raise ValueError(f"{name} must be in [{low}, {high}], got {value}")
    else:
        if not (low < value < high):
            raise ValueError(f"{name} must be in ({low}, {high}), got {value}")


def warn_large_coefficients(
    A: Sequence[Sequence[float]],
    threshold: float = 1e10,
    *,
    name: str = "A",
) -> None:
    """Warn if matrix has very large coefficients that may cause numerical issues."""
    max_val = 0.0
    for row in A:
        for val in row:
            abs_val = abs(val)
            if abs_val > max_val:
                max_val = abs_val

    if max_val > threshold:
        warn(
            f"Large coefficients in {name} (max={max_val:.2e}) may cause numerical issues. "
            "Consider scaling your problem.",
            stacklevel=3,
        )


def check_graph_nodes(
    graph: dict,
    *nodes: tuple[object, str],
) -> None:
    """Check that nodes exist in graph."""
    for node, name in nodes:
        if node not in graph:
            raise ValueError(f"{name} '{node}' not found in graph")


def check_integers_valid(
    integers: Sequence[int],
    n_vars: int,
    *,
    name: str = "integers",
) -> None:
    """Check that integer variable indices are valid."""
    for idx in integers:
        if not isinstance(idx, int):
            raise TypeError(f"{name} must contain integers, got {type(idx).__name__}")
        if idx < 0 or idx >= n_vars:
            raise ValueError(f"Invalid index in {name}: {idx} (valid range: 0 to {n_vars - 1})")


def check_edge_nodes(
    edges: Sequence[tuple[int, int, float]],
    n_nodes: int,
    *,
    name: str = "edges",
) -> None:
    """Check that edge endpoints are valid node indices."""
    for i, (u, v, _) in enumerate(edges):
        if u < 0 or u >= n_nodes:
            raise ValueError(f"Invalid node in {name}[{i}]: u={u} (valid range: 0 to {n_nodes - 1})")
        if v < 0 or v >= n_nodes:
            raise ValueError(f"Invalid node in {name}[{i}]: v={v} (valid range: 0 to {n_nodes - 1})")

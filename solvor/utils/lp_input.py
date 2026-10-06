"""
Input normalization for LP and MILP solvers.

Turns the public forms of a linear problem (dense or sparse constraint rows,
per-row senses, variable bounds) into one internal form that the solvers
share. Validation happens here, once per public call.

    from solvor.utils.lp_input import normalize_lp

    prob = normalize_lp(c, A, b, lb=lb, ub=ub, senses=senses)
    prob.rows[0]      # {column: coefficient}, zeros dropped
    prob.columns()    # per column: [(row, coefficient), ...]
"""

from collections.abc import Mapping, Sequence
from math import inf, isnan
from typing import NamedTuple
from warnings import warn

__all__ = ["LinearProblem", "SENSES", "normalize_bounds", "normalize_lp", "normalize_rows"]

SENSES = ("<=", ">=", "=")


class LinearProblem(NamedTuple):
    """Normalized linear constraints: rows[i] . x (senses[i]) b[i], lb <= x <= ub."""

    n: int
    rows: list[dict[int, float]]
    b: list[float]
    senses: list[str]
    lb: list[float]
    ub: list[float]

    def columns(self) -> list[list[tuple[int, float]]]:
        """Column-wise view: for each variable, the (row, coefficient) pairs where it appears."""
        cols: list[list[tuple[int, float]]] = [[] for _ in range(self.n)]
        for i, row in enumerate(self.rows):
            for j, a in row.items():
                cols[j].append((i, a))
        return cols


def normalize_lp(
    c: Sequence[float],
    A: Sequence[Sequence[float] | Mapping[int, float]],
    b: Sequence[float],
    *,
    lb: Sequence[float] | None = None,
    ub: Sequence[float] | None = None,
    senses: Sequence[str] | None = None,
) -> LinearProblem:
    """Validate and normalize LP/MILP input.

    Each row of A is either a dense sequence of length len(c) or a mapping
    {column: coefficient}. Rows may mix both forms. senses defaults to "<="
    for every row; lb defaults to 0.0 and ub to +inf for every variable.
    """
    n = len(c)
    if any(isnan(float(v)) for v in c):
        raise ValueError("c contains NaN")
    rows, rhs, sense_list = normalize_rows(n, A, b, senses, stacklevel=4)
    lower, upper = normalize_bounds(n, lb, ub)
    return LinearProblem(n, rows, rhs, sense_list, lower, upper)


def normalize_rows(
    n: int,
    A: Sequence[Sequence[float] | Mapping[int, float]],
    b: Sequence[float],
    senses: Sequence[str] | None = None,
    *,
    stacklevel: int = 3,
) -> tuple[list[dict[int, float]], list[float], list[str]]:
    """Validate constraint rows for n variables; returns (sparse rows, rhs, senses)."""
    m = len(b)
    if len(A) != m:
        raise ValueError(f"Dimension mismatch: b has {m} constraints but A has {len(A)} rows")

    rows: list[dict[int, float]] = []
    max_abs = 0.0
    for i, row in enumerate(A):
        sparse: dict[int, float] = {}
        if isinstance(row, Mapping):
            for j, v in row.items():
                if not isinstance(j, int):
                    raise TypeError(f"A row {i} has a non-integer column key {j!r}")
                if not 0 <= j < n:
                    raise ValueError(f"A row {i} has column {j} outside the valid range 0 to {n - 1}")
                fv = float(v)
                if fv != 0.0:
                    sparse[j] = fv
        else:
            if len(row) != n:
                raise ValueError(f"Dimension mismatch: c has {n} variables but A row {i} has {len(row)} columns")
            for j, v in enumerate(row):
                fv = float(v)
                if fv != 0.0:
                    sparse[j] = fv
        for fv in sparse.values():
            if isnan(fv):
                raise ValueError(f"A row {i} contains NaN")
            if abs(fv) > max_abs:
                max_abs = abs(fv)
        rows.append(sparse)

    if max_abs > 1e10:
        warn(
            f"Large coefficients in A (max={max_abs:.2e}) may cause numerical issues. Consider scaling your problem.",
            stacklevel=stacklevel,
        )

    rhs = [float(v) for v in b]
    if any(isnan(v) for v in rhs):
        raise ValueError("b contains NaN")

    if senses is None:
        sense_list = ["<="] * m
    else:
        sense_list = list(senses)
        if len(sense_list) != m:
            raise ValueError(f"Length mismatch: expected {m} elements in senses, got {len(sense_list)}")
        for i, s in enumerate(sense_list):
            if s not in SENSES:
                raise ValueError(f"senses[{i}] must be one of '<=', '>=', '=', got {s!r}")
    return rows, rhs, sense_list


def normalize_bounds(
    n: int, lb: Sequence[float] | None = None, ub: Sequence[float] | None = None
) -> tuple[list[float], list[float]]:
    """Validate variable bounds; defaults 0.0 and +inf. lb = +inf or ub = -inf leaves no value, so it is rejected."""
    return _bounds(lb, n, 0.0, "lb", inf), _bounds(ub, n, inf, "ub", -inf)


def _bounds(values: Sequence[float] | None, n: int, default: float, name: str, impossible: float) -> list[float]:
    if values is None:
        return [default] * n
    out = [float(v) for v in values]
    if len(out) != n:
        raise ValueError(f"Length mismatch: expected {n} elements in {name}, got {len(out)}")
    if any(isnan(v) for v in out):
        raise ValueError(f"{name} contains NaN")
    if impossible in out:
        raise ValueError(f"{name} contains {impossible}")
    return out

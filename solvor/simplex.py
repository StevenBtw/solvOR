r"""
Simplex Solver, linear programming that aged like wine.

You're walking along edges of a giant crystal always uphill, until you hit a corner
that's optimal. You can visualize it. Most algorithms are abstract symbol
manipulation. Simplex is a journey through space.

    from solvor.simplex import solve_lp

    # minimize c @ x, subject to A @ x <= b, x >= 0
    result = solve_lp(c, A, b)
    result = solve_lp(c, A, b, minimize=False)  # maximize

    # sparse rows, other senses and variable bounds
    result = solve_lp(c, [{0: 1.0, 2: 3.0}, {1: 1.0}], [4.0, 2.0], senses=[">=", "="],
                      lb=[0, -1, 0], ub=[5, 1, float("inf")])

How it works: starts at a vertex of the feasible polytope (phase 1 finds one if
needed). Each iteration pivots to an adjacent vertex with better objective value.
Variable bounds are handled inside the ratio test (a variable can also jump from
one bound to the other without a pivot), so they cost no extra rows. Bland's rule
prevents cycling. Terminates when no improving neighbor exists.

Use this for:

- Linear objectives with linear constraints
- Resource allocation, blending, production planning
- Transportation and assignment problems
- When you need exact optimum (not approximate)

Parameters:

    c: objective coefficients (minimize c @ x)
    A: constraint rows, dense lists or sparse {column: coefficient} dicts
    b: right-hand sides
    minimize: True for min, False for max (default: True)
    senses: per row "<=", ">=" or "=" (default: all "<=")
    lb, ub: variable bounds (default: 0 and +inf; use float("-inf") / float("inf") for none)
    backend: "auto", "rust", or "python" (default: "auto")

This also does the grunt work inside MILP, solving LP relaxations at each node.

Backend: the simplex tableau has an optional Rust backend (3-100x faster) that
returns the same results, bit for bit. Use backend="python" for the pure Python
implementation.

Don't use this for: integer constraints (use MILP), non-linear objectives
(use gradient or anneal), or problems with poor numerical scaling (simplex
can struggle with badly scaled coefficients).
"""

from collections.abc import Mapping, Sequence
from typing import Literal

from solvor.lp_engine import solve_cold
from solvor.types import Result, Status  # noqa: F401  (Status stays importable from here, as in 0.6.2)
from solvor.utils.lp_input import normalize_lp
from solvor.utils.validate import check_non_negative

__all__ = ["solve_lp"]


def solve_lp(
    c: Sequence[float],
    A: Sequence[Sequence[float] | Mapping[int, float]],
    b: Sequence[float],
    *,
    minimize: bool = True,
    eps: float = 1e-10,
    max_iter: int = 100_000,
    senses: Sequence[str] | None = None,
    lb: Sequence[float] | None = None,
    ub: Sequence[float] | None = None,
    backend: Literal["auto", "rust", "python"] | None = None,
) -> Result:
    """Solve linear program: minimize c @ x subject to A @ x (senses) b, lb <= x <= ub."""
    check_non_negative(eps, name="eps")
    prob = normalize_lp(c, A, b, lb=lb, ub=ub, senses=senses)
    cost = [float(v) for v in c]
    return solve_cold(prob, cost, prob.lb, prob.ub, minimize=minimize, eps=eps, max_iter=max_iter, backend=backend)

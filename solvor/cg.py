r"""
Column Generation for problems with exponentially many variables.

Gilmore-Gomory's elegant insight: instead of enumerating all columns upfront
(exponentially many), generate them on demand. The master problem asks
"given these columns, what's the optimal combination?" The pricing subproblem
asks "is there a column worth adding?" Repeat until no improving column exists.

    from solvor import solve_cg

    # Cutting stock: minimize rolls to cut required pieces
    result = solve_cg(
        demands=[97, 610, 395, 211],
        roll_width=100,
        piece_sizes=[45, 36, 31, 14],
    )
    print(f"Rolls needed: {result.objective}")  # 454 rolls
    print(result.solution)  # {pattern: count, ...}

    # Generic column generation with custom pricing
    def my_pricing(duals):
        # Return (column, reduced_cost) or (None, 0) if no improving column
        ...

    result = solve_cg(
        demands=[10, 20, 30],
        pricing_fn=my_pricing,
        initial_columns=[(1,0,0), (0,1,0), (0,0,1)],
    )

How it works:

1. Start with initial patterns (one piece type per roll, max copies)
2. Solve master LP to get dual prices (shadow prices for constraints)
3. Pricing subproblem finds column with negative reduced cost
   - For cutting stock: bounded knapsack with dual values as profits
   - For custom: user-provided pricing function
4. If reduced cost < 0, add column and repeat; else done
5. Round LP solution for integer result

Use this for:

- Cutting stock (paper, steel, glass, fabric)
- Bin packing (reformulated as set covering)
- Vehicle routing (routes as columns)
- Crew scheduling (shifts as columns)
- Graph coloring (independent sets as columns)

Parameters:

    roll_width: capacity of each roll (cutting stock mode)
    piece_sizes: sizes of piece types to cut (cutting stock mode)
    demands: number of each piece/constraint to satisfy
    pricing_fn: custom pricing callable(duals) -> (column, reduced_cost)
    initial_columns: starting columns for custom pricing

The LP relaxation provides a lower bound. Rounding up gives a feasible integer
solution. For cutting stock, this gap is typically small (often zero).
"""

from collections.abc import Callable, Sequence
from math import ceil

from solvor.types import ProgressCallback, Result, Status
from solvor.utils.helpers import report_progress
from solvor.utils.pricing import greedy_knapsack, knapsack_pricing, simplex_phase
from solvor.utils.validate import check_non_negative, check_positive, check_sequence_lengths

__all__ = ["solve_cg"]

# Type alias for pricing function
PricingFn = Callable[[list[float]], tuple[tuple[int, ...] | None, float]]


def solve_cg(
    demands: Sequence[int],
    *,
    roll_width: float | None = None,
    piece_sizes: Sequence[float] | None = None,
    pricing_fn: PricingFn | None = None,
    initial_columns: Sequence[Sequence[int]] | None = None,
    max_iter: int = 1000,
    eps: float = 1e-9,
    on_progress: ProgressCallback | None = None,
    progress_interval: int = 0,
) -> Result[dict[tuple[int, ...], int]]:
    """Solve set covering problem via column generation.

    Two modes:
    - Cutting stock: provide roll_width and piece_sizes
    - Custom: provide pricing_fn and initial_columns

    Returns Result with solution as dict mapping column tuples to usage counts.
    """
    m = len(demands)
    if m == 0:
        return Result({}, 0.0, 0, 0, Status.OPTIMAL)

    for i, d in enumerate(demands):
        check_non_negative(d, name=f"demands[{i}]")

    if all(d == 0 for d in demands):
        return Result({}, 0.0, 0, 0, Status.OPTIMAL)

    # Determine mode
    cutting_stock = roll_width is not None and piece_sizes is not None
    custom = pricing_fn is not None

    if cutting_stock and custom:
        raise ValueError("Provide (roll_width, piece_sizes) or (pricing_fn), not both")
    if not cutting_stock and not custom:
        raise ValueError("Provide (roll_width, piece_sizes) or (pricing_fn)")

    if cutting_stock:
        assert roll_width is not None and piece_sizes is not None
        return _solve_cutting_stock(roll_width, piece_sizes, demands, max_iter, eps, on_progress, progress_interval)

    assert pricing_fn is not None
    if initial_columns is None:
        raise ValueError("Custom mode requires initial_columns")
    return _solve_custom(demands, pricing_fn, initial_columns, max_iter, eps, on_progress, progress_interval)


def _solve_cutting_stock(roll_width, piece_sizes, demands, max_iter, eps, on_progress, progress_interval):
    """Cutting stock via column generation with knapsack pricing."""
    n = check_sequence_lengths((piece_sizes, "piece_sizes"), (demands, "demands"))

    for i, size in enumerate(piece_sizes):
        check_positive(size, name=f"piece_sizes[{i}]")
        if size > roll_width:
            raise ValueError(f"piece_sizes[{i}]={size} exceeds roll_width={roll_width}")

    # Initial patterns: max copies of each piece type per roll
    patterns: list[tuple[int, ...]] = []
    for j in range(n):
        if demands[j] > 0:
            count = int(roll_width // piece_sizes[j])
            pattern = tuple(count if i == j else 0 for i in range(n))
            patterns.append(pattern)

    iteration = 0
    lp_obj = float("inf")

    while iteration < max_iter:
        x_vals, duals, lp_obj = _solve_master_lp(patterns, demands, eps)

        if report_progress(on_progress, progress_interval, iteration, lp_obj, lp_obj, iteration):
            break

        new_pattern, pricing_value = knapsack_pricing(piece_sizes, roll_width, duals, eps)

        # Reduced cost = 1 - pricing_value; stop if >= 0
        if pricing_value <= 1.0 + eps:
            break

        if new_pattern not in patterns:
            patterns.append(new_pattern)

        iteration += 1

    # Final LP solve
    x_vals, duals, lp_obj = _solve_master_lp(patterns, demands, eps)

    # Round up to integer solution
    solution: dict[tuple[int, ...], int] = {}
    total_rolls = 0
    for pattern, x in zip(patterns, x_vals):
        if x > eps:
            count = ceil(x - eps)
            if count > 0:
                solution[pattern] = count
                total_rolls += count

    # Verify demands met
    for i in range(n):
        produced = sum(p[i] * cnt for p, cnt in solution.items())
        if produced < demands[i]:
            return Result(
                solution,
                float(total_rolls),
                iteration,
                iteration,
                Status.INFEASIBLE,
                error=f"Demand not met for piece {i}: {produced} < {demands[i]}",
            )

    lb = ceil(lp_obj - eps)
    status = Status.OPTIMAL if total_rolls <= lb else Status.FEASIBLE

    return Result(solution, float(total_rolls), iteration, iteration, status)


def _solve_custom(demands, pricing_fn, initial_columns, max_iter, eps, on_progress, progress_interval):
    """Generic column generation with user-provided pricing."""
    m = len(demands)

    for i, col in enumerate(initial_columns):
        if len(col) != m:
            raise ValueError(f"column {i} has wrong length: {len(col)} vs {m}")

    columns: list[tuple[int, ...]] = [tuple(c) for c in initial_columns]
    column_set: set[tuple[int, ...]] = set(columns)

    iteration = 0
    lp_obj = float("inf")

    while iteration < max_iter:
        x_vals, duals, lp_obj = _solve_master_lp(columns, demands, eps)

        if report_progress(on_progress, progress_interval, iteration, lp_obj, lp_obj, iteration):
            break

        new_col, reduced_cost = pricing_fn(duals)

        if new_col is None or reduced_cost >= -eps:
            break

        new_col_tuple = tuple(new_col)
        if new_col_tuple not in column_set:
            columns.append(new_col_tuple)
            column_set.add(new_col_tuple)

        iteration += 1

    # Final solve
    x_vals, duals, lp_obj = _solve_master_lp(columns, demands, eps)

    # Round up
    solution: dict[tuple[int, ...], int] = {}
    total = 0
    for col, x in zip(columns, x_vals):
        if x > eps:
            count = ceil(x - eps)
            if count > 0:
                solution[col] = count
                total += count

    lb = ceil(lp_obj - eps)
    status = Status.OPTIMAL if total <= lb else Status.FEASIBLE

    return Result(solution, float(total), iteration, iteration, status)


def _solve_master_lp(
    columns: list[tuple[int, ...]],
    demands: Sequence[int],
    eps: float,
) -> tuple[list[float], list[float], float]:
    """Solve master LP: min sum(x) s.t. Ax >= b, x >= 0.

    Returns (primal_values, dual_values, objective).
    """
    m = len(demands)
    n = len(columns)

    if n == 0:
        return [], [0.0] * m, float("inf")

    # Two-phase simplex
    n_vars = n + m + m  # x, surplus, artificial
    n_rows = m

    # Tableau [A | -I | I | b]
    tab = [[0.0] * (n_vars + 1) for _ in range(n_rows + 1)]

    for i in range(m):
        for j, col in enumerate(columns):
            tab[i][j] = float(col[i])
        tab[i][n + i] = -1.0  # surplus
        tab[i][n + m + i] = 1.0  # artificial
        tab[i][-1] = float(demands[i])

    # Phase 1: minimize artificials
    for i in range(m):
        for j in range(n_vars + 1):
            tab[-1][j] -= tab[i][j]
        tab[-1][n + m + i] = 0.0

    basis = list(range(n + m, n + 2 * m))

    simplex_phase(tab, basis, n + m, n_rows, eps)

    if tab[-1][-1] < -eps:
        return [0.0] * n, [0.0] * m, float("inf")

    # Phase 2: minimize sum of x (all costs = 1)
    for j in range(n_vars + 1):
        tab[-1][j] = 0.0
    for j in range(n):
        tab[-1][j] = 1.0

    for i, b in enumerate(basis):
        cost = 1.0 if b < n else 0.0
        if abs(cost) > eps:
            for j in range(n_vars + 1):
                tab[-1][j] -= cost * tab[i][j]

    simplex_phase(tab, basis, n + m, n_rows, eps)

    x_vals = [0.0] * n
    for i, b in enumerate(basis):
        if b < n:
            x_vals[b] = max(0.0, tab[i][-1])

    # Dual = reduced cost of surplus variable (for >= constraints in minimization)
    duals = [tab[-1][n + i] for i in range(m)]
    objective = -tab[-1][-1]

    return x_vals, duals, objective

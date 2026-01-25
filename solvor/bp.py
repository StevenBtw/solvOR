r"""
Branch-and-Price for integer solutions to problems with exponentially many variables.

Column generation finds optimal LP relaxations. Branch-and-price embeds it in
branch-and-bound to find provably optimal integer solutions.

    from solvor import solve_bp

    # Cutting stock: minimize rolls with guaranteed integer optimality
    result = solve_bp(
        demands=[97, 610, 395, 211],
        roll_width=100,
        piece_sizes=[45, 36, 31, 14],
    )
    print(f"Rolls needed: {result.objective}")  # Integer optimal
    print(result.solution)  # {pattern: count, ...}

    # Custom pricing with branch-and-price
    def my_pricing(duals):
        # Return (column, reduced_cost) or (None, 0) if no improving column
        ...

    result = solve_bp(
        demands=[10, 20, 30],
        pricing_fn=my_pricing,
        initial_columns=[(1,0,0), (0,1,0), (0,0,1)],
    )

How it works:

1. Root node: solve LP relaxation via column generation
2. If LP solution is integer, done (optimal)
3. Otherwise, branch on fractional pattern variable
4. Each child node: re-solve column generation with branching constraints
5. Global column pool shared across all nodes
6. Best-first search prioritizes promising nodes

Use this for:

- Cutting stock when optimal integer solution matters
- Bin packing with guaranteed optimality
- Any column generation problem requiring integer solution

Parameters:

    demands: number of each piece/constraint to satisfy
    roll_width: capacity of each roll (cutting stock mode)
    piece_sizes: sizes of piece types to cut (cutting stock mode)
    pricing_fn: custom pricing callable(duals) -> (column, reduced_cost)
    initial_columns: starting columns for custom pricing
    max_nodes: branch-and-bound node limit (default 10000)
    gap_tol: optimality gap tolerance (default 1e-6)

The B&B tree explores nodes in best-first order, pruning branches that cannot
improve the incumbent. Returns proven optimal when gap reaches zero.
"""

from collections.abc import Callable, Sequence
from heapq import heappop, heappush
from math import ceil, floor
from typing import NamedTuple

from solvor.types import ProgressCallback, Result, Status
from solvor.utils.helpers import report_progress
from solvor.utils.validate import check_non_negative, check_positive, check_sequence_lengths

__all__ = ["solve_bp"]

PricingFn = Callable[[list[float]], tuple[tuple[int, ...] | None, float]]


class _BPNode(NamedTuple):
    bound: float
    column_bounds: tuple[tuple[int, float, float], ...]  # (col_idx, lower, upper)
    depth: int


def solve_bp(
    demands: Sequence[int],
    *,
    roll_width: float | None = None,
    piece_sizes: Sequence[float] | None = None,
    pricing_fn: PricingFn | None = None,
    initial_columns: Sequence[Sequence[int]] | None = None,
    max_iter: int = 1000,
    max_nodes: int = 10000,
    gap_tol: float = 1e-6,
    eps: float = 1e-9,
    on_progress: ProgressCallback | None = None,
    progress_interval: int = 0,
) -> Result[dict[tuple[int, ...], int]]:
    """Solve set covering problem via branch-and-price.

    Two modes:
    - Cutting stock: provide roll_width and piece_sizes
    - Custom: provide pricing_fn and initial_columns

    Returns Result with solution as dict mapping column tuples to integer usage counts.
    """
    m = len(demands)
    if m == 0:
        return Result({}, 0.0, 0, 0, Status.OPTIMAL)

    for i, d in enumerate(demands):
        check_non_negative(d, name=f"demands[{i}]")

    if all(d == 0 for d in demands):
        return Result({}, 0.0, 0, 0, Status.OPTIMAL)

    cutting_stock = roll_width is not None and piece_sizes is not None
    custom = pricing_fn is not None

    if cutting_stock and custom:
        raise ValueError("Provide (roll_width, piece_sizes) or (pricing_fn), not both")
    if not cutting_stock and not custom:
        raise ValueError("Provide (roll_width, piece_sizes) or (pricing_fn)")

    if cutting_stock:
        assert roll_width is not None and piece_sizes is not None
        return _solve_bp_cutting_stock(
            roll_width, piece_sizes, demands, max_iter, max_nodes, gap_tol, eps, on_progress, progress_interval
        )

    assert pricing_fn is not None
    if initial_columns is None:
        raise ValueError("Custom mode requires initial_columns")
    return _solve_bp_custom(
        demands, pricing_fn, initial_columns, max_iter, max_nodes, gap_tol, eps, on_progress, progress_interval
    )


def _solve_bp_cutting_stock(
    roll_width, piece_sizes, demands, max_iter, max_nodes, gap_tol, eps, on_progress, progress_interval
):
    """Branch-and-price for cutting stock."""
    n = check_sequence_lengths((piece_sizes, "piece_sizes"), (demands, "demands"))

    for i, size in enumerate(piece_sizes):
        check_positive(size, name=f"piece_sizes[{i}]")
        if size > roll_width:
            raise ValueError(f"piece_sizes[{i}]={size} exceeds roll_width={roll_width}")

    # Initial patterns: max copies of each piece type per roll
    columns: list[tuple[int, ...]] = []
    for j in range(n):
        if demands[j] > 0:
            count = int(roll_width // piece_sizes[j])
            pattern = tuple(count if i == j else 0 for i in range(n))
            columns.append(pattern)

    column_set: set[tuple[int, ...]] = set(columns)

    def pricing_fn(duals):
        return _knapsack_pricing(piece_sizes, roll_width, duals, eps)

    return _branch_and_price(
        demands,
        columns,
        column_set,
        pricing_fn,
        True,
        max_iter,
        max_nodes,
        gap_tol,
        eps,
        on_progress,
        progress_interval,
    )


def _solve_bp_custom(
    demands, pricing_fn, initial_columns, max_iter, max_nodes, gap_tol, eps, on_progress, progress_interval
):
    """Branch-and-price with custom pricing."""
    m = len(demands)

    for i, col in enumerate(initial_columns):
        if len(col) != m:
            raise ValueError(f"column {i} has wrong length: {len(col)} vs {m}")

    columns: list[tuple[int, ...]] = [tuple(c) for c in initial_columns]
    column_set: set[tuple[int, ...]] = set(columns)

    return _branch_and_price(
        demands,
        columns,
        column_set,
        pricing_fn,
        False,
        max_iter,
        max_nodes,
        gap_tol,
        eps,
        on_progress,
        progress_interval,
    )


def _branch_and_price(
    demands,
    columns,
    column_set,
    pricing_fn,
    is_cutting_stock,
    max_iter,
    max_nodes,
    gap_tol,
    eps,
    on_progress,
    progress_interval,
):
    """Main branch-and-price algorithm."""
    total_cg_iters = 0

    # Solve root node LP via column generation
    x_vals, lp_obj, cg_iters = _solve_node_lp(
        columns, column_set, demands, {}, pricing_fn, is_cutting_stock, max_iter, eps
    )
    total_cg_iters += cg_iters

    if lp_obj == float("inf"):
        return Result(None, float("inf"), 0, total_cg_iters, Status.INFEASIBLE)

    # Check if root LP is already integer
    frac_idx, frac_val = _most_fractional(x_vals, eps)
    if frac_idx is None:
        solution = _build_solution(x_vals, columns, eps)
        return Result(solution, lp_obj, 0, total_cg_iters, Status.OPTIMAL)

    # Initialize B&B
    best_solution: dict[tuple[int, ...], int] | None = None
    best_obj = float("inf")

    # Compute initial upper bound by rounding
    rounded = _round_solution(x_vals, columns, demands, eps)
    if rounded is not None:
        best_solution, best_obj = rounded

    tree: list[tuple[float, int, _BPNode]] = []
    counter = 0
    heappush(tree, (lp_obj, counter, _BPNode(lp_obj, (), 0)))
    counter += 1
    nodes_explored = 0

    while tree and nodes_explored < max_nodes:
        _, _, node = heappop(tree)

        # Prune by bound
        if node.bound >= best_obj - eps:
            continue

        # Convert column_bounds tuple to dict
        col_bounds = {idx: (lo, hi) for idx, lo, hi in node.column_bounds}

        # Solve node LP with column generation
        x_vals, lp_obj, cg_iters = _solve_node_lp(
            columns, column_set, demands, col_bounds, pricing_fn, is_cutting_stock, max_iter, eps
        )
        total_cg_iters += cg_iters
        nodes_explored += 1

        if report_progress(on_progress, progress_interval, nodes_explored, lp_obj, best_obj, total_cg_iters):
            break

        # Prune infeasible or dominated
        if lp_obj == float("inf") or lp_obj >= best_obj - eps:
            continue

        # Check integrality
        frac_idx, frac_val = _most_fractional(x_vals, eps)

        if frac_idx is None:
            # Integer feasible - update incumbent
            obj = sum(x for x in x_vals if x > eps)
            if obj < best_obj - eps:
                best_solution = _build_solution(x_vals, columns, eps)
                best_obj = obj

                # Check gap
                gap = (best_obj - lp_obj) / max(abs(best_obj), 1e-10)
                if gap < gap_tol:
                    return Result(best_solution, best_obj, nodes_explored, total_cg_iters, Status.OPTIMAL)
            continue

        # Branch on most fractional column
        val = frac_val

        # Left child: x[frac_idx] <= floor(val)
        left_bounds = list(node.column_bounds)
        left_bounds.append((frac_idx, 0.0, floor(val)))
        heappush(tree, (lp_obj, counter, _BPNode(lp_obj, tuple(left_bounds), node.depth + 1)))
        counter += 1

        # Right child: x[frac_idx] >= ceil(val)
        right_bounds = list(node.column_bounds)
        right_bounds.append((frac_idx, ceil(val), float("inf")))
        heappush(tree, (lp_obj, counter, _BPNode(lp_obj, tuple(right_bounds), node.depth + 1)))
        counter += 1

    if best_solution is None:
        return Result(None, float("inf"), nodes_explored, total_cg_iters, Status.INFEASIBLE)

    status = Status.OPTIMAL if not tree else Status.FEASIBLE
    return Result(best_solution, best_obj, nodes_explored, total_cg_iters, status)


def _solve_node_lp(columns, column_set, demands, col_bounds, pricing_fn, is_cutting_stock, max_iter, eps):
    """Solve LP relaxation at a B&B node via column generation."""
    cg_iters = 0

    for _ in range(max_iter):
        x_vals, duals, lp_obj = _solve_bounded_master_lp(columns, demands, col_bounds, eps)

        if lp_obj == float("inf"):
            return x_vals, lp_obj, cg_iters

        # Pricing
        new_col, pricing_value = pricing_fn(duals)

        # Check reduced cost
        if is_cutting_stock:
            if pricing_value <= 1.0 + eps:
                break
        else:
            if new_col is None or pricing_value >= -eps:
                break

        if new_col is not None and new_col not in column_set:
            columns.append(new_col)
            column_set.add(new_col)

        cg_iters += 1

    # Final solve
    x_vals, duals, lp_obj = _solve_bounded_master_lp(columns, demands, col_bounds, eps)
    return x_vals, lp_obj, cg_iters


def _solve_bounded_master_lp(columns, demands, col_bounds, eps):
    """Solve master LP with column bounds: min sum(x) s.t. Ax >= b, lo <= x <= hi."""
    m = len(demands)
    n = len(columns)

    if n == 0:
        return [], [0.0] * m, float("inf")

    # Count bound constraints
    n_lower = sum(1 for idx in col_bounds if col_bounds[idx][0] > eps)
    n_upper = sum(1 for idx in col_bounds if col_bounds[idx][1] < float("inf"))

    # Two-phase simplex with bounds
    # Variables: x (n), surplus for >= constraints (m), slack for <= bounds (n_upper),
    #            surplus for >= bounds (n_lower), artificial (m + n_lower)
    n_slack = n_upper
    n_surplus_bounds = n_lower
    n_artificial = m + n_lower
    n_surplus = m

    n_vars = n + n_surplus + n_slack + n_surplus_bounds + n_artificial
    n_rows = m + n_lower + n_upper

    # Build tableau
    tab = [[0.0] * (n_vars + 1) for _ in range(n_rows + 1)]

    # Demand constraints: Ax >= b  =>  Ax - s + a = b
    for i in range(m):
        for j, col in enumerate(columns):
            tab[i][j] = float(col[i])
        tab[i][n + i] = -1.0  # surplus
        tab[i][n + n_surplus + n_slack + n_surplus_bounds + i] = 1.0  # artificial
        tab[i][-1] = float(demands[i])

    # Lower bound constraints: x[idx] >= lo  =>  x[idx] - s + a = lo
    row_idx = m
    art_idx = m
    surplus_idx = 0
    lower_bound_rows = {}
    for idx in sorted(col_bounds.keys()):
        lo, hi = col_bounds[idx]
        if lo > eps:
            tab[row_idx][idx] = 1.0
            tab[row_idx][n + n_surplus + n_slack + surplus_idx] = -1.0  # surplus for bound
            tab[row_idx][n + n_surplus + n_slack + n_surplus_bounds + art_idx] = 1.0  # artificial
            tab[row_idx][-1] = lo
            lower_bound_rows[idx] = row_idx
            row_idx += 1
            art_idx += 1
            surplus_idx += 1

    # Upper bound constraints: x[idx] <= hi  =>  x[idx] + s = hi
    slack_idx = 0
    for idx in sorted(col_bounds.keys()):
        lo, hi = col_bounds[idx]
        if hi < float("inf"):
            tab[row_idx][idx] = 1.0
            tab[row_idx][n + n_surplus + slack_idx] = 1.0  # slack
            tab[row_idx][-1] = hi
            row_idx += 1
            slack_idx += 1

    # Phase 1: minimize artificial variables
    for i in range(n_artificial):
        art_col = n + n_surplus + n_slack + n_surplus_bounds + i
        # Find which row has this artificial
        for r in range(n_rows):
            if abs(tab[r][art_col] - 1.0) < eps:
                for j in range(n_vars + 1):
                    tab[-1][j] -= tab[r][j]
                tab[-1][art_col] = 0.0
                break

    basis = []
    # Artificial variables for demand constraints
    for i in range(m):
        basis.append(n + n_surplus + n_slack + n_surplus_bounds + i)
    # Artificial variables for lower bound constraints
    for i in range(n_lower):
        basis.append(n + n_surplus + n_slack + n_surplus_bounds + m + i)
    # Slack variables for upper bound constraints
    for i in range(n_upper):
        basis.append(n + n_surplus + i)

    _simplex_phase(tab, basis, n + n_surplus + n_slack + n_surplus_bounds, n_rows, eps)

    if tab[-1][-1] < -eps:
        return [0.0] * n, [0.0] * m, float("inf")

    # Phase 2: minimize sum of x
    for j in range(n_vars + 1):
        tab[-1][j] = 0.0
    for j in range(n):
        tab[-1][j] = 1.0

    for i, b in enumerate(basis):
        cost = 1.0 if b < n else 0.0
        if abs(cost) > eps:
            for j in range(n_vars + 1):
                tab[-1][j] -= cost * tab[i][j]

    _simplex_phase(tab, basis, n + n_surplus + n_slack + n_surplus_bounds, n_rows, eps)

    x_vals = [0.0] * n
    for i, b in enumerate(basis):
        if b < n:
            x_vals[b] = max(0.0, tab[i][-1])

    # Duals from demand constraint surplus variables
    duals = [tab[-1][n + i] for i in range(m)]
    objective = -tab[-1][-1]

    return x_vals, duals, objective


def _simplex_phase(tab, basis, n_orig, n_rows, eps):
    """Run simplex iterations on tableau in place."""
    n_cols = len(tab[0])
    basis_set = set(basis)

    for _ in range(100_000):
        # Bland's rule
        enter = -1
        for j in range(n_orig):
            if j not in basis_set and tab[-1][j] < -eps:
                enter = j
                break

        if enter == -1:
            return

        # Minimum ratio test
        leave = -1
        min_ratio = float("inf")
        for i in range(n_rows):
            if tab[i][enter] > eps:
                ratio = tab[i][-1] / tab[i][enter]
                if ratio < min_ratio - eps:
                    min_ratio = ratio
                    leave = i
                elif abs(ratio - min_ratio) <= eps and leave >= 0 and basis[i] < basis[leave]:
                    leave = i

        if leave == -1:
            return

        # Pivot
        piv = tab[leave][enter]
        for j in range(n_cols):
            tab[leave][j] /= piv

        for i in range(n_rows + 1):
            if i != leave:
                factor = tab[i][enter]
                if abs(factor) > eps:
                    for j in range(n_cols):
                        tab[i][j] -= factor * tab[leave][j]

        basis_set.discard(basis[leave])
        basis[leave] = enter
        basis_set.add(enter)


def _most_fractional(x_vals, eps):
    """Find most fractional variable (closest to 0.5)."""
    best_idx, best_frac = None, 0.0
    for i, x in enumerate(x_vals):
        if x > eps:
            frac = abs(x - round(x))
            if frac > eps and frac > best_frac:
                best_idx, best_frac = i, frac
    if best_idx is not None:
        return best_idx, x_vals[best_idx]
    return None, None


def _build_solution(x_vals, columns, eps):
    """Build solution dict from LP values (must be integer)."""
    solution: dict[tuple[int, ...], int] = {}
    for i, x in enumerate(x_vals):
        if x > eps:
            count = int(round(x))
            if count > 0:
                solution[columns[i]] = count
    return solution


def _round_solution(x_vals, columns, demands, eps):
    """Round LP solution to get feasible integer solution."""
    n_cols = len(columns)
    m = len(demands)

    # Round up each fractional value
    rounded = [ceil(x - eps) if x > eps else 0 for x in x_vals]

    # Verify feasibility
    for i in range(m):
        produced = sum(columns[j][i] * rounded[j] for j in range(n_cols))
        if produced < demands[i]:
            return None

    # Build solution
    solution: dict[tuple[int, ...], int] = {}
    total = 0
    for j in range(n_cols):
        if rounded[j] > 0:
            solution[columns[j]] = rounded[j]
            total += rounded[j]

    return solution, float(total)


def _knapsack_pricing(sizes, capacity, values, eps):
    """Solve bounded knapsack for pricing."""
    n = len(sizes)
    if n == 0:
        return (), 0.0

    max_copies = [int(capacity // sizes[i]) if sizes[i] > 0 else 0 for i in range(n)]

    # Scale to integers for DP
    scale = 100
    cap_int = int(capacity * scale + 0.5)
    sizes_int = [max(1, int(sizes[i] * scale + 0.5)) for i in range(n)]

    dp_val = [-float("inf")] * (cap_int + 1)
    dp_pat = [[0] * n for _ in range(cap_int + 1)]
    dp_val[0] = 0.0

    for i in range(n):
        if values[i] <= eps:
            continue

        size_i = sizes_int[i]
        for _ in range(max_copies[i]):
            for w in range(cap_int, size_i - 1, -1):
                prev_w = w - size_i
                if dp_val[prev_w] > -float("inf"):
                    new_val = dp_val[prev_w] + values[i]
                    if new_val > dp_val[w] + eps:
                        dp_val[w] = new_val
                        dp_pat[w] = list(dp_pat[prev_w])
                        dp_pat[w][i] += 1

    best_w = 0
    best_val = 0.0
    for w in range(cap_int + 1):
        if dp_val[w] > best_val + eps:
            best_val = dp_val[w]
            best_w = w

    best_pat = dp_pat[best_w] if best_val > eps else [0] * n

    # Fall back to greedy if scaling caused infeasibility
    total_size = sum(best_pat[i] * sizes[i] for i in range(n))
    if total_size > capacity + eps:
        return _greedy_knapsack(sizes, capacity, values, max_copies)

    return tuple(best_pat), best_val


def _greedy_knapsack(sizes, capacity, values, max_copies):
    """Greedy fallback for knapsack."""
    n = len(sizes)
    indices = sorted(range(n), key=lambda i: values[i] / sizes[i] if sizes[i] > 0 else 0.0, reverse=True)

    pattern = [0] * n
    remaining = capacity
    total_val = 0.0

    for i in indices:
        if values[i] <= 0 or sizes[i] <= 0:
            continue
        copies = min(max_copies[i], int(remaining / sizes[i]))
        if copies > 0:
            pattern[i] = copies
            remaining -= copies * sizes[i]
            total_val += copies * values[i]

    return tuple(pattern), total_val

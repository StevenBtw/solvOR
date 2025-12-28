r"""
0/1 Knapsack solver using dynamic programming.

The problem that launched a thousand algorithms. Select items to maximize value
without exceeding capacity. Each item is all-or-nothing (0/1), no cutting that
gold bar in half.

    from solvor.knapsack import solve_knapsack

    result = solve_knapsack(values, weights, capacity)
    print(result.solution)  # indices of selected items

How it works: DP builds a table asking "what's the best I can do with capacity w?"
for every w from 0 to your limit. Each item either fits (add its value) or doesn't.
Backtrack through the table to find which items were selected.

Use this for:

- Resource allocation with discrete choices
- Budget optimization
- Subset selection problems
- Capital budgeting

Parameters:

    values: value/profit of each item
    weights: weight/cost of each item
    capacity: maximum total weight allowed
    minimize: if True, minimize total value (unusual but supported)

For fractional items, use LP (greedy by value/weight ratio is optimal).
For multiple constraints, use MILP or CP-SAT.
"""

from collections.abc import Sequence

from solvor.types import Result, Status
from solvor.utils import check_non_negative, check_sequence_lengths

__all__ = ["solve_knapsack"]


def solve_knapsack(
    values: Sequence[float],
    weights: Sequence[float],
    capacity: float,
    *,
    minimize: bool = False,
) -> Result:
    """Solve 0/1 knapsack using dynamic programming.

    Returns Result with solution as tuple of selected item indices.
    """
    n = len(values)
    if n == 0:
        return Result((), 0.0, 0, 0, Status.OPTIMAL)

    check_sequence_lengths((values, "values"), (weights, "weights"))
    check_non_negative(capacity, name="capacity")

    # Handle minimize by negating values
    sign = -1 if minimize else 1
    vals = [sign * v for v in values]

    # Convert to integer capacity for DP (scale if needed)
    int_capacity, scale = _to_int_capacity(capacity, weights)

    if int_capacity == 0:
        # No capacity, can't take anything
        return Result((), 0.0, 0, n, Status.OPTIMAL)

    # Scale weights
    int_weights = [max(1, int(w * scale)) if w > 0 else 0 for w in weights]

    # DP table: dp[w] = max value achievable with capacity w
    dp = [0.0] * (int_capacity + 1)

    # Track which items were selected
    # keep[i][w] = True if item i was taken at capacity w
    keep = [[False] * (int_capacity + 1) for _ in range(n)]

    for i in range(n):
        w_i = int_weights[i]
        v_i = vals[i]

        # Traverse backwards to avoid using same item twice
        for w in range(int_capacity, w_i - 1, -1):
            if dp[w - w_i] + v_i > dp[w]:
                dp[w] = dp[w - w_i] + v_i
                keep[i][w] = True

    # Backtrack to find selected items
    selected = []
    w = int_capacity
    for i in range(n - 1, -1, -1):
        if keep[i][w]:
            selected.append(i)
            w -= int_weights[i]

    selected.reverse()
    selected_tuple = tuple(selected)

    # Compute actual objective with original values
    objective = sum(values[i] for i in selected)

    # Verify weight constraint (in case of scaling errors)
    total_weight = sum(weights[i] for i in selected)
    if total_weight > capacity + 1e-9:
        # Scaling caused infeasibility, fall back to greedy
        return _greedy_fallback(values, weights, capacity, minimize)

    return Result(selected_tuple, objective, 0, n, Status.OPTIMAL)


def _to_int_capacity(capacity: float, weights: Sequence[float]) -> tuple[int, float]:
    """Convert capacity to integer, returning (int_capacity, scale_factor)."""
    # Find minimum granularity needed
    all_vals = [capacity] + [w for w in weights if w > 0]

    # Check if already integers
    if all(v == int(v) for v in all_vals):
        return int(capacity), 1.0

    # Find scale to make everything integer-ish
    # Limit to reasonable precision to avoid huge DP tables
    max_capacity = 100000

    if capacity <= 0:
        return 0, 1.0

    scale = min(max_capacity / capacity, 1000.0)
    return int(capacity * scale), scale


def _greedy_fallback(
    values: Sequence[float],
    weights: Sequence[float],
    capacity: float,
    minimize: bool,
) -> Result:
    """Greedy fallback when DP scaling fails."""
    n = len(values)
    if n == 0:
        return Result((), 0.0, 0, 0, Status.FEASIBLE)

    # Sort by value/weight ratio (or weight/value for minimize)
    indices = list(range(n))
    if minimize:
        # For minimize, prefer low value items that fit
        indices.sort(key=lambda i: values[i] / weights[i] if weights[i] > 0 else float("inf"))
    else:
        # For maximize, prefer high value/weight ratio
        indices.sort(key=lambda i: values[i] / weights[i] if weights[i] > 0 else float("inf"), reverse=True)

    selected = []
    remaining = capacity

    for i in indices:
        if weights[i] <= remaining:
            selected.append(i)
            remaining -= weights[i]

    selected.sort()  # Return in original order
    selected_tuple = tuple(selected)
    objective = sum(values[i] for i in selected)
    if minimize:
        objective = -objective

    return Result(selected_tuple, objective, 0, n, Status.FEASIBLE)

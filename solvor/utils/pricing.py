"""
Shared pricing utilities for column generation solvers.

Internal functions used by both cg.py (column generation) and bp.py (branch-and-price).
These implement the knapsack-based pricing subproblem and simplex tableau operations.
"""

from collections.abc import Sequence

__all__ = ["knapsack_pricing", "greedy_knapsack", "simplex_phase"]


def knapsack_pricing(
    sizes: Sequence[float],
    capacity: float,
    values: Sequence[float],
    eps: float,
) -> tuple[tuple[int, ...], float]:
    """Solve bounded knapsack: max sum(v[i]*x[i]) s.t. sum(s[i]*x[i]) <= W.

    Used as the pricing subproblem in cutting stock column generation.
    Returns (pattern, total_value) where pattern is the item counts.
    """
    n = len(sizes)
    if n == 0:
        return (), 0.0

    max_copies = [int(capacity // sizes[i]) if sizes[i] > 0 else 0 for i in range(n)]

    # Scale to integers for DP
    scale = 100
    cap_int = int(capacity * scale + 0.5)
    sizes_int = [max(1, int(sizes[i] * scale + 0.5)) for i in range(n)]

    # dp_val[w] = best value at weight w, dp_pat[w] = pattern achieving it
    dp_val = [-float("inf")] * (cap_int + 1)
    dp_pat = [[0] * n for _ in range(cap_int + 1)]
    dp_val[0] = 0.0

    for i in range(n):
        if values[i] <= eps:
            continue

        size_i = sizes_int[i]
        # Reverse iteration + multiple passes = bounded knapsack
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
        return greedy_knapsack(sizes, capacity, values, max_copies)

    return tuple(best_pat), best_val


def greedy_knapsack(
    sizes: Sequence[float],
    capacity: float,
    values: Sequence[float],
    max_copies: list[int],
) -> tuple[tuple[int, ...], float]:
    """Greedy knapsack fallback: sort by value/size density.

    Used when DP scaling causes infeasibility.
    """
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


def simplex_phase(
    tab: list[list[float]],
    basis: list[int],
    n_orig: int,
    n_rows: int,
    eps: float,
) -> None:
    """Run simplex iterations on tableau in place.

    Implements Bland's rule to prevent cycling.
    Used by master LP solvers in column generation.
    """
    n_cols = len(tab[0])
    basis_set = set(basis)

    for _ in range(100_000):
        # Bland's rule: smallest index with negative reduced cost
        enter = -1
        for j in range(n_orig):
            if j not in basis_set and tab[-1][j] < -eps:
                enter = j
                break

        if enter == -1:
            return

        # Minimum ratio test with Bland's tie-breaking
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

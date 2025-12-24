"""
Hungarian Algorithm for optimal assignment.

Got workers and tasks? This finds who does what at minimum total cost. O(n³),
the go-to for pure assignment problems.

    from solvor import solve_hungarian

    result = solve_hungarian(cost_matrix)
    result = solve_hungarian(cost_matrix, minimize=False)  # maximize

             Task A  Task B  Task C
    Worker 0   10      5      13        solve_hungarian finds: 0→B, 1→A, 2→C
    Worker 1    3      9      18        total cost: 5 + 3 + 12 = 20
    Worker 2   10      6      12

Returns assignment[i] = column assigned to row i. Rectangular matrices work
fine, assigns min(rows, cols) pairs.

Don't use this for: matching within a single group (roommates, tennis doubles),
or when one worker handles multiple tasks / one task needs multiple workers
(use min_cost_flow).
"""

from collections.abc import Sequence
from math import inf

from solvor.types import Result

__all__ = ["solve_hungarian"]


def solve_hungarian(
    cost_matrix: Sequence[Sequence[float]],
    *,
    minimize: bool = True,
) -> Result:
    if not cost_matrix or not cost_matrix[0]:
        return Result([], 0.0, 0, 0)

    n_rows = len(cost_matrix)
    n_cols = len(cost_matrix[0])
    n = max(n_rows, n_cols)

    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n_rows):
        for j in range(n_cols):
            matrix[i][j] = cost_matrix[i][j]

    if not minimize:
        max_val = max(cost_matrix[i][j] for i in range(n_rows) for j in range(n_cols))
        for i in range(n):
            for j in range(n):
                if i < n_rows and j < n_cols:
                    matrix[i][j] = max_val - cost_matrix[i][j]
                else:
                    matrix[i][j] = 0.0

    row_potential = [0.0] * (n + 1)
    col_potential = [0.0] * (n + 1)
    col_match = [0] * (n + 1)
    augment_path = [0] * (n + 1)

    iterations = 0

    for i in range(1, n + 1):
        col_match[0] = i
        current_col = 0
        min_slack = [inf] * (n + 1)
        used = [False] * (n + 1)

        while col_match[current_col] != 0:
            iterations += 1
            used[current_col] = True
            matched_row = col_match[current_col]
            delta = inf
            next_col = 0

            for j in range(1, n + 1):
                if not used[j]:
                    reduced_cost = matrix[matched_row - 1][j - 1] - row_potential[matched_row] - col_potential[j]
                    if reduced_cost < min_slack[j]:
                        min_slack[j] = reduced_cost
                        augment_path[j] = current_col
                    if min_slack[j] < delta:
                        delta = min_slack[j]
                        next_col = j

            for j in range(n + 1):
                if used[j]:
                    row_potential[col_match[j]] += delta
                    col_potential[j] -= delta
                else:
                    min_slack[j] -= delta

            current_col = next_col

        while current_col != 0:
            prev_col = augment_path[current_col]
            col_match[current_col] = col_match[prev_col]
            current_col = prev_col

    assignment = [-1] * n_rows
    for j in range(1, n + 1):
        if col_match[j] != 0 and col_match[j] <= n_rows and j <= n_cols:
            assignment[col_match[j] - 1] = j - 1

    total_cost = 0.0
    for i in range(n_rows):
        if assignment[i] != -1 and assignment[i] < n_cols:
            total_cost += cost_matrix[i][assignment[i]]

    return Result(assignment, total_cost, iterations, n * n)

"""
Hungarian Algorithm for optimal assignment.

Got workers and tasks? This finds who does what at minimum total cost. O(n³),
the go-to for pure assignment problems.

    from solvor.hungarian import hungarian

    result = hungarian(cost_matrix)
    result = hungarian(cost_matrix, minimize=False)  # maximize

             Task A  Task B  Task C
    Worker 0   10      5      13        hungarian finds: 0→B, 1→A, 2→C
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

__all__ = ["hungarian"]

def hungarian(
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

    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)

    iterations = 0

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (n + 1)
        used = [False] * (n + 1)

        while p[j0] != 0:
            iterations += 1
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = 0

            for j in range(1, n + 1):
                if not used[j]:
                    cur = matrix[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1

        while j0 != 0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    assignment = [-1] * n_rows
    for j in range(1, n + 1):
        if p[j] != 0 and p[j] <= n_rows and j <= n_cols:
            assignment[p[j] - 1] = j - 1

    total_cost = 0.0
    for i in range(n_rows):
        if assignment[i] != -1 and assignment[i] < n_cols:
            total_cost += cost_matrix[i][assignment[i]]

    return Result(assignment, total_cost, iterations, n * n)
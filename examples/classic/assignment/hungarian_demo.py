"""
Assignment Problem: Workers to Tasks

Assign n workers to n tasks to minimize total cost, where each worker
does exactly one task and each task is done by exactly one worker.

Formulation:
    Given: n x n cost matrix cost[i][j] = cost of worker i doing task j
    Find: permutation p where worker i does task p[i]
    Minimize: sum(cost[i][p[i]] for i in range(n))

Why this solver:
    The Hungarian algorithm solves assignment in O(n^3), much faster than
    brute-force O(n!). Works for any cost matrix, handles ties gracefully.

Expected result:
    Worker 0 -> Task 2, Worker 1 -> Task 1, Worker 2 -> Task 0, Worker 3 -> Task 3
    Total cost: 140

Reference:
    Kuhn, H.W. (1955) "The Hungarian Method for the Assignment Problem"
"""

from solvor import solve_hungarian

# Cost matrix: cost[worker][task]
# Worker 0 costs 82 for task 0, 83 for task 1, etc.
cost = [
    [82, 83, 69, 92],
    [77, 37, 49, 92],
    [11, 69, 5, 86],
    [8, 9, 98, 23],
]

result = solve_hungarian(cost)

print("Optimal assignment:")
for worker, task in enumerate(result.solution):
    print(f"  Worker {worker} -> Task {task} (cost: {cost[worker][task]})")
print(f"Total cost: {result.objective}")

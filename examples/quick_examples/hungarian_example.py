"""
Hungarian Algorithm Example

Find optimal assignment of workers to tasks.
"""

from solvor import solve_hungarian

# Cost matrix: cost[i][j] = cost of assigning worker i to task j
cost = [
    [10, 5, 13, 4],
    [3, 9, 18, 13],
    [10, 6, 12, 8],
    [8, 8, 9, 11],
]

result = solve_hungarian(cost)
print(f"Assignment: {result.solution}")  # worker i -> task assignment[i]
print(f"Total cost: {result.objective}")

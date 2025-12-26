"""
Assignment Problem Example

Assign workers to tasks minimizing total cost (via flow reduction).
"""

from solvor import solve_assignment

# Cost matrix: cost[worker][task]
costs = [
    [10, 5, 13, 4],
    [3, 9, 18, 7],
    [10, 6, 12, 5],
    [8, 7, 9, 11],
]

result = solve_assignment(costs)
print(f"Assignment: {result.solution}")
print(f"Total cost: {result.objective}")

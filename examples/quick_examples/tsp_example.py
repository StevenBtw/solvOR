"""
TSP Example

Traveling Salesman Problem using built-in solver.
"""

from solvor import solve_tsp

# Distance matrix for 5 cities
distances = [
    [0, 10, 15, 20, 25],
    [10, 0, 35, 25, 30],
    [15, 35, 0, 30, 20],
    [20, 25, 30, 0, 15],
    [25, 30, 20, 15, 0],
]

result = solve_tsp(distances, max_iter=500)
print(f"Best tour: {result.solution}")
print(f"Tour length: {result.objective}")

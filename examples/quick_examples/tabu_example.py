"""
Tabu Search Example: Small TSP

Find shortest tour visiting 5 cities and returning to start.
"""

from solvor import tabu_search

# Distance matrix for 5 cities
distances = [
    [0, 10, 15, 20, 25],
    [10, 0, 35, 25, 30],
    [15, 35, 0, 30, 20],
    [20, 25, 30, 0, 15],
    [25, 30, 20, 15, 0],
]


def tour_cost(perm):
    return sum(distances[perm[i]][perm[(i + 1) % len(perm)]] for i in range(len(perm)))


def neighbors(perm):
    # Generate all 2-opt swaps with move descriptors
    for i in range(len(perm)):
        for j in range(i + 2, len(perm)):
            new_perm = perm[:i] + perm[i : j + 1][::-1] + perm[j + 1 :]
            yield (i, j), new_perm  # (move, solution) tuple


x0 = [0, 1, 2, 3, 4]
result = tabu_search(x0, tour_cost, neighbors, cooldown=10, max_iter=100)
print(f"Best tour: {result.solution}")
print(f"Tour length: {result.objective}")

"""
Large Neighborhood Search Example

Optimize a permutation using destroy-repair operations.
"""

from random import Random

from solvor import lns

# Distance matrix for TSP
distances = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0],
]


def tour_cost(perm):
    return sum(distances[perm[i]][perm[(i + 1) % len(perm)]] for i in range(len(perm)))


def destroy(perm, rng: Random):
    """Remove 2 random cities (keep first city fixed as depot)."""
    perm = list(perm)
    to_remove = rng.sample(range(1, len(perm)), k=min(2, len(perm) - 1))
    return [c for i, c in enumerate(perm) if i not in to_remove]


def repair(partial, rng: Random):
    """Insert missing cities at best positions."""
    perm = list(partial)
    missing = [c for c in range(4) if c not in perm]
    for city in missing:
        best_pos, best_cost = 0, float("inf")
        for i in range(len(perm) + 1):
            candidate = perm[:i] + [city] + perm[i:]
            cost = tour_cost(candidate)
            if cost < best_cost:
                best_pos, best_cost = i, cost
        perm.insert(best_pos, city)
    return perm


initial = [0, 1, 2, 3]
result = lns(initial, tour_cost, destroy, repair, max_iter=100)
print(f"Best tour: {result.solution}")
print(f"Tour length: {result.objective}")

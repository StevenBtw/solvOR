"""
Traveling Salesman Problem with Tabu Search

Solve TSP using tabu search with 2-opt neighborhood.

Formulation:
    Given: n cities with pairwise distances
    Find: permutation (tour) minimizing total distance
    Neighborhood: 2-opt moves (reverse a segment of the tour)

Why this solver:
    Tabu search is deterministic and systematic. It forbids recently visited
    solutions (via "tabu list"), forcing exploration beyond local optima.
    More predictable than simulated annealing, often finds good solutions faster.

Expected result:
    Tour length around 280-320 for this 10-city random instance (seed=42).
    Tabu search typically converges quickly due to its greedy + memory approach.

Reference:
    Glover, F. (1986) "Future Paths for Integer Programming and Links to AI"
"""

import random

from solvor import tabu_search

# Generate random city coordinates
random.seed(42)
N = 10
cities = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(N)]


def distance(i, j):
    dx = cities[i][0] - cities[j][0]
    dy = cities[i][1] - cities[j][1]
    return (dx**2 + dy**2) ** 0.5


def tour_length(tour):
    return sum(distance(tour[i], tour[(i + 1) % N]) for i in range(N))


def two_opt_neighbors(tour):
    """Generate all 2-opt neighbors with move descriptors."""
    for i in range(N):
        for j in range(i + 2, N):
            if i == 0 and j == N - 1:
                continue  # Skip: same as (1, N-1)
            new_tour = tour[:i] + tour[i : j + 1][::-1] + tour[j + 1 :]
            yield (i, j), new_tour  # (move, solution) tuple


# Initial tour
initial = list(range(N))

result = tabu_search(
    initial,
    tour_length,
    two_opt_neighbors,
    cooldown=20,
    max_iter=500,
)

print(f"Best tour: {result.solution}")
print(f"Tour length: {result.objective:.2f}")
print(f"Iterations: {result.iterations}")

"""
Traveling Salesman Problem with Simulated Annealing

Solve TSP using simulated annealing with 2-opt moves.

Formulation:
    Given: n cities with pairwise distances
    Find: permutation (tour) minimizing total distance
    Neighborhood: 2-opt moves (reverse a segment of the tour)

Why this solver:
    Simulated annealing accepts worse solutions probabilistically, controlled
    by "temperature". High temp = more exploration, low temp = greedy refinement.
    Simple to implement, no memory overhead, good for rough optimization.

Expected result:
    Tour length around 280-320 for this 10-city random instance (seed=42).
    Results vary with temperature schedule; this uses geometric cooling.

Reference:
    Kirkpatrick, S. et al. (1983) "Optimization by Simulated Annealing"
"""

import random

from solvor import anneal

# Same 10-city instance as tsp_tabu.py
random.seed(42)
N = 10
cities = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(N)]


def distance(i, j):
    dx = cities[i][0] - cities[j][0]
    dy = cities[i][1] - cities[j][1]
    return (dx**2 + dy**2) ** 0.5


def tour_length(tour):
    return sum(distance(tour[i], tour[(i + 1) % N]) for i in range(N))


def neighbor(tour):
    """Random 2-opt move."""
    i, j = sorted(random.sample(range(N), 2))
    if i == 0 and j == N - 1:
        j -= 1
    return tour[:i] + tour[i : j + 1][::-1] + tour[j + 1 :]


initial = list(range(N))

result = anneal(
    initial,
    tour_length,
    neighbor,
    temperature=100.0,
    cooling=0.999,
    max_iter=5000,
    seed=42,
)

print(f"Best tour: {result.solution}")
print(f"Tour length: {result.objective:.2f}")
print(f"Iterations: {result.iterations}")

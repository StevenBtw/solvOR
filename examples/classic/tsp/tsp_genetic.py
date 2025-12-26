"""
Traveling Salesman Problem with Genetic Algorithm

Solve TSP using a genetic algorithm with order crossover (OX) and swap mutation.
Genetic algorithms are particularly good at exploration and avoiding local optima.

Formulation:
    Given: n cities with pairwise distances
    Find: permutation (tour) minimizing total distance
    Solution representation: list of city indices [0, 1, 2, ..., n-1]

Why this solver:
    Genetic algorithms maintain population diversity, making them less likely
    to get stuck than single-solution methods. Order crossover (OX) preserves
    valid tours while combining parent solutions effectively.

Crossover: Order Crossover (OX)
    1. Select random segment from parent1
    2. Copy segment to child
    3. Fill remaining positions with cities from parent2 (in order, skipping duplicates)

Mutation: Swap Mutation
    Randomly swap two cities in the tour.

Expected result:
    Tour length around 280-320 for this 10-city random instance (seed=42).
    Genetic algorithm explores more broadly than tabu/anneal, may find different local optima.

Reference:
    Goldberg, D.E. & Lingle, R. (1985) "Alleles, Loci, and the Traveling Salesman Problem"
    https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)#Order_crossover_(OX1)
"""

import random

from solvor import evolve

# Same 10-city instance as tsp_tabu.py and tsp_anneal.py
random.seed(42)
N = 10
cities = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(N)]


def distance(i, j):
    dx = cities[i][0] - cities[j][0]
    dy = cities[i][1] - cities[j][1]
    return (dx**2 + dy**2) ** 0.5


def tour_length(tour):
    return sum(distance(tour[i], tour[(i + 1) % N]) for i in range(N))


def order_crossover(parent1, parent2):
    """Order crossover (OX): preserves relative order from parents."""
    size = len(parent1)

    # Select random segment from parent1
    start, end = sorted(random.sample(range(size), 2))

    # Create child with segment from parent1
    child = [None] * size
    child[start : end + 1] = parent1[start : end + 1]
    segment = set(child[start : end + 1])

    # Fill remaining positions with cities from parent2 (in order)
    pos = (end + 1) % size
    for city in parent2:
        if city not in segment:
            while child[pos] is not None:
                pos = (pos + 1) % size
            child[pos] = city
            pos = (pos + 1) % size

    return child


def swap_mutate(tour):
    """Swap mutation: exchange two random cities."""
    new_tour = tour.copy()
    i, j = random.sample(range(N), 2)
    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour


def main():
    print("TSP - Genetic Algorithm")
    print("=" * 40)
    print(f"Cities: {N}")
    print("Coordinates: seed=42 random in [0,100]x[0,100]")
    print()

    # Create initial population (random permutations)
    random.seed(123)  # Different seed for population
    pop_size = 50
    population = [random.sample(range(N), N) for _ in range(pop_size)]

    result = evolve(
        tour_length,
        population,
        order_crossover,
        swap_mutate,
        max_gen=200,
        mutation_rate=0.2,
        elite_size=5,
        tournament_k=3,
        seed=42,
    )

    print("Results:")
    print("-" * 40)
    print(f"Best tour: {result.solution}")
    print(f"Tour length: {result.objective:.2f}")
    print(f"Generations: {result.iterations}")
    print(f"Evaluations: {result.evaluations}")


if __name__ == "__main__":
    main()

"""
Genetic Algorithm Example: OneMax

Maximize the number of 1s in a binary string.
Optimal solution: all 1s.
"""

import random

from solvor import evolve

N = 20  # String length


def fitness(individual):
    return sum(individual)  # Count 1s


def crossover(p1, p2):
    # Single-point crossover
    pt = random.randint(1, len(p1) - 1)
    return p1[:pt] + p2[pt:]


def mutate(ind):
    # Flip one random bit
    i = random.randint(0, len(ind) - 1)
    new_ind = list(ind)
    new_ind[i] = 1 - new_ind[i]
    return tuple(new_ind)


# Initial population
population = [tuple(random.randint(0, 1) for _ in range(N)) for _ in range(50)]

result = evolve(fitness, population, crossover, mutate, max_gen=100, minimize=False)
print(f"Best: {''.join(map(str, result.solution))}")
print(f"Fitness: {result.objective}")

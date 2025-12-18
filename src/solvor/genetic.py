"""
Genetic Algorithm - Population-based Evolutionary Optimization

Evolves a population of solutions through selection, crossover, and mutation.
Good for combinatorial optimization where the solution space is discrete
and the fitness landscape may have many local optima.

Usage:
    from solvor.genetic import evolve, Status
    result = evolve(objective_fn, population, crossover, mutate)
    result = evolve(f, pop, cross, mut, minimize=False)  # maximize

Parameters:
    objective_fn  : solution -> float (fitness function)
    population    : Initial population (list of solutions)
    crossover     : (parent1, parent2) -> child (recombination operator)
    mutate        : solution -> solution (mutation operator)
    minimize      : True for min, False for max (default: True)
    elite_size    : Number of best solutions to preserve (default: 2)
    mutation_rate : Probability of mutation per child (default: 0.1)
    max_gen       : Maximum generations (default: 100)
    tournament_k  : Tournament selection size (default: 3)
    seed          : Random seed (default: None)

Returns Result(solution, objective, iterations, evaluations, status)
    solution = best solution found
    iterations = generations completed
    evaluations = total fitness evaluations

Example (TSP):
    def crossover(p1, p2):  # Order crossover
        ...
    def mutate(tour):  # 2-opt swap
        i, j = random pair
        return tour[:i] + tour[i:j][::-1] + tour[j:]
    result = evolve(tour_length, initial_pop, crossover, mutate)
"""

from collections import namedtuple
from collections.abc import Callable, Sequence
from operator import attrgetter
from random import Random
from solvor.types import Status, Result

__all__ = ["evolve", "Status", "Result"]

Individual = namedtuple('Individual', ['solution', 'fitness'])

def evolve[T](
    objective_fn: Callable[[T], float],
    population: Sequence[T],
    crossover: Callable[[T, T], T],
    mutate: Callable[[T], T],
    *,
    minimize: bool = True,
    elite_size: int = 2,
    mutation_rate: float = 0.1,
    max_gen: int = 100,
    tournament_k: int = 3,
    seed: int | None = None,
) -> Result:
    """(objective_fn, population, crossover, mutate, opts) -> Result with best solution."""
    rng = Random(seed)
    sign = 1 if minimize else -1
    pop_size = len(population)
    evals = 0

    def evaluate(sol):
        nonlocal evals
        evals += 1
        return sign * objective_fn(sol)

    pop = [Individual(sol, evaluate(sol)) for sol in population]
    pop.sort(key=attrgetter('fitness'))
    best = pop[0]

    def tournament():
        contestants = rng.sample(pop, min(tournament_k, len(pop)))
        return min(contestants, key=attrgetter('fitness'))

    for gen in range(max_gen):
        new_pop = pop[:elite_size]

        while len(new_pop) < pop_size:
            p1 = tournament()
            p2 = tournament()
            child_sol = crossover(p1.solution, p2.solution)

            if rng.random() < mutation_rate:
                child_sol = mutate(child_sol)

            child = Individual(child_sol, evaluate(child_sol))
            new_pop.append(child)

        pop = sorted(new_pop, key=attrgetter('fitness'))[:pop_size]

        if pop[0].fitness < best.fitness:
            best = pop[0]

    final_obj = best.fitness * sign
    return Result(best.solution, final_obj, max_gen, evals, Status.FEASIBLE)

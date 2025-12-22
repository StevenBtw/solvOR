"""
Genetic Algorithm, population-based search that excels at multi-objective problems.

Slower than modern solvers for single-objective problems, and gradients will beat
it on continuous optimization. But genetic algorithms are king at multi-objective
optimization where objectives compete (Pareto fronts). Great at exploration when
you're searching in the dark without structure to exploit. And they parallelize
beautifully while most solvers don't.

Unlike anneal/tabu (single solution), this evolves a whole population. More
overhead, but better diversity and less likely to get trapped.

    from solvor.genetic import evolve

    result = evolve(objective_fn, population, crossover_fn, mutate_fn)
    result = evolve(objective_fn, pop, cross, mut, minimize=False)  # maximize

Parameters:
    objective_fn  : solution -> float, your fitness function
    population    : list of starting solutions, bigger = more diversity but slower
    crossover     : (parent1, parent2) -> child, how solutions combine - this matters
                    a lot, bad crossover = expensive random search
    mutate        : solution -> solution, small random changes - keep it subtle
    elite_size    : survivors per generation, too high = stagnation (default: 2)
    mutation_rate : how often to mutate, too low = premature convergence (default: 0.1)
    max_gen       : generations to run (default: 100)
    tournament_k  : selection pressure, higher = greedier (default: 3)

Don't use this for: problems with gradient info (use gradient descent), convex
problems (use simplex), or discrete structured problems (use CP/SAT).
"""

from collections import namedtuple
from collections.abc import Callable, Sequence
from operator import attrgetter
from random import Random

from solvor.types import Progress, ProgressCallback, Result, Status

__all__ = ["evolve"]

Individual = namedtuple("Individual", ["solution", "fitness"])


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
    on_progress: ProgressCallback | None = None,
    progress_interval: int = 0,
) -> Result:
    rng = Random(seed)
    sign = 1 if minimize else -1
    pop_size = len(population)
    evals = 0

    def evaluate(sol):
        nonlocal evals
        evals += 1
        return sign * objective_fn(sol)

    pop = [Individual(sol, evaluate(sol)) for sol in population]
    pop.sort(key=attrgetter("fitness"))
    best = pop[0]

    def tournament():
        contestants = rng.sample(pop, min(tournament_k, len(pop)))
        return min(contestants, key=attrgetter("fitness"))

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

        pop = sorted(new_pop, key=attrgetter("fitness"))[:pop_size]

        if pop[0].fitness < best.fitness:
            best = pop[0]

        if on_progress and progress_interval > 0 and (gen + 1) % progress_interval == 0:
            current_obj = pop[0].fitness * sign
            best_so_far = best.fitness * sign
            progress = Progress(gen + 1, current_obj, best_so_far if best_so_far != current_obj else None, evals)
            if on_progress(progress) is True:
                return Result(best.solution, best_so_far, gen + 1, evals, Status.FEASIBLE)

    final_obj = best.fitness * sign
    return Result(best.solution, final_obj, max_gen, evals, Status.FEASIBLE)

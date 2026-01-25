r"""
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

How it works: each generation, select parents via tournament, combine with
crossover to produce children, apply random mutations. Elite individuals
survive unchanged. Adaptive mutation rate increases when stagnating.

Use this for:

- Multi-objective optimization (Pareto fronts)
- Exploration when problem structure is unknown
- Problems that parallelize well
- When you can define good crossover/mutation operators

Parameters:

    objective_fn: fitness function mapping solution to score
    population: starting solutions (bigger = more diversity but slower)
    crossover: combine two parents into one child
    mutate: small random changes to a solution
    elite_size: survivors per generation (default: 2)
    mutation_rate: how often to mutate (default: 0.1)
    tournament_k: selection pressure (default: 3)

Don't use this for: problems with gradient info (use gradient descent), convex
problems (use simplex), or discrete structured problems (use CP/SAT).
"""

from collections import namedtuple
from collections.abc import Callable, Sequence
from operator import attrgetter
from random import Random

from solvor.types import ProgressCallback, Result, Status
from solvor.utils.helpers import Evaluator, report_progress

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
    adaptive_mutation: bool = False,
    max_iter: int = 100,
    tournament_k: int = 3,
    seed: int | None = None,
    on_progress: ProgressCallback | None = None,
    progress_interval: int = 0,
) -> Result:
    """
    Genetic algorithm with optional adaptive mutation.

    Args:
        adaptive_mutation: If True, increase mutation rate when population
            diversity is low (stagnation), decrease when improving.
    """
    rng = Random(seed)
    evaluate = Evaluator(objective_fn, minimize)
    pop_size = len(population)

    pop = [Individual(sol, evaluate(sol)) for sol in population]
    pop.sort(key=attrgetter("fitness"))
    best_solution = pop[0].solution
    best_fitness = pop[0].fitness

    def tournament():
        contestants = rng.sample(pop, min(tournament_k, len(pop)))
        return min(contestants, key=attrgetter("fitness"))

    current_mutation_rate = mutation_rate
    stagnation_count = 0

    for iteration in range(max_iter):
        new_pop = pop[:elite_size]

        while len(new_pop) < pop_size:
            p1 = tournament()
            p2 = tournament()
            child_sol = crossover(p1.solution, p2.solution)

            if rng.random() < current_mutation_rate:
                child_sol = mutate(child_sol)

            child = Individual(child_sol, evaluate(child_sol))
            new_pop.append(child)

        pop = sorted(new_pop, key=attrgetter("fitness"))[:pop_size]

        improved = False
        if pop[0].fitness < best_fitness:
            best_solution = pop[0].solution
            best_fitness = pop[0].fitness
            improved = True

        # Adaptive mutation: adjust rate based on progress
        if adaptive_mutation:
            if improved:
                stagnation_count = 0
                # Decrease mutation when improving
                current_mutation_rate = max(0.01, current_mutation_rate * 0.95)
            else:
                stagnation_count += 1
                if stagnation_count >= 5:
                    # Increase mutation when stagnating
                    current_mutation_rate = min(0.5, current_mutation_rate * 1.2)

        if report_progress(
            on_progress,
            progress_interval,
            iteration + 1,
            evaluate.to_user(pop[0].fitness),
            evaluate.to_user(best_fitness),
            evaluate.evals,
        ):
            return Result(best_solution, evaluate.to_user(best_fitness), iteration + 1, evaluate.evals, Status.FEASIBLE)

    final_obj = evaluate.to_user(best_fitness)
    return Result(best_solution, final_obj, max_iter, evaluate.evals, Status.FEASIBLE)

"""
Simulated Annealing, a black box optimization that handles local optima well.

Use this when you have an objective function and can generate neighbors, but don't
know much else about the problem structure. Fast to prototype, low constraints on
problem formulation. Works well on landscapes with many local optima where gradient
descent would get stuck.

Don't use this for: problems where you need guarantees, or when you have constraints
that are easier to encode in MILP/CP. If you need more control over the search,
consider tabu (memory of visited states) or genetic (population-based exploration).

    from solvor.anneal import anneal, linear_cooling

    result = anneal(start, objective_fn, neighbor_fn)
    result = anneal(start, objective_fn, neighbor_fn, minimize=False)  # maximize
    result = anneal(start, objective_fn, neighbor_fn, cooling=linear_cooling())

The neighbor function is the key part, good neighbors make small moves, not random
jumps. Think "swap two cities" for TSP, not "shuffle everything". Small perturbations
let the algorithm actually explore the neighborhood instead of teleporting randomly.

If it's getting stuck: try higher starting temperature or slower cooling. If it's
taking forever: cool faster.

Cooling schedules:
    exponential_cooling(rate)  : temp = initial * rate^iter (default, classic)
    linear_cooling(min_temp)   : temp decreases linearly to min_temp
    logarithmic_cooling(c)     : temp = initial / (1 + c * log(1 + iter)), slow cooling

Or pass any callable: (initial_temp, iteration, max_iter) -> temperature
"""

from collections.abc import Callable
from math import exp, log
from random import Random

from solvor.types import Progress, ProgressCallback, Result, Status

__all__ = ["anneal", "exponential_cooling", "linear_cooling", "logarithmic_cooling"]

CoolingSchedule = Callable[[float, int, int], float]


def exponential_cooling(rate: float = 0.9995) -> CoolingSchedule:
    """Geometric cooling: temp = initial * rate^iteration."""

    def schedule(initial_temp: float, iteration: int, max_iter: int) -> float:
        return initial_temp * (rate**iteration)

    return schedule


def linear_cooling(min_temp: float = 1e-8) -> CoolingSchedule:
    """Linear cooling: temp decreases linearly from initial to min_temp."""

    def schedule(initial_temp: float, iteration: int, max_iter: int) -> float:
        return initial_temp - (initial_temp - min_temp) * iteration / max_iter

    return schedule


def logarithmic_cooling(c: float = 1.0) -> CoolingSchedule:
    """Logarithmic cooling: temp = initial / (1 + c * log(1 + iteration))."""

    def schedule(initial_temp: float, iteration: int, max_iter: int) -> float:
        return initial_temp / (1 + c * log(1 + iteration))

    return schedule


def anneal[T](
    initial: T,
    objective_fn: Callable[[T], float],
    neighbors: Callable[[T], T],
    *,
    minimize: bool = True,
    temperature: float = 1000.0,
    cooling: float | CoolingSchedule = 0.9995,
    min_temp: float = 1e-8,
    max_iter: int = 100_000,
    seed: int | None = None,
    on_progress: ProgressCallback | None = None,
    progress_interval: int = 0,
) -> Result:
    rng = Random(seed)
    sign = 1 if minimize else -1
    evals = 0

    if callable(cooling):
        schedule = cooling
    else:
        schedule = exponential_cooling(cooling)

    def evaluate(sol):
        nonlocal evals
        evals += 1
        return sign * objective_fn(sol)

    solution, obj = initial, evaluate(initial)
    best_solution, best_obj = solution, obj
    initial_temp = temperature

    for iteration in range(1, max_iter + 1):
        temperature = schedule(initial_temp, iteration, max_iter)

        if temperature < min_temp:
            break

        neighbor = neighbors(solution)
        neighbor_obj = evaluate(neighbor)
        delta = neighbor_obj - obj

        if delta < 0 or rng.random() < exp(-delta / temperature):
            solution, obj = neighbor, neighbor_obj

            if obj < best_obj:
                best_solution, best_obj = solution, obj

        if on_progress and progress_interval > 0 and iteration % progress_interval == 0:
            current_obj = obj * sign
            best_so_far = best_obj * sign
            progress = Progress(iteration, current_obj, best_so_far if best_so_far != current_obj else None, evals)
            if on_progress(progress) is True:
                return Result(best_solution, best_so_far, iteration, evals, Status.FEASIBLE)

    final_obj = best_obj * sign
    status = Status.MAX_ITER if iteration == max_iter else Status.FEASIBLE
    return Result(best_solution, final_obj, iteration, evals, status)

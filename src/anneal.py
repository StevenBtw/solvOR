"""
Simulated Annealing - Probabilistic Metaheuristic for Global Optimization

Escapes local optima by accepting worse solutions with probability exp(-delta/T).
Temperature T decreases over time following a cooling schedule, gradually shifting
from exploration (high T) to exploitation (low T).

Usage:
    from src.sa import anneal, Status
    result = anneal(initial, objective_fn, neighbors)
    result = anneal(initial, objective_fn, neighbors, minimize=False)

Parameters:
    initial       : Starting solution
    objective_fn  : solution -> float (value to minimize)
    neighbors     : solution -> new_solution (random neighbor generator)
    minimize      : True for min, False for max (default: True)
    temperature   : Starting temperature (default: 1000.0)
    cooling       : Multiplicative cooling factor per iteration (default: 0.9995)
    min_temp      : Stop when temperature falls below this (default: 1e-8)
    max_iter      : Maximum iterations (default: 100,000)

Returns Result(solution, objective, iterations, evaluations, status)

Parameter impact:
    temperature too low  -> trapped in local_optimum early
    temperature too high -> random walk, slow convergence
    cooling too fast     -> insufficient exploration
    cooling too slow     -> wastes iterations at high temperature
"""

from collections import namedtuple
from enum import IntEnum, auto
from math import exp
from random import random

__all__ = ["anneal", "Status", "Result"]

class Status(IntEnum):
    OPTIMAL = auto()
    FEASIBLE = auto()
    INFEASIBLE = auto()
    UNBOUNDED = auto()
    MAX_ITER = auto()

Result = namedtuple('Result', ['solution', 'objective', 'iterations', 'evaluations', 'status'])

def anneal(initial, objective_fn, neighbors, minimize=True, temperature=1000.0, cooling=0.9995, min_temp=1e-8, max_iter=100_000):
    """(initial, objective_fn, neighbors, opts) -> Result with best_solution found."""
    sign = 1 if minimize else -1
    evals = 0

    def evaluate(sol):
        nonlocal evals
        evals += 1
        return sign * objective_fn(sol)

    solution, obj = initial, evaluate(initial)
    best_solution, best_obj = solution, obj

    for iteration in range(1, max_iter + 1):
        if temperature < min_temp:
            break

        neighbor = neighbors(solution)
        neighbor_obj = evaluate(neighbor)
        delta = neighbor_obj - obj

        if delta < 0 or random() < exp(-delta / temperature):
            solution, obj = neighbor, neighbor_obj

            if obj < best_obj:
                best_solution, best_obj = solution, obj

        temperature *= cooling

    final_obj = best_obj * sign
    status = Status.MAX_ITER if iteration == max_iter else Status.FEASIBLE
    return Result(best_solution, final_obj, iteration, evals, status)

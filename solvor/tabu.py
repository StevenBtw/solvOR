"""
Tabu Search - Metaheuristic for Combinatorial Optimization

Iteratively moves to the best neighbor while maintaining a tabu list of
recently visited moves to escape local optima. Uses aspiration criteria
to override tabu status when a move improves the best_solution.

Usage:
    from solvor.tabu import tabu_search, solve_tsp, Status
    result = tabu_search(initial, objective_fn, neighbors)
    result = solve_tsp(distance_matrix)

Parameters:
    initial       : Starting solution
    objective_fn  : solution -> float (value to minimize)
    neighbors     : solution -> [(move, new_solution), ...] where move is hashable
    minimize      : True for min, False for max (default: True)
    cooldown      : How many iterations a move stays forbidden (default: 10)
    max_iter      : Maximum iterations (default: 1000)
    max_no_improve: Stop if no improvement for this many iterations (default: 100)

Returns Result(solution, objective, iterations, evaluations, status)

Parameter impact:
    cooldown too low  -> may cycle between solutions
    cooldown too high -> over-constrained, slow exploration
    max_no_improve    -> controls diversification vs early termination
"""

from collections import deque
from collections.abc import Callable, Sequence
from itertools import pairwise
from solvor.types import Status, Result

__all__ = ["tabu_search", "solve_tsp", "Status", "Result"]

def tabu_search[T, M](
    initial: T,
    objective_fn: Callable[[T], float],
    neighbors: Callable[[T], Sequence[tuple[M, T]]],
    *,
    minimize: bool = True,
    cooldown: int = 10,
    max_iter: int = 1000,
    max_no_improve: int = 100,
) -> Result:
    """(initial, objective_fn, neighbors, opts) -> Result with best_solution found."""
    sign = 1 if minimize else -1
    evals = 0

    def evaluate(sol):
        nonlocal evals
        evals += 1
        return sign * objective_fn(sol)

    solution, obj = initial, evaluate(initial)
    best_solution, best_obj, best_iter = solution, obj, 0
    tabu_list, tabu_set = deque(maxlen=cooldown), set()

    for iteration in range(1, max_iter + 1):
        candidates = neighbors(solution)
        if not candidates:
            break

        best_move, best_neighbor, best_neighbor_obj = None, None, float('inf')

        for move, neighbor in candidates:
            neighbor_obj = evaluate(neighbor)
            if move in tabu_set and neighbor_obj >= best_obj:
                continue
            if neighbor_obj < best_neighbor_obj:
                best_neighbor_obj, best_neighbor, best_move = neighbor_obj, neighbor, move

        if best_neighbor is None:
            break

        solution, obj = best_neighbor, best_neighbor_obj

        if len(tabu_list) == cooldown:
            tabu_set.discard(tabu_list[0])

        tabu_list.append(best_move)
        tabu_set.add(best_move)

        if obj < best_obj:
            best_solution, best_obj, best_iter = solution, obj, iteration

        if iteration - best_iter >= max_no_improve:
            break

    final_obj = best_obj * sign
    return Result(best_solution, final_obj, iteration, evals, Status.FEASIBLE)

def solve_tsp(
    matrix: Sequence[Sequence[float]],
    *,
    minimize: bool = True,
    **kwargs,
) -> Result:
    """(distance_matrix, opts) -> Result with optimal tour as solution."""
    n = len(matrix)

    if n < 4:
        tour = list(range(n))
        obj = sum(matrix[a][b] for a, b in pairwise(tour + [tour[0]]))
        return Result(tour, obj, 0, 1, Status.FEASIBLE)

    def objective_fn(tour):
        return sum(matrix[a][b] for a, b in pairwise(tour + [tour[0]]))

    def neighbors(tour):
        moves = []

        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                new_tour = tour[:i + 1] + tour[i + 1:j + 1][::-1] + tour[j + 1:]
                moves.append(((i, j), new_tour))

        return moves

    tour, remaining = [0], set(range(1, n))

    while remaining:
        nearest = min(remaining, key=lambda c: matrix[tour[-1]][c])
        tour.append(nearest)
        remaining.remove(nearest)

    return tabu_search(tour, objective_fn, neighbors, minimize=minimize, **kwargs)

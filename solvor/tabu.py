"""
Tabu Search, local search with memory to escape cycles.

Like anneal, this explores neighbors. Unlike anneal, it's greedy: always pick
the best neighbor. The trick is the tabu list, a memory of recent moves that
are temporarily forbidden. This prevents cycling back to solutions you just
left, forcing the search to explore new territory.

Use this for routing (traveling salesmen probelm), scheduling, or when you want more control than
anneal. Tabu is deterministic where anneal is probabilistic, so results are
more reproducible and easier to debug.

    from solvor.tabu import tabu_search, solve_tsp

    result = tabu_search(start, objective_fn, neighbors_fn)
    result = solve_tsp(distance_matrix)  # built-in TSP helper

The neighbor function is different from anneal: it must return a list of
(move, new_solution) pairs, where move is hashable (so it can go in the tabu
list). Think (i, j) for "swap cities i and j" rather than just the new tour.

Genetic is population-based (more overhead, better diversity), anneal is
probabilistic (simpler setup), tabu is greedy with memory (more predictable).
"""

from collections import deque
from collections.abc import Callable, Sequence
from itertools import pairwise
from solvor.types import Status, Result, Progress, ProgressCallback

__all__ = ["tabu_search", "solve_tsp"]

def tabu_search[T, M](
    initial: T,
    objective_fn: Callable[[T], float],
    neighbors: Callable[[T], Sequence[tuple[M, T]]],
    *,
    minimize: bool = True,
    cooldown: int = 10,
    max_iter: int = 1000,
    max_no_improve: int = 100,
    on_progress: ProgressCallback | None = None,
    progress_interval: int = 0,
) -> Result:

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

        if on_progress and progress_interval > 0 and iteration % progress_interval == 0:
            current_obj = obj * sign
            best_so_far = best_obj * sign
            progress = Progress(iteration, current_obj, best_so_far if best_so_far != current_obj else None, evals)
            if on_progress(progress) is True:
                return Result(best_solution, best_so_far, iteration, evals, Status.FEASIBLE)

        if iteration - best_iter >= max_no_improve:
            break

    final_obj = best_obj * sign
    return Result(best_solution, final_obj, iteration, evals, Status.FEASIBLE)

def solve_tsp(
    matrix: Sequence[Sequence[float]],
    *,
    minimize: bool = True,
    on_progress: ProgressCallback | None = None,
    progress_interval: int = 0,
    **kwargs,
) -> Result:

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

    return tabu_search(tour, objective_fn, neighbors, minimize=minimize, on_progress=on_progress, progress_interval=progress_interval, **kwargs)

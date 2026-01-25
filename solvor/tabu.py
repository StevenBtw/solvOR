r"""
Tabu Search, local search with memory to escape cycles.

Like anneal, this explores neighbors. Unlike anneal, it's greedy: always pick
the best neighbor. The trick is the tabu list, a memory of recent moves that
are temporarily forbidden. This prevents cycling back to solutions you just
left, forcing the search to explore new territory.

Use this for routing (traveling salesmen problem), scheduling, or when you want more control than
anneal. Tabu is deterministic where anneal is probabilistic, so results are
more reproducible and easier to debug.

    from solvor.tabu import tabu_search, solve_tsp

    result = tabu_search(start, objective_fn, neighbors_fn)
    result = solve_tsp(distance_matrix)  # built-in TSP helper

How it works: each iteration picks the best neighbor not on the tabu list.
The chosen move goes on the list for `cooldown` iterations. Aspiration allows
tabu moves if they beat the global best.

Use this for:

- Routing (TSP, VRP)
- Scheduling and timetabling
- When you want deterministic, reproducible results
- When you need more control than anneal

The neighbor function must return (move, new_solution) pairs where move is
hashable (so it can go in the tabu list). Think (i, j) for "swap cities i and j".

Parameters:

    initial: starting solution
    objective_fn: function mapping solution to score
    neighbors: returns list of (move, new_solution) pairs
    cooldown: how long moves stay tabu (default: 10)
    max_no_improve: stop after this many iterations without improvement
    seed: random seed for tie-breaking when multiple neighbors have equal cost

Genetic is population-based (more overhead, better diversity), anneal is
probabilistic (simpler setup), tabu is greedy with memory (more predictable).
"""

from collections import deque
from collections.abc import Callable, Sequence
from itertools import pairwise
from random import Random

from solvor.types import ProgressCallback, Result, Status
from solvor.utils.helpers import Evaluator, report_progress

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
    seed: int | None = None,
    on_progress: ProgressCallback | None = None,
    progress_interval: int = 0,
) -> Result:
    rng = Random(seed)
    evaluate = Evaluator(objective_fn, minimize)

    solution, obj = initial, evaluate(initial)
    best_solution, best_obj, best_iter = solution, obj, 0
    tabu_list, tabu_set = deque(maxlen=cooldown), set()

    for iteration in range(1, max_iter + 1):
        candidates = list(neighbors(solution))
        if not candidates:
            break

        rng.shuffle(candidates)
        best_move, best_neighbor, best_neighbor_obj = None, None, float("inf")

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

        if report_progress(
            on_progress, progress_interval, iteration, evaluate.to_user(obj), evaluate.to_user(best_obj), evaluate.evals
        ):
            return Result(best_solution, evaluate.to_user(best_obj), iteration, evaluate.evals, Status.FEASIBLE)

        if iteration - best_iter >= max_no_improve:
            break

    final_obj = evaluate.to_user(best_obj)
    return Result(best_solution, final_obj, iteration, evaluate.evals, Status.FEASIBLE)


def solve_tsp(
    matrix: Sequence[Sequence[float]],
    *,
    minimize: bool = True,
    seed: int | None = None,
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
                new_tour = tour[: i + 1] + tour[i + 1 : j + 1][::-1] + tour[j + 1 :]
                moves.append(((i, j), new_tour))

        return moves

    tour, remaining = [0], set(range(1, n))

    while remaining:
        nearest = min(remaining, key=lambda c: matrix[tour[-1]][c])
        tour.append(nearest)
        remaining.remove(nearest)

    return tabu_search(
        tour,
        objective_fn,
        neighbors,
        minimize=minimize,
        seed=seed,
        on_progress=on_progress,
        progress_interval=progress_interval,
        **kwargs,
    )

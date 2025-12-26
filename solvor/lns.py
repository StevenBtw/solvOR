"""
Large Neighborhood Search (LNS) and Adaptive LNS (ALNS).

LNS iteratively destroys and repairs solutions, exploring large neighborhoods
that local search methods like 2-opt can't reach. ALNS extends this with
multiple operators and adaptive selection based on past performance.

Use this for complex routing, scheduling, or assignment problems where:
- Local moves (swaps, 2-opt) get stuck in local optima
- The solution structure allows partial destruction/reconstruction
- You have domain knowledge to design good destroy/repair operators

    from solvor.lns import lns, alns

    result = lns(initial, objective_fn, destroy, repair)
    result = alns(initial, objective_fn, destroy_ops, repair_ops)

The destroy function takes a solution and RNG, returns a partially destroyed
solution (e.g., with some elements removed). The repair function takes a
partial solution and RNG, returns a complete feasible solution.

Unlike tabu search which explores small neighborhoods exhaustively, LNS makes
large jumps by removing and reinserting significant portions of the solution.
This makes it effective for highly constrained problems like VRPTW where good
solutions are sparse.

ALNS adds adaptivity: multiple destroy/repair operators compete, and operators
that produce good results get selected more often. This automates the tuning
of which neighborhoods to explore.
"""

from collections.abc import Callable, Sequence
from math import exp
from random import Random

from solvor.types import ProgressCallback, Result, Status
from solvor.utils.helpers import Evaluator, report_progress

__all__ = ["lns", "alns"]

AcceptFn = Callable[[float, float, int, Random], bool]


def _accept_improving(current_obj: float, new_obj: float, iteration: int, rng: Random) -> bool:
    return new_obj < current_obj


def _accept_all(current_obj: float, new_obj: float, iteration: int, rng: Random) -> bool:
    return True


def _make_sa_accept(start_temp: float, cooling_rate: float) -> AcceptFn:
    def accept(current_obj: float, new_obj: float, iteration: int, rng: Random) -> bool:
        if new_obj < current_obj:
            return True
        temp = start_temp * (cooling_rate**iteration)
        if temp < 1e-10:
            return False
        delta = new_obj - current_obj
        return rng.random() < exp(-delta / temp)

    return accept


def _get_accept_fn(
    accept: str | AcceptFn,
    start_temp: float = 100.0,
    cooling_rate: float = 0.9995,
) -> AcceptFn:
    if not isinstance(accept, str):
        return accept
    if accept == "improving":
        return _accept_improving
    if accept == "accept_all":
        return _accept_all
    if accept == "simulated_annealing":
        return _make_sa_accept(start_temp, cooling_rate)
    raise ValueError(f"Unknown acceptance criterion: {accept}")


def lns[T](
    initial: T,
    objective_fn: Callable[[T], float],
    destroy: Callable[[T, Random], T],
    repair: Callable[[T, Random], T],
    *,
    minimize: bool = True,
    accept: str | AcceptFn = "improving",
    start_temp: float = 100.0,
    cooling_rate: float = 0.9995,
    max_iter: int = 1000,
    max_no_improve: int = 100,
    seed: int | None = None,
    on_progress: ProgressCallback | None = None,
    progress_interval: int = 0,
) -> Result:
    """Large Neighborhood Search with single destroy/repair operator."""
    rng = Random(seed)
    evaluate = Evaluator(objective_fn, minimize)
    accept_fn = _get_accept_fn(accept, start_temp, cooling_rate)

    current = initial
    current_obj = evaluate(current)
    best_solution, best_obj = current, current_obj
    best_iter = 0

    for iteration in range(1, max_iter + 1):
        partial = destroy(current, rng)
        candidate = repair(partial, rng)
        candidate_obj = evaluate(candidate)

        if accept_fn(current_obj, candidate_obj, iteration, rng):
            current, current_obj = candidate, candidate_obj
            if current_obj < best_obj:
                best_solution, best_obj = current, current_obj
                best_iter = iteration

        if report_progress(on_progress, progress_interval, iteration,
                          evaluate.to_user(current_obj), evaluate.to_user(best_obj), evaluate.evals):
            return Result(best_solution, evaluate.to_user(best_obj), iteration, evaluate.evals, Status.FEASIBLE)

        if iteration - best_iter >= max_no_improve:
            break

    final_obj = evaluate.to_user(best_obj)
    return Result(best_solution, final_obj, iteration, evaluate.evals, Status.FEASIBLE)


def alns[T](
    initial: T,
    objective_fn: Callable[[T], float],
    destroy_ops: Sequence[Callable[[T, Random], T]],
    repair_ops: Sequence[Callable[[T, Random], T]],
    *,
    minimize: bool = True,
    accept: str | AcceptFn = "simulated_annealing",
    start_temp: float = 100.0,
    cooling_rate: float = 0.9995,
    segment_size: int = 100,
    reaction_factor: float = 0.1,
    score_best: float = 3.0,
    score_better: float = 2.0,
    score_accept: float = 1.0,
    destroy_weights: Sequence[float] | None = None,
    repair_weights: Sequence[float] | None = None,
    max_iter: int = 10000,
    max_no_improve: int = 500,
    seed: int | None = None,
    on_progress: ProgressCallback | None = None,
    progress_interval: int = 0,
) -> Result:
    """Adaptive LNS with multiple competing destroy/repair operators."""
    if not destroy_ops or not repair_ops:
        raise ValueError("Need at least one destroy and one repair operator")

    rng = Random(seed)
    evaluate = Evaluator(objective_fn, minimize)
    accept_fn = _get_accept_fn(accept, start_temp, cooling_rate)

    n_destroy = len(destroy_ops)
    n_repair = len(repair_ops)

    d_weights = list(destroy_weights) if destroy_weights else [1.0] * n_destroy
    r_weights = list(repair_weights) if repair_weights else [1.0] * n_repair
    d_scores = [0.0] * n_destroy
    r_scores = [0.0] * n_repair
    d_counts = [0] * n_destroy
    r_counts = [0] * n_repair

    def select_weighted(weights: list[float]) -> int:
        total = sum(weights)
        r = rng.random() * total
        cumsum = 0.0
        for i, w in enumerate(weights):
            cumsum += w
            if r <= cumsum:
                return i
        return len(weights) - 1

    def update_weights(
        weights: list[float],
        scores: list[float],
        counts: list[int],
    ) -> None:
        for i in range(len(weights)):
            if counts[i] > 0:
                avg_score = scores[i] / counts[i]
                weights[i] = weights[i] * (1 - reaction_factor) + reaction_factor * avg_score
                weights[i] = max(weights[i], 0.1)  # Minimum weight
            scores[i] = 0.0
            counts[i] = 0

    current = initial
    current_obj = evaluate(current)
    best_solution, best_obj = current, current_obj
    best_iter = 0

    for iteration in range(1, max_iter + 1):
        d_idx = select_weighted(d_weights)
        r_idx = select_weighted(r_weights)

        partial = destroy_ops[d_idx](current, rng)
        candidate = repair_ops[r_idx](partial, rng)
        candidate_obj = evaluate(candidate)

        score = 0.0
        accepted = False

        if candidate_obj < best_obj:
            best_solution, best_obj = candidate, candidate_obj
            best_iter = iteration
            current, current_obj = candidate, candidate_obj
            score = score_best
            accepted = True
        elif candidate_obj < current_obj:
            current, current_obj = candidate, candidate_obj
            score = score_better
            accepted = True
        elif accept_fn(current_obj, candidate_obj, iteration, rng):
            current, current_obj = candidate, candidate_obj
            score = score_accept
            accepted = True

        if accepted:
            d_scores[d_idx] += score
            r_scores[r_idx] += score
        d_counts[d_idx] += 1
        r_counts[r_idx] += 1

        if iteration % segment_size == 0:
            update_weights(d_weights, d_scores, d_counts)
            update_weights(r_weights, r_scores, r_counts)

        if report_progress(on_progress, progress_interval, iteration,
                          evaluate.to_user(current_obj), evaluate.to_user(best_obj), evaluate.evals):
            return Result(best_solution, evaluate.to_user(best_obj), iteration, evaluate.evals, Status.FEASIBLE)

        if iteration - best_iter >= max_no_improve:
            break

    final_obj = evaluate.to_user(best_obj)
    return Result(best_solution, final_obj, iteration, evaluate.evals, Status.FEASIBLE)

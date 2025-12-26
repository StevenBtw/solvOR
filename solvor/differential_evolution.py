"""
Differential Evolution (DE) for global optimization.

Population-based stochastic search that's more systematic than genetic
algorithms for continuous problems. DE uses mutation by adding weighted
differences between population members, then crosses over with the target.

    from solvor.differential_evolution import differential_evolution

    result = differential_evolution(objective_fn, bounds)
    result = differential_evolution(objective_fn, bounds, strategy='best/1')

    # warm start from previous solutions
    result = differential_evolution(objective_fn, bounds, initial_population=[prev.solution])

Use DE when you need global optimization without gradients:
- Parameter fitting for models
- Black-box optimization
- Multi-modal functions with many local minima
- Hyperparameter tuning

For smooth local optimization, gradient methods are faster. For
high-dimensional problems (>100 vars), consider CMA-ES (Covariance
Matrix Adaptation Evolution Strategy) or other methods.
"""

from collections.abc import Callable, Sequence
from random import Random

from solvor.types import ProgressCallback, Result, Status
from solvor.utils.helpers import Evaluator, report_progress

__all__ = ["differential_evolution"]


def differential_evolution(
    objective_fn: Callable[[Sequence[float]], float],
    bounds: Sequence[tuple[float, float]],
    *,
    minimize: bool = True,
    population_size: int = 15,
    mutation: float = 0.8,
    crossover: float = 0.7,
    strategy: str = "rand/1",
    max_iter: int = 1000,
    tol: float = 1e-8,
    seed: int | None = None,
    initial_population: Sequence[Sequence[float]] | None = None,
    on_progress: ProgressCallback | None = None,
    progress_interval: int = 0,
) -> Result:
    """Global optimization using mutation and crossover of population vectors."""
    n = len(bounds)
    if n == 0:
        raise ValueError("bounds cannot be empty")

    rng = Random(seed)
    evaluate = Evaluator(objective_fn, minimize)

    # Parse strategy
    base, num_diffs = _parse_strategy(strategy)

    def clip(x: list[float]) -> list[float]:
        return [max(lo, min(hi, x[i])) for i, (lo, hi) in enumerate(bounds)]

    # Initialize population (use provided initial_population or random)
    pop_size = max(population_size, 4)  # Need at least 4 for mutation
    population: list[list[float]] = []

    if initial_population is not None:
        for ind in initial_population:
            if len(population) >= pop_size:
                break
            population.append(clip(list(ind)))

    # Fill remaining with random individuals
    while len(population) < pop_size:
        individual = [rng.uniform(lo, hi) for lo, hi in bounds]
        population.append(individual)

    fitness = [evaluate(ind) for ind in population]

    # Track best
    best_idx = min(range(pop_size), key=lambda i: fitness[i])
    best_solution = population[best_idx][:]
    best_obj = fitness[best_idx]

    for iteration in range(1, max_iter + 1):
        for i in range(pop_size):
            # Select base vector
            if base == "rand":
                candidates = [j for j in range(pop_size) if j != i]
                base_idx = rng.choice(candidates)
                base_vec = population[base_idx]
            else:  # "best"
                base_vec = best_solution

            # Select difference vectors
            candidates = [j for j in range(pop_size) if j != i]
            if base == "rand":
                candidates = [j for j in candidates if j != base_idx]

            if len(candidates) < 2 * num_diffs:
                # Fallback to rand/1 if not enough candidates
                diff_indices = rng.sample([j for j in range(pop_size) if j != i], 2)
            else:
                diff_indices = rng.sample(candidates, 2 * num_diffs)

            # Mutant vector
            mutant = base_vec[:]
            for d in range(num_diffs):
                r1, r2 = diff_indices[2 * d], diff_indices[2 * d + 1]
                for j in range(n):
                    mutant[j] += mutation * (population[r1][j] - population[r2][j])

            mutant = clip(mutant)

            # Crossover (binomial)
            trial = population[i][:]
            j_rand = rng.randrange(n)  # Ensure at least one dimension from mutant
            for j in range(n):
                if j == j_rand or rng.random() < crossover:
                    trial[j] = mutant[j]

            # Selection
            trial_fit = evaluate(trial)
            if trial_fit <= fitness[i]:
                population[i] = trial
                fitness[i] = trial_fit

                if trial_fit < best_obj:
                    best_solution = trial[:]
                    best_obj = trial_fit

        # Convergence check: population spread
        if _population_converged(population, tol):
            break

        if report_progress(on_progress, progress_interval, iteration,
                          evaluate.to_user(best_obj), evaluate.to_user(best_obj), evaluate.evals):
            return Result(best_solution, evaluate.to_user(best_obj), iteration, evaluate.evals, Status.FEASIBLE)

    final_obj = evaluate.to_user(best_obj)
    status = Status.OPTIMAL if iteration < max_iter else Status.MAX_ITER
    return Result(best_solution, final_obj, iteration, evaluate.evals, status)


def _parse_strategy(strategy: str) -> tuple[str, int]:
    """Parse DE strategy string like 'rand/1' or 'best/2'."""
    parts = strategy.lower().split("/")
    if len(parts) != 2:
        raise ValueError(f"Invalid strategy format: {strategy}. Use 'rand/1' or 'best/1'")

    base = parts[0]
    if base not in ("rand", "best"):
        raise ValueError(f"Unknown base vector type: {base}. Use 'rand' or 'best'")

    try:
        num_diffs = int(parts[1])
    except ValueError:
        raise ValueError(f"Invalid number of differences: {parts[1]}")

    if num_diffs < 1:
        raise ValueError("Number of differences must be at least 1")

    return base, num_diffs


def _population_converged(population: list[list[float]], tol: float) -> bool:
    """Check if population has converged (low variance in all dimensions)."""
    n = len(population[0])
    pop_size = len(population)

    for j in range(n):
        vals = [population[i][j] for i in range(pop_size)]
        mean_val = sum(vals) / pop_size
        variance = sum((v - mean_val) ** 2 for v in vals) / pop_size
        if variance > tol:
            return False

    return True

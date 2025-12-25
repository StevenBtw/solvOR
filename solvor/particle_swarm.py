"""
Particle Swarm Optimization (PSO) for global optimization.

Swarm intelligence metaheuristic where particles fly through the search space,
influenced by their own best position and the swarm's best position. Each
particle adjusts velocity based on cognitive (personal best) and social
(global best) components.

    from solvor.particle_swarm import particle_swarm

    result = particle_swarm(objective_fn, bounds)
    result = particle_swarm(objective_fn, bounds, n_particles=50, inertia=0.9)

    # warm start from previous solutions
    result = particle_swarm(objective_fn, bounds, initial_positions=[prev.solution])

Use PSO when you need:
- Global optimization without gradients
- Multi-modal functions with many local optima
- Parameter tuning, hyperparameter optimization
- Continuous optimization problems

PSO is simpler than differential evolution and often faster to converge on
well-behaved functions. For highly constrained or discrete problems, consider
genetic algorithms or CP-SAT instead.
"""

from collections.abc import Callable, Sequence
from random import Random

from solvor.types import Progress, ProgressCallback, Result, Status

__all__ = ["particle_swarm"]


def particle_swarm(
    objective_fn: Callable[[Sequence[float]], float],
    bounds: Sequence[tuple[float, float]],
    *,
    minimize: bool = True,
    n_particles: int = 30,
    max_iter: int = 1000,
    inertia: float = 0.7,
    inertia_decay: float | None = None,
    cognitive: float = 1.5,
    social: float = 1.5,
    v_max: float | None = None,
    seed: int | None = None,
    initial_positions: Sequence[Sequence[float]] | None = None,
    on_progress: ProgressCallback | None = None,
    progress_interval: int = 0,
) -> Result:
    """
    Swarm intelligence optimization using particle velocities.

    Args:
        inertia_decay: If set, linearly decay inertia from `inertia` to this value
        v_max: Maximum velocity per dimension (default: 0.2 * range)
    """
    n = len(bounds)
    if n == 0:
        raise ValueError("bounds cannot be empty")

    rng = Random(seed)
    sign = 1 if minimize else -1
    evals = 0

    # Velocity limits per dimension
    v_limits = []
    for lo, hi in bounds:
        v_lim = v_max if v_max is not None else 0.2 * (hi - lo)
        v_limits.append(v_lim)

    def evaluate(x: list[float]) -> float:
        nonlocal evals
        evals += 1
        return sign * objective_fn(x)

    def clip(x: list[float]) -> list[float]:
        return [max(lo, min(hi, x[j])) for j, (lo, hi) in enumerate(bounds)]

    def clip_velocity(vel: list[float]) -> list[float]:
        return [max(-v_limits[j], min(v_limits[j], vel[j])) for j in range(n)]

    # Initialize particles (use provided initial_positions or random)
    positions: list[list[float]] = []
    velocities: list[list[float]] = []

    if initial_positions is not None:
        for pos in initial_positions:
            if len(positions) >= n_particles:
                break
            positions.append(clip(list(pos)))
            vel = [(hi - lo) * (rng.random() - 0.5) * 0.1 for lo, hi in bounds]
            velocities.append(clip_velocity(vel))

    # Fill remaining with random particles
    while len(positions) < n_particles:
        pos = [rng.uniform(lo, hi) for lo, hi in bounds]
        vel = [(hi - lo) * (rng.random() - 0.5) * 0.1 for lo, hi in bounds]
        positions.append(pos)
        velocities.append(clip_velocity(vel))

    # Evaluate initial positions
    fitness = [evaluate(pos) for pos in positions]

    # Personal best for each particle
    p_best = [pos[:] for pos in positions]
    p_best_fit = fitness[:]

    # Global best
    best_idx = min(range(n_particles), key=lambda i: fitness[i])
    best_solution = positions[best_idx][:]
    best_obj = fitness[best_idx]

    for iteration in range(1, max_iter + 1):
        # Compute current inertia (with optional decay)
        if inertia_decay is not None:
            w = inertia - (inertia - inertia_decay) * (iteration / max_iter)
        else:
            w = inertia

        for i in range(n_particles):
            # Update velocity
            for j in range(n):
                r1, r2 = rng.random(), rng.random()
                cog = cognitive * r1 * (p_best[i][j] - positions[i][j])
                soc = social * r2 * (best_solution[j] - positions[i][j])
                velocities[i][j] = w * velocities[i][j] + cog + soc

            # Clamp velocity
            velocities[i] = clip_velocity(velocities[i])

            # Update position
            for j in range(n):
                positions[i][j] += velocities[i][j]

            # Clip to bounds
            positions[i] = clip(positions[i])

            # Evaluate new position
            fitness[i] = evaluate(positions[i])

            # Update personal best
            if fitness[i] < p_best_fit[i]:
                p_best[i] = positions[i][:]
                p_best_fit[i] = fitness[i]

                # Update global best
                if fitness[i] < best_obj:
                    best_solution = positions[i][:]
                    best_obj = fitness[i]

        if on_progress and progress_interval > 0 and iteration % progress_interval == 0:
            obj = best_obj * sign
            progress = Progress(iteration, obj, None, evals)
            if on_progress(progress) is True:
                return Result(best_solution, obj, iteration, evals, Status.FEASIBLE)

    final_obj = best_obj * sign
    return Result(best_solution, final_obj, iteration, evals, Status.FEASIBLE)

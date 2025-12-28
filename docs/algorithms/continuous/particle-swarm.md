# particle_swarm

Swarm intelligence. Particles fly through the search space, attracted to their personal best and the global best. Like peer pressure for optimization.

## Example

```python
from solvor import particle_swarm

def objective(x):
    return sum(xi**2 for xi in x)

result = particle_swarm(
    objective,
    bounds=[(-10, 10)] * 5,
    n_particles=30,
    max_iter=1000
)
print(result.solution)  # Close to [0, 0, 0, 0, 0]
```

## Parameters

| Parameter | Description |
|-----------|-------------|
| `bounds` | List of (min, max) for each dimension |
| `n_particles` | Swarm size (default 30) |
| `inertia` | Inertia weight/momentum (default 0.7) |
| `cognitive` | Personal best attraction (default 1.5) |
| `social` | Global best attraction (default 1.5) |
| `inertia_decay` | If set, linearly decay inertia to this value |
| `initial_positions` | Warm-start with known good positions |

## How It Works

**The swarm metaphor:** Imagine birds searching for food. Each bird remembers where it personally found the best food, and the flock shares where the best food was found overall. Birds fly toward both, with some randomness.

**The velocity equation:** Each particle's velocity combines three forces:

```text
v = inertia·v + cognitive·r₁·(pbest - x) + social·r₂·(gbest - x)
```

- **Inertia:** Keep going the direction you were going (momentum)
- **Cognitive:** Pull toward your personal best position
- **Social:** Pull toward the swarm's global best

r₁ and r₂ are random factors (0 to 1) that add stochasticity.

**The algorithm:**

1. Initialize particles with random positions and velocities
2. Evaluate objective for all particles
3. Update each particle's personal best if current is better
4. Update global best if any particle found a new best
5. Update velocities using the equation above
6. Update positions: x = x + v
7. Clamp positions to bounds
8. Repeat until converged

**Why it works:** The swarm balances exploitation (converging toward known good areas) and exploration (inertia carries particles through new territory). Random factors prevent premature convergence.

**Inertia decay:** Start with high inertia (explore widely), decrease over time (exploit best regions). This is like cooling in simulated annealing.

**Compared to GA:** No crossover or selection operators—just physics-like attraction. Simpler to implement, fewer hyperparameters, often works well out of the box.

## Tips

- **Velocity clamping built-in.** Particles won't yeet into infinity.
- **Good for exploration.** Swarm naturally spreads out.
- **Fewer parameters than GA.** Easier to tune.

## See Also

- [Differential Evolution](differential-evolution.md) - Another population method

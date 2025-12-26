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
| `n_particles` | Swarm size |
| `w` | Inertia weight (momentum) |
| `c1` | Cognitive coefficient (personal best attraction) |
| `c2` | Social coefficient (global best attraction) |

## How It Works

1. Initialize particles with random positions and velocities
2. Evaluate all particles
3. Update personal and global bests
4. Update velocities: v = w·v + c1·r1·(pbest - x) + c2·r2·(gbest - x)
5. Update positions: x = x + v
6. Repeat

## Tips

- **Velocity clamping built-in.** Particles won't yeet into infinity.
- **Good for exploration.** Swarm naturally spreads out.
- **Fewer parameters than GA.** Easier to tune.

## See Also

- [Differential Evolution](differential-evolution.md) - Another population method

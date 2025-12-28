# evolve

[Genetic algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm). Maintains a population, combines solutions via crossover, occasionally mutates. Slower than single-solution methods but explores more diversity. Good for multi-objective problems.

## Signature

```python
def evolve[T](
    objective_fn: Callable[[T], float],
    population: Sequence[T],
    crossover: Callable[[T, T], T],
    mutate: Callable[[T], T],
    *,
    minimize: bool = True,
    elite_size: int = 2,
    mutation_rate: float = 0.1,
    adaptive_mutation: bool = False,
    max_iter: int = 100,
    tournament_k: int = 3,
    seed: int | None = None,
    on_progress: ProgressCallback | None = None,
    progress_interval: int = 0,
) -> Result[T]
```

## Parameters

| Parameter | Description |
|-----------|-------------|
| `objective_fn` | Function to minimize (or maximize if `minimize=False`) |
| `population` | Initial population of solutions |
| `crossover` | Combine two parents into child |
| `mutate` | Randomly modify a solution |
| `minimize` | If False, maximize instead |
| `elite_size` | Keep best N across generations |
| `mutation_rate` | Probability of mutation (0.0–1.0) |
| `adaptive_mutation` | Increase rate when stuck, decrease when improving |
| `max_iter` | Number of generations |
| `tournament_k` | Tournament size for selection (higher = greedier) |
| `seed` | Random seed for reproducibility |
| `on_progress` | Progress callback (return True to stop early) |
| `progress_interval` | Call progress every N iterations (0 = disabled) |

## Example

```python
from solvor import evolve
import random

def fitness(x):
    return sum(xi**2 for xi in x)

def crossover(a, b):
    # Uniform crossover
    return [ai if random.random() < 0.5 else bi for ai, bi in zip(a, b)]

def mutate(x):
    x = list(x)
    i = random.randint(0, len(x)-1)
    x[i] += random.gauss(0, 0.5)
    return x

population = [[random.uniform(-10, 10) for _ in range(5)] for _ in range(50)]
result = evolve(fitness, population, crossover, mutate, max_iter=100)
print(result.solution)  # Close to [0, 0, 0, 0, 0]
```

## How It Works

1. Evaluate fitness of all individuals
2. Select parents (tournament selection)
3. Crossover parents to create children
4. Mutate children with some probability
5. Keep elite individuals unchanged
6. Replace population and repeat

## Tips

- **Crossover is critical.** Bad crossover = expensive random search.
- **Elite preservation.** Keep the best solutions across generations.
- **Adaptive mutation.** Enable to escape plateaus.

## See Also

- [differential_evolution](../continuous/differential-evolution.md) - For continuous spaces
- [particle_swarm](../continuous/particle-swarm.md) - Another population method

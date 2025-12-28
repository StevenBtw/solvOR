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

**The biological metaphor:** Evolution optimizes fitness through selection, recombination, and mutation. GA mimics this: fit individuals reproduce, offspring inherit traits from both parents, occasional mutations introduce novelty.

**Selection pressure:** Tournament selection picks k random individuals and keeps the fittest. Higher k = more pressure toward best solutions. Too high and you converge prematurely; too low and you waste time on bad solutions.

**Crossover (recombination):** The key operator. Combine two good solutions hoping the child inherits the best parts of each.

```text
Parent A: [1, 2, 3, 4, 5]
Parent B: [5, 4, 3, 2, 1]

Uniform crossover (coin flip each position):
Child:    [1, 4, 3, 2, 5]
```

**Mutation:** Random small changes prevent premature convergence. If everyone looks the same, mutation introduces diversity. Rate typically 1-10%.

**Elitism:** Copy the best k individuals unchanged to the next generation. Guarantees you never lose your best solution to bad luck.

**The algorithm:**

1. Evaluate fitness of all individuals
2. Select parents via tournament: pick k random, keep fittest
3. Crossover pairs of parents to create children
4. Mutate children with probability mutation_rate
5. Keep elite_size best individuals unchanged
6. Replace population with children + elites
7. Repeat for max_iter generations

**Building block hypothesis:** GA works when good solutions share "building blocks"—partial solutions that can combine. If your crossover destroys good structure, GA degrades to random search.

**Population diversity:** Track fitness variance. If everyone has similar fitness, increase mutation or restart with fresh individuals.

## Tips

- **Crossover is critical.** Bad crossover = expensive random search.
- **Elite preservation.** Keep the best solutions across generations.
- **Adaptive mutation.** Enable to escape plateaus.

## See Also

- [differential_evolution](../continuous/differential-evolution.md) - For continuous spaces
- [particle_swarm](../continuous/particle-swarm.md) - Another population method

# differential_evolution

Evolution strategy for continuous optimization. Maintains a population, mutates by adding weighted differences between individuals.

## Example

```python
from solvor import differential_evolution

def objective(x):
    return sum(xi**2 for xi in x)

result = differential_evolution(
    objective,
    bounds=[(-10, 10)] * 5,
    population_size=50,
    max_iter=1000
)
print(result.solution)  # Close to [0, 0, 0, 0, 0]
```

## Parameters

| Parameter | Description |
|-----------|-------------|
| `bounds` | List of (min, max) for each dimension |
| `population_size` | Population size (default: 15) |
| `mutation` | Mutation scale (default: 0.8) |
| `crossover` | Crossover probability (default: 0.7) |
| `strategy` | Mutation strategy: `rand/1`, `best/1`, etc. (default: `rand/1`) |

## When to Use

- Global search in continuous spaces
- Non-convex landscapes with many local minima
- No gradient information

## How It Works

**The core insight:** Use the *difference between population members* as a mutation direction. If two individuals differ, that difference vector points from one to the other—a direction in the search space. Add this to a third individual to create a mutant.

**The mutation step (rand/1):**

```text
mutant = x_r1 + F × (x_r2 - x_r3)
```

Pick three random individuals. The difference (x_r2 - x_r3) is scaled by F (mutation factor) and added to x_r1. The mutant "points" in a direction informed by population diversity.

**The crossover step:** Mix the mutant with the target vector:

```text
trial[i] = mutant[i]  if rand() < CR or i == j_rand
           target[i]  otherwise
```

CR is crossover probability. j_rand ensures at least one dimension comes from the mutant.

**The selection step:** Simple and greedy—keep whichever is better:

```text
if f(trial) < f(target):
    next_gen[i] = trial
else:
    next_gen[i] = target
```

**The algorithm:**

1. Initialize population randomly within bounds
2. For each individual (target):
   - Pick 3 other random individuals
   - Create mutant via mutation formula
   - Create trial via crossover with target
   - Replace target with trial if trial is better
3. Repeat for max_iter generations

**Why differences work:** Population diversity naturally creates adaptive step sizes. When the population is spread out, differences are large (explore). When converging, differences shrink (exploit). No manual step size tuning needed.

**Strategies:** "rand/1" uses random base vector. "best/1" uses the best individual as base—faster convergence but may get stuck. "rand/2" uses two difference vectors for more exploration.

## See Also

- [Particle Swarm](particle-swarm.md) - Another population method
- [Genetic Algorithms](../metaheuristics/genetic.md) - For discrete spaces

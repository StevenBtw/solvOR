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

## See Also

- [Particle Swarm](particle-swarm.md) - Another population method
- [Genetic Algorithms](../metaheuristics/genetic.md) - For discrete spaces

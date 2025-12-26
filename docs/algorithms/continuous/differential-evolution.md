# differential_evolution

Evolution strategy for continuous optimization. Maintains a population, mutates by adding weighted differences between individuals.

## Example

```python
from solvor import differential_evolution

def objective(x):
    return sum(xi**2 for xi in x)

result = differential_evolution(
    objective,
    bounds=[(−10, 10)] * 5,
    pop_size=50,
    max_iter=1000
)
print(result.solution)  # Close to [0, 0, 0, 0, 0]
```

## Parameters

| Parameter | Description |
|-----------|-------------|
| `bounds` | List of (min, max) for each dimension |
| `pop_size` | Population size (default: 10 × dimensions) |
| `F` | Mutation scale (default: 0.8) |
| `CR` | Crossover probability (default: 0.9) |

## When to Use

- Global search in continuous spaces
- Non-convex landscapes with many local minima
- No gradient information

## See Also

- [Particle Swarm](particle-swarm.md) - Another population method
- [Genetic Algorithms](../metaheuristics/genetic.md) - For discrete spaces

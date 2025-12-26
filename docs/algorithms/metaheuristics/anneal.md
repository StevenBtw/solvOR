# anneal

Simulated annealing. Accepts worse solutions probabilistically, cooling down over time. Like a ball rolling on a landscape, energetic enough early on to escape local valleys, settling into the best valley it finds.

## Signature

```python
def anneal[T](
    initial: T,
    objective_fn: Callable[[T], float],
    neighbors: Callable[[T], T],
    *,
    minimize: bool = True,
    temperature: float = 1000.0,
    cooling: float | CoolingSchedule = 0.9995,
    min_temp: float = 1e-8,
    max_iter: int = 100_000,
    seed: int | None = None,
    on_progress: ProgressCallback | None = None,
    progress_interval: int = 0,
) -> Result[T]
```

## Parameters

| Parameter | Description |
|-----------|-------------|
| `initial` | Starting solution |
| `objective_fn` | Function to minimize (or maximize if `minimize=False`) |
| `neighbors` | Function returning a random neighbor |
| `minimize` | If False, maximize instead |
| `temperature` | Starting temperature (higher = more exploration) |
| `cooling` | Multiplier per iteration, or a CoolingSchedule function |
| `min_temp` | Stop when temperature drops below this |
| `max_iter` | Maximum iterations |
| `seed` | Random seed for reproducibility |
| `on_progress` | Progress callback (return True to stop early) |
| `progress_interval` | Call progress every N iterations (0 = disabled) |

## Example

```python
from solvor import anneal
import random

def objective(x):
    return sum(xi**2 for xi in x)

def neighbor(x):
    i = random.randint(0, len(x)-1)
    x_new = list(x)
    x_new[i] += random.uniform(-0.5, 0.5)
    return x_new

result = anneal([5, 5, 5], objective, neighbor, max_iter=50000)
print(result.solution)  # Close to [0, 0, 0]
```

## Cooling Schedules

The `cooling` parameter can be a float (exponential decay rate) or a schedule function:

```python
from solvor import anneal, exponential_cooling, linear_cooling, logarithmic_cooling

# Exponential (default): temp = initial * rate^iter
result = anneal(initial, obj, neighbors, cooling=0.9995)
result = anneal(initial, obj, neighbors, cooling=exponential_cooling(0.999))

# Linear: temp decreases linearly to min_temp
result = anneal(initial, obj, neighbors, cooling=linear_cooling(min_temp=1e-6))

# Logarithmic: temp = initial / (1 + c * log(1 + iter)), very slow cooling
result = anneal(initial, obj, neighbors, cooling=logarithmic_cooling(c=1.0))
```

## How It Works

1. Start with initial solution at high temperature
2. Generate random neighbor
3. If neighbor is better, accept it
4. If neighbor is worse, accept with probability exp(-delta/T)
5. Reduce temperature according to cooling schedule
6. Stop when temperature drops below `min_temp` or `max_iter` reached

## Reproducibility

Use `seed` for deterministic runs:

```python
result1 = anneal(initial, obj, neighbors, seed=42)
result2 = anneal(initial, obj, neighbors, seed=42)
# result1.solution == result2.solution
```

## Tips

- **Higher temperature = more exploration.** Start hot to escape local optima.
- **Slower cooling = better solutions.** But takes longer.
- **Small neighbor moves.** Make local perturbations, don't teleport randomly.
- **Getting stuck?** Try higher `temperature` or slower `cooling` (closer to 1.0).

## See Also

- [tabu_search](tabu.md) - Deterministic alternative with memory
- [Metaheuristics Overview](index.md)

# anneal

[Simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing). Accepts worse solutions probabilistically, cooling down over time. Like a ball rolling on a landscape, energetic enough early on to escape local valleys, settling into the best valley it finds.

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

**The metallurgy analogy:** In real annealing, you heat metal until atoms move freely, then cool slowly so atoms settle into a low-energy crystal structure. Cool too fast and you get brittle metal with defects (stuck in local minimum). The algorithm mimics this.

**The math:** Accept worse solutions with probability:

```text
P(accept) = exp(-Δcost / temperature)
```

At high temperature, this is near 1, so accept almost anything. At low temperature, this approaches 0, so only accept improvements. The exponential form comes from statistical mechanics (Boltzmann distribution).

**Why it works:** Early on, high temperature lets you escape local optima by accepting worse moves. As you cool, you become more selective, converging toward a good solution. The probability of accepting a bad move depends on *how bad*—small backward steps are more likely than large ones.

**The algorithm:**

1. Start at high temperature with initial solution
2. Pick a random neighbor
3. If better, accept. If worse, accept with probability exp(-Δ/T)
4. Cool down: T ← T × cooling_rate
5. Repeat until cold or max iterations

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

# anneal

Simulated annealing. Accepts worse solutions probabilistically, cooling down over time. Like a ball rolling on a landscape, energetic enough early on to escape local valleys, settling into the best valley it finds.

## Signature

```python
def anneal[T](
    initial: T,
    objective_fn: Callable[[T], float],
    neighbors: Callable[[T], T],
    *,
    temperature: float = 1000.0,
    cooling: float = 0.9995,
    max_iter: int = 100_000,
    on_progress: Callable[[Progress], bool | None] | None = None,
    progress_interval: int = 0,
) -> Result[T]
```

## Parameters

| Parameter | Description |
|-----------|-------------|
| `initial` | Starting solution |
| `objective_fn` | Function to minimize |
| `neighbors` | Function returning a random neighbor |
| `temperature` | Starting temperature (higher = more exploration) |
| `cooling` | Multiplier per iteration (closer to 1 = slower cooling) |
| `max_iter` | Maximum iterations |
| `on_progress` | Progress callback (return True to stop) |
| `progress_interval` | Call progress every N iterations |

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

## How It Works

1. Start with initial solution at high temperature
2. Generate random neighbor
3. If neighbor is better, accept it
4. If neighbor is worse, accept with probability exp(-delta/T)
5. Reduce temperature: T = T × cooling
6. Repeat

## Tips

- **Higher temperature = more exploration.** Start hot to escape local optima.
- **Slower cooling = better solutions.** But takes longer.
- **Small neighbor moves.** Make local perturbations, don't teleport randomly.

## See Also

- [tabu_search](tabu.md) - Deterministic alternative
- [Metaheuristics Overview](index.md)

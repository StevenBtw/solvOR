# lns / alns

Large Neighborhood Search. Destroy part of your solution and rebuild it better. ALNS (Adaptive LNS) learns which operators work best.

## Signature

```python
def lns[T](
    initial: T,
    objective_fn: Callable[[T], float],
    destroy_ops: Sequence[Callable[[T], T]],
    repair_ops: Sequence[Callable[[T], T]],
    *,
    max_iter: int = 1000,
    on_progress: Callable[[Progress], bool | None] | None = None,
    progress_interval: int = 0,
) -> Result[T]

def alns[T](
    initial: T,
    objective_fn: Callable[[T], float],
    destroy_ops: Sequence[Callable[[T], T]],
    repair_ops: Sequence[Callable[[T], T]],
    *,
    max_iter: int = 1000,
    learning_rate: float = 0.1,
    on_progress: Callable[[Progress], bool | None] | None = None,
    progress_interval: int = 0,
) -> Result[T]
```

## Parameters

| Parameter | Description |
|-----------|-------------|
| `initial` | Starting solution |
| `objective_fn` | Function to minimize |
| `destroy_ops` | List of destroy operators |
| `repair_ops` | List of repair operators |
| `max_iter` | Maximum iterations |
| `learning_rate` | How fast ALNS adapts weights |

## Example

```python
from solvor import lns, alns
import random

# VRP-like problem
def destroy_random(routes):
    # Remove 20% of customers randomly
    routes = [list(r) for r in routes]
    to_remove = random.sample(range(len(routes)), k=max(1, len(routes)//5))
    for i in sorted(to_remove, reverse=True):
        del routes[i]
    return routes

def destroy_worst(routes):
    # Remove customers with highest cost contribution
    # ... implementation
    return routes

def repair_greedy(routes):
    # Insert unassigned customers greedily
    # ... implementation
    return routes

result = lns(initial, objective, [destroy_random, destroy_worst], [repair_greedy])

# ALNS learns which operators work best
result = alns(initial, objective, [destroy_random, destroy_worst], [repair_greedy])
```

## How It Works

**LNS:**
1. Select random destroy and repair operators
2. Destroy part of solution
3. Repair to get new solution
4. Accept if better (or probabilistically)
5. Repeat

**ALNS adds:**
- Track success of each operator
- Increase weight for successful operators
- Select operators by weight

## Tips

- **Multiple destroy operators.** Random, worst-cost, cluster-based, etc.
- **Multiple repair operators.** Greedy, regret, random.
- **Destruction degree.** Destroy 10-40% of solution typically.

## See Also

- [tabu_search](tabu.md) - Simpler local search
- [Cookbook: Job Shop](../../cookbook/job-shop.md) - Scheduling example

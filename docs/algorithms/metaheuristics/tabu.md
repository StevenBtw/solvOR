# tabu_search

Greedy local search with memory. Always picks the best neighbor, but maintains a "tabu list" of recent moves to prevent cycling. More deterministic than annealing, easier to debug.

## Signature

```python
def tabu_search[T, M](
    initial: T,
    objective_fn: Callable[[T], float],
    neighbors: Callable[[T], Sequence[tuple[M, T]]],
    *,
    minimize: bool = True,
    cooldown: int = 10,
    max_iter: int = 1000,
    max_no_improve: int = 100,
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
| `neighbors` | Returns (move, solution) pairs - move must be hashable |
| `minimize` | If False, maximize instead |
| `cooldown` | How long a move stays forbidden |
| `max_iter` | Maximum iterations |
| `max_no_improve` | Stop if no improvement for this many iterations |
| `seed` | Random seed for reproducibility |
| `on_progress` | Progress callback (return True to stop) |
| `progress_interval` | Call progress every N iterations (0 = disabled) |

## Example

```python
from solvor import tabu_search

def objective(perm):
    # TSP-like: sum of adjacent distances
    return sum(abs(perm[i] - perm[i+1]) for i in range(len(perm)-1))

def neighbors(perm):
    # Generate all 2-opt swaps
    moves = []
    for i in range(len(perm)):
        for j in range(i+2, len(perm)):
            new = list(perm)
            new[i], new[j] = new[j], new[i]
            moves.append(((i, j), tuple(new)))
    return moves

result = tabu_search([0, 3, 1, 4, 2], objective, neighbors, cooldown=5)
print(result.solution)
```

## How It Works

1. Evaluate all neighbors
2. Pick the best non-tabu neighbor (or tabu if it's a new best)
3. Add the move to tabu list with cooldown counter
4. Decrement all cooldowns, remove expired
5. Repeat

## Tips

- **Cooldown length matters.** Too short: cycles. Too long: missed opportunities.
- **Aspiration criteria.** Override tabu if you find a new global best (built-in).
- **Diversification.** If stuck, restart or add random moves.

## See Also

- [anneal](anneal.md) - Probabilistic alternative
- [solve_tsp](../../cookbook/tsp.md) - Uses tabu search internally

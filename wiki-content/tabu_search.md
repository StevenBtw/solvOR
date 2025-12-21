# tabu_search

**Greedy local search with memory to prevent cycling.**

## Summary

- **Category:** [Metaheuristics](Metaheuristics)
- **Problem Type:** Combinatorial optimization
- **Complexity:** O(iterations × neighbors × eval_cost)
- **Guarantees:** None (deterministic with seed)
- **Status:** Returns `FEASIBLE`

## When to Use This

Use for routing, scheduling, or when you want more control than annealing. Tabu is deterministic and reproducible, making it easier to debug.

**Real examples:**
- Vehicle routing
- Job shop scheduling
- Graph coloring
- Frequency assignment

## Quick Example

```python
from solvor import tabu_search

def objective(tour):
    # Tour cost

def neighbors(tour):
    # Return list of (move, new_tour) pairs
    return []

result = tabu_search(initial_tour, objective, neighbors)
```

## See Also
- [`solve_tsp`](solve_tsp)
- [`anneal`](anneal)

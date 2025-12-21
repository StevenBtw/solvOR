# anneal

**Simulated annealing - probabilistic hill climbing that escapes local optima.**

## Summary

- **Category:** [Metaheuristics](Metaheuristics)
- **Problem Type:** Black-box optimization
- **Complexity:** O(iterations × neighbor_cost)
- **Guarantees:** None (heuristic)
- **Status:** Returns `FEASIBLE` or `MAX_ITER`

## When to Use This

Perfect when you have an objective function and can generate neighbors, but don't know much else about the problem structure. Fast to prototype, handles many local optima well.

**Real examples:**
- TSP with custom constraints
- Layout optimization
- Parameter tuning for simulators
- Any "move pieces around" problem

## Quick Example

```python
from solvor import anneal

def objective(x):
    return sum(xi**2 for xi in x)

def neighbor(x):
    import random
    i = random.randint(0, len(x)-1)
    x_new = list(x)
    x_new[i] += random.uniform(-0.5, 0.5)
    return x_new

result = anneal([5, 5, 5], objective, neighbor, max_iter=10000)
print(result.solution)  # Close to [0, 0, 0]
```

##See Also
- [Metaheuristics](Metaheuristics)
- [`tabu_search`](tabu_search)

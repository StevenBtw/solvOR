# solve_tsp

**Traveling Salesman Problem solver using tabu search with 2-opt.**

## Summary 

- **Category:** [Metaheuristics](Metaheuristics)
- **Problem Type:** TSP
- **Complexity:** O(n² iterations)
- **Guarantees:** Heuristic solution
- **Status:** Returns `FEASIBLE`

## When to Use This

The go-to for traveling salesman. Just pass a distance matrix and get a good tour.

## Quick Example

```python
from solvor import solve_tsp

distances = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]

result = solve_tsp(distances)
print(result.solution)  # Tour like [0, 1, 3, 2]
print(result.objective) # Total distance
```

## See Also
- [Cookbook: TSP](Cookbook-TSP)

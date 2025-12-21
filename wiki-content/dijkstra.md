# dijkstra

**Dijkstra's algorithm, weighted shortest paths.**

## Summary 

- **Category:** [Pathfinding](Pathfinding)
- **Problem Type:** Weighted shortest path (non-negative)
- **Complexity:** O((V + E) log V)
- **Guarantees:** Optimal for non-negative weights

Dijkstra's negativity was legendary, just not in his algorithm.

## Quick Example

```python
from solvor import dijkstra

graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('C', 2), ('D', 5)],
    'C': [('D', 1)],
    'D': []
}

result = dijkstra('A', 'D', lambda n: graph[n])
print(result.solution)  # ['A', 'B', 'C', 'D']
print(result.objective)  # 4
```

## See Also
- [`astar`](astar-&-astar_grid) - With heuristic
- [`bellman_ford`](bellman_ford) - For negative edges

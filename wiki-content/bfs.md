# bfs

**Breadth-first search, shortest path in unweighted graphs.**

## Summary 

- **Category:** [Pathfinding](Pathfinding)
- **Problem Type:** Unweighted shortest path
- **Complexity:** O(V + E)
- **Guarantees:** Optimal for unweighted graphs

## Quick Example

```python
from solvor import bfs

graph = {
    'A': ['B', 'C'],
    'B': ['D'],
    'C': ['D'],
    'D': []
}

result = bfs('A', 'D', lambda n: graph[n])
print(result.solution)  # ['A', 'B', 'D'] or ['A', 'C', 'D']
```

## See Also
- [`dijkstra`](dijkstra) - For weighted graphs

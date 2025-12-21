# bellman_ford

**Shortest paths with negative edges - the true edgelord.**

## At a Glance

- **Category:** [Pathfinding](Pathfinding)
- **Problem Type:** Shortest path with negative weights
- **Complexity:** O(VE)
- **Guarantees:** Detects negative cycles

## Quick Example

```python
from solvor import bellman_ford

edges = [
    (0, 1, 4),   # u, v, weight
    (0, 2, 3),
    (1, 2, -2),  # Negative edge!
    (2, 3, 1)
]

result = bellman_ford(start=0, edges=edges, n_nodes=4, target=3)
```

## See Also
- [`dijkstra`](dijkstra) - Faster for non-negative

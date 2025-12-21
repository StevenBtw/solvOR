# floyd_warshall

**All-pairs shortest paths, O(V³) but gets everything at once.**

## Summary 

- **Category:** [Pathfinding](Pathfinding)
- **Problem Type:** All-pairs shortest paths
- **Complexity:** O(V³)

## Quick Example

```python
from solvor import floyd_warshall

edges = [(0, 1, 4), (1, 2, 3), (0, 2, 10)]
result = floyd_warshall(n_nodes=3, edges=edges)
distances = result.solution  # dist[i][j] = shortest path from i to j
```

# kruskal

**Minimum spanning tree - greedy edge selection with union-find.**

## At a Glance

- **Category:** [Network Flow & MST](Network-Flow-&-MST)
- **Problem Type:** MST
- **Complexity:** O(E log E)
- **Guarantees:** Optimal MST

One of those rare greedy = optimal algorithms.

## Quick Example

```python
from solvor import kruskal

edges = [
    (0, 1, 4), (0, 2, 3), (1, 2, 2),  # (u, v, weight)
    (1, 3, 5), (2, 3, 6)
]

result = kruskal(n_nodes=4, edges=edges)
print(result.solution)  # [(1, 2, 2), (0, 2, 3), (0, 1, 4)]
print(result.objective)  # 9
```

## See Also
- [`prim`](prim) - Alternative MST algorithm

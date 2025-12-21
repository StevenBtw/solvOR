# prim

**Minimum spanning tree - grow tree from a start node.**

## At a Glance

- **Category:** [Network Flow & MST](Network-Flow-&-MST)
- **Problem Type:** MST
- **Complexity:** O((V + E) log V)

## Quick Example

```python
from solvor import prim

graph = {
    0: [(1, 4), (2, 3)],  # (neighbor, weight)
    1: [(0, 4), (2, 2), (3, 5)],
    2: [(0, 3), (1, 2), (3, 6)],
    3: [(1, 5), (2, 6)]
}

result = prim(graph, start=0)
print(result.solution)  # MST edges
```

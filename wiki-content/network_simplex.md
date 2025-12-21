# network_simplex

**Network simplex, specialized simplex for flow networks.**

## At a Glance

- **Category:** [Network Flow & MST](Network-Flow-&-MST)
- **Problem Type:** Min-cost flow (large scale)
- **Complexity:** O(V²E) typical

Simplex on a diet, exploits flow network structure.

## Quick Example

```python
from solvor import network_simplex

arcs = [
    (0, 1, 10, 2),  # (from, to, capacity, cost)
    (0, 2, 5, 3),
    (1, 3, 10, 1),
    (2, 3, 10, 1)
]
supplies = [15, 0, 0, -15]  # +produce, -consume, 0=passthrough

result = network_simplex(n_nodes=4, arcs=arcs, supplies=supplies)
```

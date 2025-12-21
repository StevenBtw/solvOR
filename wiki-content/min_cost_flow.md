# min_cost_flow

**Minimum cost flow, route flow at minimum cost.**

## Summary 

- **Category:** [Network Flow & MST](Network-Flow-&-MST)
- **Problem Type:** Min-cost routing
- **Complexity:** O(demand × Bellman-Ford)

## Quick Example

```python
from solvor import min_cost_flow

graph = {
    's': [('a', 10, 2), ('b', 10, 3)],  # (neighbor, capacity, cost)
    'a': [('t', 10, 1)],
    'b': [('t', 10, 1)],
    't': []
}

result = min_cost_flow(graph, source='s', sink='t', demand=10)
print(result.objective)  # Total cost
```

## See Also
- [`network_simplex`](network_simplex) - Faster for large networks

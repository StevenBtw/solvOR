# max_flow

**Maximum flow from source to sink - find the bottleneck.**

## At a Glance

- **Category:** [Network Flow & MST](Network-Flow-&-MST)
- **Problem Type:** Maximum throughput
- **Complexity:** O(VE²)
- **Guarantees:** Optimal max flow

## Quick Example

```python
from solvor import max_flow

graph = {
    's': [('a', 10, 0), ('b', 5, 0)],  # (neighbor, capacity, cost)
    'a': [('t', 5, 0), ('b', 15, 0)],
    'b': [('t', 10, 0)],
    't': []
}

result = max_flow(graph, source='s', sink='t')
print(result.objective)  # 15 (total flow)
```

## See Also
- [`min_cost_flow`](min_cost_flow)
- [Cookbook: Max Flow Network](Cookbook-Max-Flow-Network)

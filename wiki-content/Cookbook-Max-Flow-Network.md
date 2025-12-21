# Cookbook: Maximum Flow

Find the maximum flow through a network from source to sink.

## Complete Example

```python
from solvor import max_flow

# Network: s -> a,b -> t
# Edges with capacities
graph = {
    's': [('a', 10), ('b', 5)],
    'a': [('t', 5), ('b', 15)],
    'b': [('t', 10)],
    't': []
}

# Convert to (neighbor, capacity, cost) format
graph_with_costs = {
    node: [(neighbor, cap, 0) for neighbor, cap in edges]
    for node, edges in graph.items()
}

result = max_flow(graph_with_costs, source='s', sink='t')

print(f"Maximum flow: {result.objective}")
print(f"Flow on edges: {result.solution}")

# Find bottleneck edges (saturated)
for (u, v), flow in result.solution.items():
    capacity = next(cap for n, cap, _ in graph_with_costs[u] if n == v)
    if flow == capacity:
        print(f"Bottleneck: {u}->{v} (flow={flow}, capacity={capacity})")
```

## See Also
- [`max_flow`](max_flow)
- [Network Flow & MST](Network-Flow-&-MST)

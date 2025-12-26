# Network Flow

Algorithms for pushing stuff through networks.

## max_flow

Maximum flow from source to sink. "How much can I push through this network?"

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
print(result.solution)   # Flow on each edge
```

**Complexity:** O(VE²)
**Guarantees:** Optimal maximum flow

### Max-Flow Min-Cut

The maximum flow equals the minimum capacity cut. Finding bottlenecks comes free.

## min_cost_flow

Route a fixed amount of flow at minimum cost.

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

**Complexity:** O(demand × Bellman-Ford)

## network_simplex

Min-cost flow using network simplex. Much faster on large networks.

```python
from solvor import network_simplex

# Format: (from, to, capacity, cost)
arcs = [(0, 1, 10, 2), (0, 2, 5, 3), (1, 2, 15, 1)]
supplies = [10, 0, -10]  # positive = source, negative = sink

result = network_simplex(n_nodes=3, arcs=arcs, supplies=supplies)
```

**Complexity:** O(V²E) typical

## When to Use

| Algorithm | Use When |
|-----------|----------|
| `max_flow` | Finding maximum capacity, bottlenecks |
| `min_cost_flow` | Transportation with costs, small networks |
| `network_simplex` | Large networks with costs |

## See Also

- [MST](mst.md) - Minimum spanning trees
- [Cookbook: Max Flow](../../cookbook/max-flow.md)

# Minimum Spanning Tree

Connect all nodes at minimum total edge weight.

## kruskal

Edge-centric approach. Sorts edges, picks cheapest that don't create cycles.

```python
from solvor import kruskal

edges = [(0, 1, 4), (0, 2, 3), (1, 2, 2), (1, 3, 5), (2, 3, 6)]
result = kruskal(n_nodes=4, edges=edges)
print(result.solution)   # [(1, 2, 2), (0, 2, 3), (1, 3, 5)]
print(result.objective)  # 10 (total weight)
```

**Complexity:** O(E log E)
**Guarantees:** Optimal MST

### Minimum Spanning Forest

If the graph is disconnected:

```python
result = kruskal(n_nodes=4, edges=edges, allow_forest=True)
# Returns forest instead of failing
```

## prim

Node-centric approach. Grows tree from a start node.

```python
from solvor import prim

edges = [(0, 1, 4), (0, 2, 3), (1, 2, 2), (1, 3, 5), (2, 3, 6)]
result = prim(n_nodes=4, edges=edges, start=0)
print(result.solution)   # MST edges
print(result.objective)  # 10
```

**Complexity:** O((V + E) log V)
**Guarantees:** Optimal MST

## Comparison

| Algorithm | Approach | Best For |
|-----------|----------|----------|
| Kruskal | Sort all edges, pick cheapest | Sparse graphs |
| Prim | Grow from one node | Dense graphs, adjacency lists |

Both find the same optimal MST.

## Applications

- **Network design:** Laying cable to connect cities
- **Clustering:** Stop early at k-1 edges = k clusters
- **Approximate TSP:** MST gives a 2-approximation

## See Also

- [Network Flow](network-flow.md)

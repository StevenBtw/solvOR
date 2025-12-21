# Cookbook: Graph Coloring

Color graph nodes so no adjacent nodes share a color.

## Complete Example

```python
from solvor import Model

def color_graph(edges, n_nodes, n_colors):
    """
    Color graph with minimum number of colors.
    edges: list of (u, v) tuples
    """
    m = Model()
    
    # color[i] ∈ {0, 1, ..., n_colors-1}
    colors = [m.int_var(0, n_colors-1, f'color_{i}') for i in range(n_nodes)]
    
    # Adjacent nodes have different colors
    for u, v in edges:
        m.add(colors[u] != colors[v])
    
    result = m.solve()
    if result.solution:
        return [result.solution[f'color_{i}'] for i in range(n_nodes)]
    return None

# Triangle graph
edges = [(0, 1), (1, 2), (2, 0)]
coloring = color_graph(edges, n_nodes=3, n_colors=3)
print(f"Coloring: {coloring}")
```

## See Also
- [`Model`](Model-(CP-SAT))
- [Constraint Programming](Constraint-Programming)

# Graph Coloring

Color graph nodes so no adjacent nodes share a color.

## The Problem

Given a graph, assign colors to nodes such that no two adjacent nodes have the same color. Classic constraint satisfaction.

## Example

```python
from solvor import Model

def color_graph(edges, n_nodes, n_colors):
    """Color graph with given number of colors."""
    m = Model()

    # color[i] in {0, 1, ..., n_colors-1}
    colors = [m.int_var(0, n_colors-1, f'color_{i}') for i in range(n_nodes)]

    # Adjacent nodes have different colors
    for u, v in edges:
        m.add(colors[u] != colors[v])

    result = m.solve()
    if result.solution:
        return [result.solution[f'color_{i}'] for i in range(n_nodes)]
    return None

# Triangle graph (needs 3 colors)
edges = [(0, 1), (1, 2), (2, 0)]
coloring = color_graph(edges, n_nodes=3, n_colors=3)
print(f"Coloring: {coloring}")  # [0, 1, 2] or similar
```

## Chromatic Number

Find the minimum colors needed:

```python
def chromatic_number(edges, n_nodes):
    for k in range(1, n_nodes + 1):
        if color_graph(edges, n_nodes, k):
            return k
    return n_nodes
```

## See Also

- [CP-SAT Model](../algorithms/constraint-programming/cp-sat.md)
- [N-Queens](n-queens.md) - Similar constraint pattern

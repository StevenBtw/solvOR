# Graph Algorithms

Paths, flows, and trees. When your problem lives on a network of nodes and edges.

!!! tip "New in 0.5.3"
    **Centrality & Community Detection** — Six new algorithms for analyzing graph structure:
    `pagerank`, `louvain`, `articulation_points`, `bridges`, `kcore_decomposition`, `kcore`

## Categories

### Pathfinding

| Solver | Edge Weights | Use When |
|--------|--------------|----------|
| [`bfs`](pathfinding.md) | Unweighted | Simplest case, mazes |
| [`dfs`](pathfinding.md) | Unweighted | Connectivity, cycle detection |
| [`dijkstra`](shortest-paths.md) | Non-negative | Road networks |
| [`astar`](shortest-paths.md) | Non-negative | Goal-directed with heuristic |
| [`bellman_ford`](shortest-paths.md) | Any (negative OK) | Negative edges |
| [`floyd_warshall`](shortest-paths.md) | Any | All-pairs distances |

### Dependency Analysis

| Solver | Purpose | Use When |
|--------|---------|----------|
| [`topological_sort`](dependency-analysis.md) | Order by dependencies | Build systems, task scheduling |
| [`strongly_connected_components`](dependency-analysis.md) | Find cycles | Detecting circular deps |
| [`condense`](dependency-analysis.md) | Collapse cycles to DAG | Simplifying cyclic graphs |

### Centrality & Structure

| Solver | Purpose | Use When |
|--------|---------|----------|
| [`pagerank`](centrality.md) | Node importance | Ranking by influence |
| [`articulation_points`](centrality.md) | Cut vertices | Single points of failure |
| [`bridges`](centrality.md) | Cut edges | Critical connections |
| [`kcore_decomposition`](centrality.md) | Core numbers | Core vs periphery analysis |

### Community Detection

| Solver | Purpose | Use When |
|--------|---------|----------|
| [`louvain`](community-detection.md) | Clustering | Finding groups of related nodes |

### Network Flow

| Solver | Objective | Use When |
|--------|-----------|----------|
| [`max_flow`](network-flow.md) | Maximize throughput | Finding capacity, bottlenecks |
| [`min_cost_flow`](network-flow.md) | Minimize cost | Transportation |
| [`network_simplex`](network-flow.md) | Minimize cost | Large networks |

### Minimum Spanning Tree

| Solver | Approach | Use When |
|--------|----------|----------|
| [`kruskal`](mst.md) | Edge-centric | Sparse graphs |
| [`prim`](mst.md) | Node-centric | Dense graphs |

## Quick Example

```python
from solvor import dijkstra, max_flow, kruskal

# Shortest path
graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('C', 2), ('D', 5)],
    'C': [('D', 1)],
    'D': []
}
result = dijkstra('A', 'D', lambda n: graph[n])
print(result.solution)  # ['A', 'B', 'C', 'D']
print(result.objective)  # 4

# Maximum flow
flow_graph = {
    's': [('a', 10, 0), ('b', 5, 0)],
    'a': [('t', 5, 0), ('b', 15, 0)],
    'b': [('t', 10, 0)],
    't': []
}
result = max_flow(flow_graph, source='s', sink='t')
print(result.objective)  # 15

# Minimum spanning tree
edges = [(0, 1, 4), (0, 2, 3), (1, 2, 2), (1, 3, 5), (2, 3, 6)]
result = kruskal(n_nodes=4, edges=edges)
print(result.objective)  # 9
```

## See Also

- [Metaheuristics](../metaheuristics/index.md) - For traveling salesman
- [Cookbook: Shortest Path Grid](../../cookbook/shortest-path-grid.md) - A* grid example

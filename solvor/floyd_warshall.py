r"""
Floyd-Warshall for all-pairs shortest paths.

Computes shortest paths between ALL pairs of nodes at once. Simple three
nested loops, handles negative edges but not negative cycles.

    from solvor.floyd_warshall import floyd_warshall

    result = floyd_warshall(n_nodes, edges)
    dist = result.solution  # dist[i][j] = shortest path from i to j

How it works: dynamic programming over intermediate nodes. For each pair (i,j),
check if going through node k gives a shorter path: dist[i][j] = min(dist[i][j],
dist[i][k] + dist[k][j]). O(n³) time complexity.

Use this for:

- All-pairs shortest paths
- Network analysis and graph diameter
- Reachability matrices
- Precomputing route tables

Parameters:

    n_nodes: number of nodes in graph
    edges: list of (from, to, weight) tuples
    directed: if False, edges are bidirectional (default: True)

O(n³) works well for smaller graphs (<500 nodes). For larger graphs, run
dijkstra from each source , it's very parallelizable
and usually faster in practice.
For single-source: also dijkstra. For negative edges single-source: bellman_ford.
"""

from solvor.types import Result, Status
from solvor.utils import check_edge_nodes, check_positive

__all__ = ["floyd_warshall"]


def floyd_warshall(
    n_nodes: int,
    edges: list[tuple[int, int, float]],
    *,
    directed: bool = True,
) -> Result:
    check_positive(n_nodes, name="n_nodes")
    check_edge_nodes(edges, n_nodes)

    n = n_nodes
    dist = [[float("inf")] * n for _ in range(n)]

    for i in range(n):
        dist[i][i] = 0.0

    for u, v, w in edges:
        dist[u][v] = min(dist[u][v], w)
        if not directed:
            dist[v][u] = min(dist[v][u], w)

    iterations = 0

    for k in range(n):
        for i in range(n):
            for j in range(n):
                iterations += 1
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    for i in range(n):
        if dist[i][i] < 0:
            return Result(None, float("-inf"), iterations, 0, Status.UNBOUNDED)

    return Result(dist, 0, iterations, 0)

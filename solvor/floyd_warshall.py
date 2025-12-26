"""
Floyd-Warshall for all-pairs shortest paths.

Use this when you need distances between ALL pairs of nodes at once.
Network analysis, finding graph diameter, checking reachability matrices,
precomputing route tables where any node might query any other.

    from solvor.floyd_warshall import floyd_warshall

    result = floyd_warshall(n_nodes, edges)
    dist = result.solution  # dist[i][j] = shortest path from i to j

Handles negative edges, but negative cycles will mess up your results.
O(n³) time, so works well for smaller graphs. Past a few hundred
nodes it slows down significantly.

For single-source: dijkstra. For negative edges single-source: bellman_ford.
For large graphs: just run dijkstra from each source, it's very parallelizable
and usually faster in practice.
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

r"""
Bellman-Ford for shortest paths with negative edge weights.

Slower than Dijkstra but handles negative edges. The true edgelord. Only use
when you actually have negative weights.

    from solvor.bellman_ford import bellman_ford

    result = bellman_ford(start, edges, n_nodes)
    result = bellman_ford(start, edges, n_nodes, target=3)

How it works: relaxes every edge V-1 times. If any edge can still be relaxed
after that, a negative cycle exists. Simple but O(VE) complexity.

Use this for:

- Graphs with negative edge weights
- Detecting negative cycles
- Currency arbitrage detection
- When Dijkstra's non-negative assumption doesn't hold

Parameters:

    start: source node index
    edges: list of (from, to, weight) tuples
    n_nodes: number of nodes in graph
    target: optional destination node

Negative weights model situations where traversing an edge gives you something
back (e.g., trading routes where some legs earn profit).

Don't use this for: non-negative edges (use dijkstra), or all-pairs (floyd_warshall).
"""

from solvor.types import Result, Status
from solvor.utils import check_edge_nodes, check_in_range, check_positive

__all__ = ["bellman_ford"]


def bellman_ford(
    start: int,
    edges: list[tuple[int, int, float]],
    n_nodes: int,
    *,
    target: int | None = None,
) -> Result:
    """Shortest paths with negative weights, detects negative cycles."""
    check_positive(n_nodes, name="n_nodes")
    check_in_range(start, 0, n_nodes - 1, name="start")
    check_edge_nodes(edges, n_nodes)
    if target is not None:
        check_in_range(target, 0, n_nodes - 1, name="target")

    dist = [float("inf")] * n_nodes
    parent = [-1] * n_nodes
    dist[start] = 0.0
    iterations = 0

    for _ in range(n_nodes - 1):
        updated = False
        for u, v, w in edges:
            iterations += 1
            if dist[u] != float("inf") and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                parent[v] = u
                updated = True
        if not updated:
            break

    for u, v, w in edges:
        iterations += 1
        if dist[u] != float("inf") and dist[u] + w < dist[v]:
            return Result(None, float("-inf"), iterations, len(edges), Status.UNBOUNDED)

    if target is not None:
        if dist[target] == float("inf"):
            return Result(None, float("inf"), iterations, len(edges), Status.INFEASIBLE)
        path = _reconstruct_indexed(parent, target)
        return Result(path, dist[target], iterations, len(edges))

    distances = {i: dist[i] for i in range(n_nodes) if dist[i] < float("inf")}
    return Result(distances, 0, iterations, len(edges))


def _reconstruct_indexed(parent, target):
    path = [target]
    while parent[path[-1]] != -1:
        path.append(parent[path[-1]])
    path.reverse()
    return path

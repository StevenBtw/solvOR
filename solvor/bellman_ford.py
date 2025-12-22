"""
Bellman-Ford for shortest paths with negative edge weights.

Use this when edges can have negative weights. Slower than Dijkstra, checks
every edge V-1 times. The true edgelord. Only use when you actually have
negative edges.

    from solvor.bellman_ford import bellman_ford

    result = bellman_ford(start, edges, n_nodes)
    result = bellman_ford(start, edges, n_nodes, target=3)

Negative weights model situations where traversing an edge gives you something
back. Example: trading routes where some legs earn profit instead of costing.

    A --(-2)--> B --(3)--> C      Path A→B→C costs -2 + 3 = 1
    A --(5)--> C                  Path A→C costs 5
                                  Bellman-Ford finds A→B→C is cheaper

A negative cycle means you can reduce cost infinitely by looping. If one exists,
shortest paths are undefined and status is UNBOUNDED.

Returns shortest paths from start to all reachable nodes, or to a specific
target if provided.

Don't use this for: non-negative edges (use dijkstra), or all-pairs (floyd_warshall).
"""

from solvor.types import Result, Status

__all__ = ["bellman_ford"]


def bellman_ford(
    start: int,
    edges: list[tuple[int, int, float]],
    n_nodes: int,
    *,
    target: int | None = None,
) -> Result:
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

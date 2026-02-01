r"""
Dijkstra's algorithm for weighted shortest paths.

The classic algorithm for finding shortest paths in graphs with non-negative
edge weights. Named after Edsger Dijkstra, who designed it in 1956.

    from solvor.dijkstra import dijkstra

    result = dijkstra(start, goal, neighbors)
    result = dijkstra(start, lambda s: s.is_target, neighbors)

How it works: maintains a priority queue of (distance, node) pairs. Each
iteration pops the closest unvisited node, marks it visited, and relaxes
its edges. Guarantees shortest path when all edges are non-negative.

Use this for:

- Road networks and routing
- Any graph where "shortest" means minimum total weight
- When edge weights are non-negative
- As foundation for A* (add heuristic for goal-directed search)

Parameters:

    start: starting node
    goal: target node, or predicate function returning True at goal
    neighbors: function returning (neighbor, edge_cost) pairs

For negative edges use bellman_ford, Dijkstra's negativity was legendary,
just not in his algorithm. For unweighted graphs use bfs (simpler).
With a good distance estimate, use a_star.
"""

from collections.abc import Callable, Iterable
from heapq import heappop, heappush

from solvor._rust import with_rust_backend
from solvor.types import Result, Status
from solvor.utils import reconstruct_path

__all__ = ["dijkstra", "dijkstra_edges"]


def dijkstra[S](
    start: S,
    goal: S | Callable[[S], bool],
    neighbors: Callable[[S], Iterable[tuple[S, float]]],
    *,
    max_iter: int = 1_000_000,
    max_cost: float | None = None,
) -> Result:
    """Find shortest path in a weighted graph with non-negative edges."""
    is_goal = goal if callable(goal) else lambda s: s == goal

    g: dict[S, float] = {start: 0.0}
    parent: dict[S, S] = {}
    closed: set[S] = set()
    counter = 0
    heap: list[tuple[float, int, S]] = [(0.0, counter, start)]
    counter += 1
    iterations = 0
    evaluations = 1  # Counts nodes added to frontier (start + discovered neighbors)

    while heap and iterations < max_iter:
        cost, _, current = heappop(heap)

        if current in closed:
            continue

        iterations += 1
        closed.add(current)

        if is_goal(current):
            path = reconstruct_path(parent, current)
            return Result(path, g[current], iterations, evaluations)

        if max_cost is not None and cost > max_cost:
            continue

        for neighbor, edge_cost in neighbors(current):
            if neighbor in closed:
                continue

            tentative_g = g[current] + edge_cost

            if tentative_g < g.get(neighbor, float("inf")):
                g[neighbor] = tentative_g
                parent[neighbor] = current
                heappush(heap, (tentative_g, counter, neighbor))
                counter += 1
                evaluations += 1

    if iterations >= max_iter:
        return Result(None, float("inf"), iterations, evaluations, Status.MAX_ITER)
    return Result(None, float("inf"), iterations, evaluations, Status.INFEASIBLE)


@with_rust_backend
def dijkstra_edges(
    n_nodes: int,
    edges: list[tuple[int, int, float]],
    source: int,
    *,
    target: int | None = None,
) -> Result:
    """Edge-list Dijkstra for integer node graphs."""
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n_nodes)]
    for u, v, w in edges:
        adj[u].append((v, w))

    if target is not None:
        return dijkstra(source, target, lambda s: adj[s])

    # All-distances mode: minimal Dijkstra to collect distances
    dist: dict[int, float] = {source: 0.0}
    heap: list[tuple[float, int]] = [(0.0, source)]
    iterations = 0
    while heap:
        d, u = heappop(heap)
        if d > dist.get(u, float("inf")):
            continue
        iterations += 1
        for v, w in adj[u]:
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heappush(heap, (nd, v))
    return Result(dist, 0.0, iterations, 0)

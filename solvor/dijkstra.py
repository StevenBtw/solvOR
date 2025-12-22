"""
Dijkstra's algorithm for weighted shortest paths.

Use this when edges have non-negative weights and you need the optimal path.
Road networks, routing with distances, any graph where "shortest" means
"minimum total weight". This is the foundation for A*, which adds a heuristic
for faster goal-directed search.

    from solvor.dijkstra import dijkstra

    result = dijkstra(start, goal, neighbors)
    result = dijkstra(start, lambda s: s.is_target, neighbors)

The neighbors function returns (neighbor, edge_cost) pairs. Edge costs must
be non-negative, use bellman_ford for negative weights.

For negative edges use bellman_ford, Dijkstra's negativity was legendary,
just not in his algorithm. For unweighted graphs use bfs (simpler).
With a good distance estimate, use astar.
"""

from collections.abc import Callable, Iterable
from heapq import heappop, heappush

from solvor.types import Result, Status

__all__ = ["dijkstra"]


def dijkstra[S](
    start: S,
    goal: S | Callable[[S], bool],
    neighbors: Callable[[S], Iterable[tuple[S, float]]],
    *,
    max_iter: int = 1_000_000,
    max_cost: float | None = None,
) -> Result:
    is_goal = goal if callable(goal) else lambda s: s == goal

    g: dict[S, float] = {start: 0.0}
    parent: dict[S, S] = {}
    closed: set[S] = set()
    counter = 0
    heap: list[tuple[float, int, S]] = [(0.0, counter, start)]
    counter += 1
    iterations = 0
    evaluations = 1

    while heap and iterations < max_iter:
        cost, _, current = heappop(heap)

        if current in closed:
            continue

        iterations += 1
        closed.add(current)

        if is_goal(current):
            path = _reconstruct(parent, current)
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


def _reconstruct(parent, current):
    path = [current]
    while current in parent:
        current = parent[current]
        path.append(current)
    path.reverse()
    return path

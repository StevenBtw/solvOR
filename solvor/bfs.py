"""
Breadth-first and depth-first search for unweighted graphs.

BFS is the polite algorithm: it waits its turn, exploring level by level,
guaranteeing shortest paths. DFS is the curious one: it dives deep before
backtracking, useful when you just need any path or want to explore everything.
Both are O(V + E), the difference is queue vs stack.

Use this for: mazes, grid navigation, finding any path, flood fill, checking
if two nodes are connected. When goal is None, explores all reachable nodes.

    from solvor.bfs import bfs, dfs

    result = bfs(start, goal, neighbors)           # shortest path
    result = dfs(start, goal, neighbors)           # any path
    result = bfs(start, None, neighbors)           # all reachable nodes

For weighted graphs, use dijkstra. For heuristic search, use astar. For
negative edges, use bellman_ford.
"""

from collections import deque
from collections.abc import Callable, Iterable

from solvor.types import Result, Status

__all__ = ["bfs", "dfs"]


def bfs[S](
    start: S,
    goal: S | Callable[[S], bool] | None,
    neighbors: Callable[[S], Iterable[S]],
    *,
    max_iter: int = 1_000_000,
) -> Result:
    """Breadth-first search, guarantees shortest path in unweighted graphs."""
    is_goal = (lambda s: s == goal) if not callable(goal) and goal is not None else goal

    parent: dict[S, S] = {}
    visited: set[S] = {start}
    queue: deque[S] = deque([start])
    iterations = 0

    while queue and iterations < max_iter:
        current = queue.popleft()
        iterations += 1

        if is_goal and is_goal(current):
            path = _reconstruct(parent, current)
            return Result(path, len(path) - 1, iterations, len(visited), Status.OPTIMAL)

        for neighbor in neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)

    if is_goal:
        if iterations >= max_iter:
            return Result(None, float("inf"), iterations, len(visited), Status.MAX_ITER)
        return Result(None, float("inf"), iterations, len(visited), Status.INFEASIBLE)

    return Result(visited, len(visited), iterations, len(visited))


def dfs[S](
    start: S,
    goal: S | Callable[[S], bool] | None,
    neighbors: Callable[[S], Iterable[S]],
    *,
    max_iter: int = 1_000_000,
) -> Result:
    """Depth-first search, finds a path (not necessarily shortest)."""
    is_goal = (lambda s: s == goal) if not callable(goal) and goal is not None else goal

    parent: dict[S, S] = {}
    visited: set[S] = {start}
    stack: list[S] = [start]
    iterations = 0

    while stack and iterations < max_iter:
        current = stack.pop()
        iterations += 1

        if is_goal and is_goal(current):
            path = _reconstruct(parent, current)
            return Result(path, len(path) - 1, iterations, len(visited), Status.FEASIBLE)

        for neighbor in neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                stack.append(neighbor)

    if is_goal:
        if iterations >= max_iter:
            return Result(None, float("inf"), iterations, len(visited), Status.MAX_ITER)
        return Result(None, float("inf"), iterations, len(visited), Status.INFEASIBLE)

    return Result(visited, len(visited), iterations, len(visited))


def _reconstruct(parent, current):
    path = [current]
    while current in parent:
        current = parent[current]
        path.append(current)
    path.reverse()
    return path

r"""
A* search for goal-directed shortest paths with heuristics.

Dijkstra explores in all directions like ripples in a pond. A* knows where it's
going and prioritizes paths that look promising. Give it a heuristic (estimate
to goal) and it expands far fewer nodes.

    from solvor.a_star import astar, astar_grid

    result = astar(start, goal, neighbors, heuristic)
    result = astar_grid(maze, (0, 0), (9, 9), directions=8)

How it works: like Dijkstra but prioritizes by f(n) = g(n) + h(n), where g is
cost so far and h is estimated cost to goal. Heuristic must be admissible
(never overestimate) for optimal results. Weighted A* (weight > 1) trades
optimality for speed.

Use this for:

- Pathfinding with known goal location
- When you have a good distance heuristic
- Grid navigation and game AI
- When Dijkstra is too slow

Parameters:

    start: starting node
    goal: target node or predicate function
    neighbors: returns (neighbor, cost) pairs
    heuristic: estimates distance to goal (must not overestimate)
    weight: heuristic weight, >1 for faster suboptimal search

astar_grid provides built-in heuristics for 2D grids.

Don't use this for: unknown goal (use dijkstra), negative edges (use bellman_ford).
"""

from collections.abc import Callable, Iterable, Sequence
from heapq import heappop, heappush
from itertools import product
from math import sqrt
from typing import Literal

from solvor.types import Result, Status
from solvor.utils import reconstruct_path

__all__ = ["astar", "astar_grid"]

_SQRT2 = sqrt(2)
_SQRT2_MINUS_1 = _SQRT2 - 1

# 8-directions: N, S, E, W + diagonal NE, NW, SE, SW
_DIRS_8 = tuple((dx, dy) for dx, dy in product((-1, 0, 1), repeat=2) if (dx, dy) != (0, 0))

# 4-directions: N, S, E, W only
_DIRS_4 = tuple((dx, dy) for dx, dy in _DIRS_8 if dx == 0 or dy == 0)


def astar[S](
    start: S,
    goal: S | Callable[[S], bool],
    neighbors: Callable[[S], Iterable[tuple[S, float]]],
    heuristic: Callable[[S], float],
    *,
    weight: float = 1.0,
    max_iter: int = 1_000_000,
    max_cost: float | None = None,
) -> Result:
    """A* search with heuristic guidance, returns optimal path when weight=1."""
    is_goal = goal if callable(goal) else lambda s: s == goal

    g: dict[S, float] = {start: 0.0}
    parent: dict[S, S] = {}
    closed: set[S] = set()
    counter = 0
    f_start = weight * heuristic(start)
    heap: list[tuple[float, float, int, S]] = [(f_start, 0.0, counter, start)]
    counter += 1
    iterations = 0
    evaluations = 1

    while heap and iterations < max_iter:
        _, _, _, current = heappop(heap)

        if current in closed:
            continue

        iterations += 1
        closed.add(current)

        if is_goal(current):
            path = reconstruct_path(parent, current)
            status = Status.OPTIMAL if weight == 1.0 else Status.FEASIBLE
            return Result(path, g[current], iterations, evaluations, status)

        if max_cost is not None and g[current] > max_cost:
            continue

        for neighbor, edge_cost in neighbors(current):
            if neighbor in closed:
                continue

            tentative_g = g[current] + edge_cost

            if tentative_g < g.get(neighbor, float("inf")):
                g[neighbor] = tentative_g
                parent[neighbor] = current
                f_new = tentative_g + weight * heuristic(neighbor)
                heappush(heap, (f_new, -tentative_g, counter, neighbor))
                counter += 1
                evaluations += 1

    if iterations >= max_iter:
        return Result(None, float("inf"), iterations, evaluations, Status.MAX_ITER)
    return Result(None, float("inf"), iterations, evaluations, Status.INFEASIBLE)


def astar_grid(
    grid: Sequence[Sequence[int]],
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    directions: Literal[4, 8] = 4,
    heuristic: Literal["auto", "manhattan", "octile", "euclidean", "chebyshev"] = "auto",
    blocked: int | set[int] = 1,
    costs: dict[int, float] | None = None,
    weight: float = 1.0,
    max_iter: int = 1_000_000,
) -> Result:
    """A* for 2D grids with built-in heuristics and neighbor generation."""
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    blocked_set = {blocked} if isinstance(blocked, int) else set(blocked)
    cost_map = costs or {}
    dirs = _DIRS_8 if directions == 8 else _DIRS_4
    gr, gc = goal

    h_name = heuristic
    if h_name == "auto":
        h_name = "octile" if directions == 8 else "manhattan"

    match h_name:
        case "manhattan":

            def h(s):
                return abs(s[0] - gr) + abs(s[1] - gc)
        case "euclidean":

            def h(s):
                return ((s[0] - gr) ** 2 + (s[1] - gc) ** 2) ** 0.5
        case "octile":

            def h(s):
                dr, dc = abs(s[0] - gr), abs(s[1] - gc)
                return max(dr, dc) + _SQRT2_MINUS_1 * min(dr, dc)
        case "chebyshev":

            def h(s):
                return max(abs(s[0] - gr), abs(s[1] - gc))
        case _:

            def h(s):
                return abs(s[0] - gr) + abs(s[1] - gc)

    def neighbors(pos: tuple[int, int]):
        r, c = pos
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                cell = grid[nr][nc]
                if cell not in blocked_set:
                    base = cost_map.get(cell, 1.0)
                    if dr != 0 and dc != 0:
                        base *= _SQRT2
                    yield (nr, nc), base

    return astar(start, goal, neighbors, h, weight=weight, max_iter=max_iter)

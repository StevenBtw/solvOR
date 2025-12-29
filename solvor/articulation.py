r"""
Articulation points and bridges for finding critical connections in graphs.

Articulation points are nodes whose removal disconnects the graph. Bridges
are edges whose removal disconnects the graph. Finding these helps identify
single points of failure and critical dependencies.

    from solvor.articulation import articulation_points, bridges

    # Service dependencies
    deps = {
        "gateway": ["auth", "api"],
        "auth": ["gateway", "db"],
        "api": ["gateway", "db"],
        "db": ["auth", "api"],
    }

    result = articulation_points(deps.keys(), lambda n: deps.get(n, []))
    # result.solution = {"gateway"}  # removing gateway disconnects the graph

    result = bridges(deps.keys(), lambda n: deps.get(n, []))
    # result.solution = [("gateway", "auth"), ("gateway", "api")]

How it works: Uses Tarjan's algorithm with DFS, tracking discovery times and
low-links. A node is an articulation point if any subtree cannot reach back
above it. An edge is a bridge if the subtree cannot reach the edge's source.
Both run in O(V + E) time with a single DFS pass.

Use this for:

- Finding single points of failure in service architectures
- Identifying critical modules that many others depend on
- Network reliability analysis
- Detecting tightly vs loosely coupled subsystems

Parameters:

    nodes: iterable of all nodes in the graph
    neighbors: function returning iterable of adjacent nodes (undirected)

Works with any hashable node type. Treats graph as undirected.
"""

from collections.abc import Callable, Iterable

from solvor.types import Result

__all__ = ["articulation_points", "bridges"]


def articulation_points[S](
    nodes: Iterable[S],
    neighbors: Callable[[S], Iterable[S]],
) -> Result[set[S]]:
    """Find articulation points (cut vertices) in an undirected graph.

    Returns a set of nodes whose removal would disconnect the graph.
    """
    node_list = list(nodes)
    n = len(node_list)

    if n <= 1:
        return Result(set(), 0, 0, n)

    node_set = set(node_list)
    discovery: dict[S, int] = {}
    low: dict[S, int] = {}
    parent: dict[S, S | None] = {}
    ap: set[S] = set()
    time = [0]
    iterations = 0

    def dfs(v: S) -> None:
        nonlocal iterations
        iterations += 1

        children = 0
        discovery[v] = time[0]
        low[v] = time[0]
        time[0] += 1

        for w in neighbors(v):
            if w not in node_set:
                continue

            if w not in discovery:
                children += 1
                parent[w] = v
                dfs(w)
                low[v] = min(low[v], low[w])

                # v is an articulation point if:
                # 1. v is root and has 2+ children, OR
                # 2. v is not root and low[w] >= discovery[v]
                if parent[v] is None:
                    if children >= 2:
                        ap.add(v)
                elif low[w] >= discovery[v]:
                    ap.add(v)

            elif w != parent[v]:
                low[v] = min(low[v], discovery[w])

    # Handle disconnected components
    for v in node_list:
        if v not in discovery:
            parent[v] = None
            dfs(v)

    return Result(ap, len(ap), iterations, n)


def bridges[S](
    nodes: Iterable[S],
    neighbors: Callable[[S], Iterable[S]],
) -> Result[list[tuple[S, S]]]:
    """Find bridges (cut edges) in an undirected graph.

    Returns a list of edges whose removal would disconnect the graph.
    Each edge is a tuple (u, v) with u < v for consistent ordering.
    """
    node_list = list(nodes)
    n = len(node_list)

    if n <= 1:
        return Result([], 0, 0, n)

    node_set = set(node_list)
    discovery: dict[S, int] = {}
    low: dict[S, int] = {}
    parent: dict[S, S | None] = {}
    bridge_list: list[tuple[S, S]] = []
    time = [0]
    iterations = 0

    def dfs(v: S) -> None:
        nonlocal iterations
        iterations += 1

        discovery[v] = time[0]
        low[v] = time[0]
        time[0] += 1

        for w in neighbors(v):
            if w not in node_set:
                continue

            if w not in discovery:
                parent[w] = v
                dfs(w)
                low[v] = min(low[v], low[w])

                # Edge (v, w) is a bridge if low[w] > discovery[v]
                if low[w] > discovery[v]:
                    # Canonical ordering for consistent results
                    edge = (v, w) if v < w else (w, v)  # type: ignore[operator]
                    bridge_list.append(edge)

            elif w != parent[v]:
                low[v] = min(low[v], discovery[w])

    # Handle disconnected components
    for v in node_list:
        if v not in discovery:
            parent[v] = None
            dfs(v)

    return Result(bridge_list, len(bridge_list), iterations, n)

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
    children: dict[S, int] = {}
    ap: set[S] = set()
    time = 0
    iterations = 0

    # Iterative DFS: an explicit stack of (node, neighbor iterator) frames, so the
    # depth is not bounded by Python's recursion limit
    for root in node_list:
        if root in discovery:
            continue
        parent[root] = None
        iterations += 1
        discovery[root] = low[root] = time
        time += 1
        children[root] = 0
        stack = [(root, iter(neighbors(root)))]

        while stack:
            v, it = stack[-1]
            for w in it:
                if w not in node_set:
                    continue

                if w not in discovery:
                    children[v] += 1
                    parent[w] = v
                    iterations += 1
                    discovery[w] = low[w] = time
                    time += 1
                    children[w] = 0
                    stack.append((w, iter(neighbors(w))))
                    break

                if w != parent[v]:
                    low[v] = min(low[v], discovery[w])
            else:
                stack.pop()
                if stack:
                    u = stack[-1][0]
                    low[u] = min(low[u], low[v])

                    # u is an articulation point if:
                    # 1. u is root and has 2+ children, OR
                    # 2. u is not root and low[v] >= discovery[u]
                    if parent[u] is None:
                        if children[u] >= 2:
                            ap.add(u)
                    elif low[v] >= discovery[u]:
                        ap.add(u)

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
    time = 0
    iterations = 0

    # Iterative DFS, same frames as in articulation_points
    for root in node_list:
        if root in discovery:
            continue
        parent[root] = None
        iterations += 1
        discovery[root] = low[root] = time
        time += 1
        stack = [(root, iter(neighbors(root)))]

        while stack:
            v, it = stack[-1]
            for w in it:
                if w not in node_set:
                    continue

                if w not in discovery:
                    parent[w] = v
                    iterations += 1
                    discovery[w] = low[w] = time
                    time += 1
                    stack.append((w, iter(neighbors(w))))
                    break

                if w != parent[v]:
                    low[v] = min(low[v], discovery[w])
            else:
                stack.pop()
                if stack:
                    u = stack[-1][0]
                    low[u] = min(low[u], low[v])

                    # Edge (u, v) is a bridge if low[v] > discovery[u]
                    if low[v] > discovery[u]:
                        # Canonical ordering for consistent results
                        edge = (u, v) if u < v else (v, u)  # type: ignore[operator]  # ty: ignore[unsupported-operator]
                        bridge_list.append(edge)

    return Result(bridge_list, len(bridge_list), iterations, n)

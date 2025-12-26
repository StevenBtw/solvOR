"""
Minimum Spanning Tree - connect everything at minimum cost.

One of those rare greedy algorithms that's actually optimal. Kruskal's is
beautifully simple: sort edges by weight, pick the smallest that doesn't
create a cycle. Uses Union-Find under the hood for near O(1) cycle detection.
Prim's grows a tree from a start node, Dijkstra-style, but with a priority queu.

    from solvor.mst import kruskal, prim

    result = kruskal(n_nodes, edges)
    result = prim(graph, start=0)

    0 --4-- 1
    |     / |        kruskal picks: 1-2 (2), 0-2 (3), 0-1 (4)
    3   2   5        total weight: 9
    | /     |        skips 1-3 (5) - would create cycle
    2 --6-- 3

Classic uses: network cabling, clustering (stop early = k clusters), circuit
layout. Both algorithms return the same MST.

Don't use this for: directed graphs, or shortest paths (that's dijkstra).
"""

from collections.abc import Iterable
from heapq import heappop, heappush

from solvor.types import Result, Status
from solvor.utils import UnionFind, check_edge_nodes, check_positive

__all__ = ["kruskal", "prim"]


def kruskal(
    n_nodes: int,
    edges: list[tuple[int, int, float]],
    *,
    allow_forest: bool = False,
) -> Result:
    """
    Find minimum spanning tree using Kruskal's algorithm.

    Args:
        n_nodes: Number of nodes in the graph
        edges: List of (u, v, weight) tuples
        allow_forest: If True, return partial result for disconnected graphs
                      (Status.FEASIBLE with forest edges). If False, return
                      Status.INFEASIBLE for disconnected graphs.

    Returns:
        Result with MST edges as solution and total weight as objective.
        If allow_forest=True and graph is disconnected, returns minimum
        spanning forest (MST for each connected component).
    """
    check_positive(n_nodes, name="n_nodes")
    check_edge_nodes(edges, n_nodes)

    uf = UnionFind(n_nodes)
    sorted_edges = sorted(edges, key=lambda e: e[2])
    mst_edges = []
    total_weight = 0.0
    iterations = 0

    for u, v, w in sorted_edges:
        iterations += 1
        if uf.union(u, v):
            mst_edges.append((u, v, w))
            total_weight += w
            if len(mst_edges) == n_nodes - 1:
                break

    if len(mst_edges) < n_nodes - 1:
        if allow_forest:
            return Result(mst_edges, total_weight, iterations, len(edges), Status.FEASIBLE)
        return Result(None, float("inf"), iterations, len(edges), Status.INFEASIBLE)

    return Result(mst_edges, total_weight, iterations, len(edges))


def prim[Node](
    graph: dict[Node, Iterable[tuple[Node, float]]],
    *,
    start: Node | None = None,
) -> Result:
    if not graph:
        return Result([], 0.0, 0, 0)

    nodes = set(graph.keys())
    for neighbors in graph.values():
        for neighbor, _ in neighbors:
            nodes.add(neighbor)

    if start is None:
        start = next(iter(graph.keys()))

    in_mst: set[Node] = {start}
    mst_edges: list[tuple[Node, Node, float]] = []
    total_weight = 0.0
    counter = 0
    heap: list[tuple[float, int, Node, Node]] = []

    for neighbor, weight in graph.get(start, []):
        heappush(heap, (weight, counter, start, neighbor))
        counter += 1

    iterations = 0
    evaluations = counter

    while heap and len(in_mst) < len(nodes):
        weight, _, u, v = heappop(heap)
        iterations += 1

        if v in in_mst:
            continue

        in_mst.add(v)
        mst_edges.append((u, v, weight))
        total_weight += weight

        for neighbor, edge_weight in graph.get(v, []):
            if neighbor not in in_mst:
                heappush(heap, (edge_weight, counter, v, neighbor))
                counter += 1
                evaluations += 1

    if len(in_mst) < len(nodes):
        return Result(None, float("inf"), iterations, evaluations, Status.INFEASIBLE)

    return Result(mst_edges, total_weight, iterations, evaluations)

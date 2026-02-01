"""Rust adapter functions - all Rust-specific logic lives here.

This module isolates all Rust-specific preprocessing and result conversion,
keeping the main algorithm files clean and readable.

Each adapter is registered with @rust_adapter("function_name") and handles:
- Any parameter preprocessing (e.g., expanding undirected edges)
- Calling the Rust function
- Converting Rust results to Python Result objects
"""

from __future__ import annotations

from solvor._rust import get_rust_module, rust_adapter
from solvor.types import Result, Status


@rust_adapter("floyd_warshall")
def _floyd_warshall_rust(
    n_nodes: int,
    edges: list[tuple[int, int, float]],
    *,
    directed: bool = True,
) -> Result:
    """Rust adapter for Floyd-Warshall algorithm."""
    rust = get_rust_module()

    # For undirected graphs, expand to bidirectional edges
    if not directed:
        edge_set: set[tuple[int, int]] = set()
        expanded: list[tuple[int, int, float]] = []
        for u, v, w in edges:
            if (u, v) not in edge_set:
                expanded.append((u, v, w))
                edge_set.add((u, v))
            if (v, u) not in edge_set:
                expanded.append((v, u, w))
                edge_set.add((v, u))
        edges = expanded

    result = rust.floyd_warshall(n_nodes, edges)

    # Convert Rust result to Python Result
    if result["has_negative_cycle"]:
        return Result(None, float("-inf"), result["iterations"], 0, Status.UNBOUNDED)

    return Result(result["distances"], 0, result["iterations"], 0)


@rust_adapter("bellman_ford")
def _bellman_ford_rust(
    start: int,
    edges: list[tuple[int, int, float]],
    n_nodes: int,
    *,
    target: int | None = None,
) -> Result:
    """Rust adapter for Bellman-Ford algorithm."""
    rust = get_rust_module()

    result = rust.bellman_ford(n_nodes, edges, start)

    if result["has_negative_cycle"]:
        return Result(None, float("-inf"), result["iterations"], 0, Status.UNBOUNDED)

    # Reconstruct path if target specified
    if target is not None:
        if result["distances"][target] == float("inf"):
            return Result(None, float("inf"), result["iterations"], 0, Status.INFEASIBLE)

        # Reconstruct path from predecessors
        path = []
        current = target
        while current != -1:
            path.append(current)
            current = result["predecessors"][current]
        path.reverse()

        return Result(path, result["distances"][target], result["iterations"], 0)

    # Return distances dict
    distances = {i: d for i, d in enumerate(result["distances"]) if d != float("inf")}
    return Result(distances, 0.0, result["iterations"], 0)


@rust_adapter("kruskal")
def _kruskal_rust(
    n_nodes: int,
    edges: list[tuple[int, int, float]],
    *,
    allow_forest: bool = False,
) -> Result:
    """Rust adapter for Kruskal's MST algorithm."""
    rust = get_rust_module()

    result = rust.kruskal(n_nodes, edges)

    mst_edges = [(u, v, w) for u, v, w in result["mst_edges"]]

    if not result["is_connected"]:
        if allow_forest:
            return Result(mst_edges, result["total_weight"], result["iterations"], len(edges), Status.FEASIBLE)
        return Result(None, float("inf"), result["iterations"], len(edges), Status.INFEASIBLE)

    return Result(mst_edges, result["total_weight"], result["iterations"], len(edges))


@rust_adapter("dijkstra_edges")
def _dijkstra_edges_rust(
    n_nodes: int,
    edges: list[tuple[int, int, float]],
    source: int,
    *,
    target: int | None = None,
) -> Result:
    """Rust adapter for edge-list Dijkstra."""
    rust = get_rust_module()

    result = rust.dijkstra(n_nodes, edges, source, target)

    if target is not None:
        if result["target_reached"]:
            return Result(list(result["path"]), result["target_distance"], result["iterations"], 0)
        return Result(None, float("inf"), result["iterations"], 0, Status.INFEASIBLE)

    # No target - return distances dict
    distances = {i: d for i, d in enumerate(result["distances"]) if d != float("inf")}
    return Result(distances, 0.0, result["iterations"], 0)


@rust_adapter("bfs_edges")
def _bfs_edges_rust(
    n_nodes: int,
    edges: list[tuple[int, int]],
    source: int,
    *,
    target: int | None = None,
) -> Result:
    """Rust adapter for edge-list BFS."""
    rust = get_rust_module()

    result = rust.bfs(n_nodes, edges, source, target)

    if target is not None:
        if result["target_reached"]:
            return Result(list(result["path"]), len(result["path"]) - 1, result["iterations"], 0)
        return Result(None, float("inf"), result["iterations"], 0, Status.INFEASIBLE)

    return Result(list(result["visited_order"]), 0, result["iterations"], 0)


@rust_adapter("dfs_edges")
def _dfs_edges_rust(
    n_nodes: int,
    edges: list[tuple[int, int]],
    source: int,
    *,
    target: int | None = None,
) -> Result:
    """Rust adapter for edge-list DFS."""
    rust = get_rust_module()

    result = rust.dfs(n_nodes, edges, source, target)

    if target is not None:
        if result["target_reached"]:
            return Result(list(result["path"]), len(result["path"]) - 1, result["iterations"], 0)
        return Result(None, float("inf"), result["iterations"], 0, Status.INFEASIBLE)

    return Result(list(result["visited_order"]), 0, result["iterations"], 0)


@rust_adapter("pagerank_edges")
def _pagerank_edges_rust(
    n_nodes: int,
    edges: list[tuple[int, int]],
    *,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Result:
    """Rust adapter for edge-list PageRank."""
    rust = get_rust_module()

    result = rust.pagerank(n_nodes, edges, damping, max_iter, tol)

    scores = {i: s for i, s in enumerate(result["scores"])}
    status = Status.OPTIMAL if result["converged"] else Status.MAX_ITER

    return Result(scores, 0.0, result["iterations"], 0, status)


@rust_adapter("strongly_connected_components_edges")
def _scc_edges_rust(
    n_nodes: int,
    edges: list[tuple[int, int]],
) -> Result:
    """Rust adapter for edge-list SCC."""
    rust = get_rust_module()

    result = rust.strongly_connected_components(n_nodes, edges)

    components = [list(comp) for comp in result["components"]]
    return Result(components, result["n_components"], 0, 0)


@rust_adapter("topological_sort_edges")
def _topo_edges_rust(
    n_nodes: int,
    edges: list[tuple[int, int]],
) -> Result:
    """Rust adapter for edge-list topological sort."""
    rust = get_rust_module()

    result = rust.topological_sort(n_nodes, edges)

    if result["is_acyclic"]:
        return Result(list(result["order"]), 0, result["iterations"], 0)
    return Result(None, 0, result["iterations"], 0, Status.INFEASIBLE)

r"""
PageRank algorithm for measuring node importance in directed graphs.

PageRank assigns importance scores to nodes based on the link structure of the
graph. Nodes that are linked to by many important nodes get higher scores.
Originally developed for ranking web pages, but useful for any directed graph.

    from solvor.pagerank import pagerank

    # Web pages linking to each other
    links = {
        "home": ["about", "products"],
        "about": ["home", "contact"],
        "products": ["home"],
        "contact": ["home"],
    }

    result = pagerank(links.keys(), lambda n: links.get(n, []))
    # result.solution = {"home": 0.38, "products": 0.22, ...}

How it works: Iteratively updates each node's score as a weighted sum of the
scores of nodes linking to it, plus a damping factor that allows random jumps.
Converges when no score changes by more than tol between iterations. Equivalent
to finding the dominant eigenvector of the stochastic transition matrix.

Use this for:

- Finding important nodes (key modules, critical components)
- Ranking pages, documents, or entities by influence
- Identifying central hubs in dependency graphs
- Prioritizing code review focus areas

Parameters:

    nodes: iterable of all nodes in the graph
    neighbors: function returning iterable of outgoing edges (successors)
    damping: probability of following links vs random jump (default 0.85)
    max_iter: maximum iterations (default 100)
    tol: convergence tolerance on the largest score change (default 1e-6)

Two variants available:

    pagerank() - Callback-based, works with any node type
    pagerank_edges() - Edge-list, integer nodes 0..n-1

Both use the Rust backend when it is installed (pagerank() maps nodes to
integers first); pass backend="python" for the pure Python implementation.
The backends agree to floating point rounding.

Works with any hashable node type. Results are deterministic when nodes are
mutually comparable: nodes are processed in sorted order. For incoming edges
(predecessors), swap the edge direction in your neighbors function.
"""

from collections.abc import Callable, Iterable
from typing import Literal

from solvor.rust import with_rust_backend
from solvor.types import Result, Status
from solvor.utils.helpers import canonical_order

__all__ = ["pagerank", "pagerank_edges"]


def pagerank[S](
    nodes: Iterable[S],
    neighbors: Callable[[S], Iterable[S]],
    *,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
    backend: Literal["auto", "rust", "python"] | None = None,
) -> Result[dict[S, float]]:
    """Compute PageRank scores for nodes in a directed graph.

    Returns a dict mapping each node to its importance score (sums to 1.0).
    Higher scores indicate more important/central nodes.
    """
    node_list = canonical_order(nodes)
    index = {v: i for i, v in enumerate(node_list)}
    edges: list[tuple[int, int]] = []
    for i, v in enumerate(node_list):
        for w in neighbors(v):
            j = index.get(w)
            if j is not None:
                edges.append((i, j))

    result = pagerank_edges(len(node_list), edges, damping=damping, max_iter=max_iter, tol=tol, backend=backend)
    scores = {node_list[i]: score for i, score in result.solution.items()}
    return Result(scores, result.objective, result.iterations, result.evaluations, result.status)


@with_rust_backend
def pagerank_edges(
    n_nodes: int,
    edges: list[tuple[int, int]],
    *,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Result:
    """Edge-list PageRank for integer node graphs. Edges with an endpoint outside 0..n_nodes-1 are ignored."""
    check_pagerank_params(n_nodes, damping, max_iter, tol)
    if n_nodes == 0:
        return Result({}, 0.0, 0, 0)

    incoming: list[list[int]] = [[] for _ in range(n_nodes)]
    out_degree = [0] * n_nodes
    for u, v in edges:
        if 0 <= u < n_nodes and 0 <= v < n_nodes:
            incoming[v].append(u)
            out_degree[u] += 1

    dangling = [u for u in range(n_nodes) if out_degree[u] == 0]
    inv_out = [1.0 / d if d else 0.0 for d in out_degree]
    scores = [1.0 / n_nodes] * n_nodes
    base = (1.0 - damping) / n_nodes
    max_diff = 0.0

    for iteration in range(1, max_iter + 1):
        share = [s * k for s, k in zip(scores, inv_out)]
        # Dangling nodes (no outgoing edges) spread their rank over all nodes
        dangling_contrib = damping * sum(scores[u] for u in dangling) / n_nodes
        new_scores = [base + damping * sum([share[u] for u in inc]) + dangling_contrib for inc in incoming]
        max_diff = max(abs(a - b) for a, b in zip(new_scores, scores))
        scores = new_scores
        if max_diff < tol:
            return Result(dict(enumerate(scores)), max_diff, iteration, n_nodes)

    return Result(dict(enumerate(scores)), max_diff, max_iter, n_nodes, Status.MAX_ITER)


def check_pagerank_params(n_nodes: int, damping: float, max_iter: int, tol: float) -> None:
    """Validate pagerank_edges arguments. Both backends call this, so errors are identical."""
    if n_nodes < 0:
        raise ValueError("n_nodes must be non-negative")
    if not 0.0 <= damping < 1.0:
        raise ValueError(f"damping {damping} must be in [0, 1)")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if tol <= 0.0:
        raise ValueError("tol must be positive")

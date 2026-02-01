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
Converges when scores stop changing significantly. Equivalent to finding the
dominant eigenvector of the stochastic transition matrix.

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
    tol: convergence tolerance (default 1e-6)

Two variants available:

    pagerank() - Callback-based, works with any node type (pure Python)
    pagerank_edges() - Edge-list, integer nodes 0..n-1, has Rust backend (10-15x faster)

Use pagerank_edges(backend="python") for the pure Python implementation.

Works with any hashable node type. For incoming edges (predecessors), swap
the edge direction in your neighbors function.
"""

from collections.abc import Callable, Iterable

from solvor._rust import with_rust_backend
from solvor.types import Result, Status

__all__ = ["pagerank", "pagerank_edges"]


def pagerank[S](
    nodes: Iterable[S],
    neighbors: Callable[[S], Iterable[S]],
    *,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Result[dict[S, float]]:
    """Compute PageRank scores for nodes in a directed graph.

    Returns a dict mapping each node to its importance score (sums to 1.0).
    Higher scores indicate more important/central nodes.
    """
    node_list = list(nodes)
    n = len(node_list)

    if n == 0:
        return Result({}, 0.0, 0, 0)

    # Build reverse adjacency (who links TO each node)
    node_set = set(node_list)
    incoming: dict[S, list[S]] = {v: [] for v in node_list}
    outgoing_count: dict[S, int] = {v: 0 for v in node_list}

    for v in node_list:
        for w in neighbors(v):
            if w in node_set:
                incoming[w].append(v)
                outgoing_count[v] += 1

    # Initialize scores uniformly
    scores: dict[S, float] = {v: 1.0 / n for v in node_list}
    base_score = (1.0 - damping) / n

    iterations = 0
    for iterations in range(1, max_iter + 1):
        new_scores: dict[S, float] = {}
        max_diff = 0.0

        # Handle dangling nodes (no outgoing edges) - distribute their rank
        dangling_sum = sum(scores[v] for v in node_list if outgoing_count[v] == 0)
        dangling_contrib = damping * dangling_sum / n

        for v in node_list:
            # Sum contributions from nodes linking to v
            rank_sum = sum(scores[u] / outgoing_count[u] for u in incoming[v])
            new_scores[v] = base_score + damping * rank_sum + dangling_contrib
            max_diff = max(max_diff, abs(new_scores[v] - scores[v]))

        scores = new_scores

        if max_diff < tol:
            return Result(scores, max_diff, iterations, n)

    return Result(scores, max_diff, iterations, n, Status.MAX_ITER)


@with_rust_backend
def pagerank_edges(
    n_nodes: int,
    edges: list[tuple[int, int]],
    *,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Result:
    """Edge-list PageRank for integer node graphs."""
    adj: list[list[int]] = [[] for _ in range(n_nodes)]
    for u, v in edges:
        adj[u].append(v)

    return pagerank(range(n_nodes), lambda s: adj[s], damping=damping, max_iter=max_iter, tol=tol)

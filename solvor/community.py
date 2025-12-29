r"""
Community detection for finding clusters in graphs.

Community detection identifies groups of nodes that are more densely connected
to each other than to the rest of the graph. Useful for discovering natural
groupings, modules, or clusters in network data.

    from solvor.community import louvain

    # Social network connections
    friends = {
        "alice": ["bob", "carol"],
        "bob": ["alice", "carol"],
        "carol": ["alice", "bob"],
        "dave": ["eve", "frank"],
        "eve": ["dave", "frank"],
        "frank": ["dave", "eve"],
    }

    result = louvain(friends.keys(), lambda n: friends.get(n, []))
    # result.solution = [{"alice", "bob", "carol"}, {"dave", "eve", "frank"}]

How it works: Louvain algorithm optimizes modularity in two phases. Phase 1:
each node moves to the community that gives the largest modularity gain.
Phase 2: build a new graph where communities become nodes. Repeat until no
improvement. Modularity measures the fraction of edges within communities
minus the expected fraction if edges were random.

Use this for:

- Grouping related code modules (files that import each other)
- Finding clusters of tightly-coupled components
- Discovering subsystems in dependency graphs
- Partitioning for parallel processing

Parameters:

    nodes: iterable of all nodes in the graph
    neighbors: function returning iterable of adjacent nodes (undirected)
    resolution: modularity resolution parameter (default 1.0, higher = smaller communities)

Works with any hashable node type. Treats graph as undirected (edges in both
directions are counted once).
"""

from collections import defaultdict
from collections.abc import Callable, Iterable

from solvor.types import Result

__all__ = ["louvain"]


def louvain[S](
    nodes: Iterable[S],
    neighbors: Callable[[S], Iterable[S]],
    *,
    resolution: float = 1.0,
) -> Result[list[set[S]]]:
    """Find communities using the Louvain algorithm.

    Returns a list of sets, each containing nodes in the same community.
    Optimizes modularity to find densely connected groups.
    """
    node_list = list(nodes)
    n = len(node_list)

    if n == 0:
        return Result([], 0.0, 0, 0)

    if n == 1:
        return Result([{node_list[0]}], 0.0, 0, 1)

    # Build adjacency with edge weights (count = 1 for each edge)
    node_set = set(node_list)
    adj: dict[S, dict[S, float]] = {v: {} for v in node_list}
    degree: dict[S, float] = {v: 0.0 for v in node_list}

    for v in node_list:
        for w in neighbors(v):
            if w in node_set and w != v:
                # Undirected: count each edge once in total weight
                if w not in adj[v]:
                    adj[v][w] = 1.0
                    adj[w][v] = 1.0
                    degree[v] += 1.0
                    degree[w] += 1.0

    total_weight = sum(degree.values()) / 2.0

    if total_weight == 0:
        # No edges - each node is its own community
        return Result([{v} for v in node_list], 0.0, 0, n)

    # Initialize: each node in its own community
    node_to_comm: dict[S, int] = {v: i for i, v in enumerate(node_list)}
    comm_nodes: dict[int, set[S]] = {i: {v} for i, v in enumerate(node_list)}
    comm_degree: dict[int, float] = {i: degree[v] for i, v in enumerate(node_list)}

    iterations = 0
    improved = True

    while improved:
        improved = False
        iterations += 1

        for v in node_list:
            current_comm = node_to_comm[v]
            v_degree = degree[v]

            # Calculate edges to each neighboring community
            comm_edges: defaultdict[int, float] = defaultdict(float)
            for w, weight in adj[v].items():
                comm_edges[node_to_comm[w]] += weight

            # Remove v from current community temporarily
            comm_nodes[current_comm].remove(v)
            comm_degree[current_comm] -= v_degree
            edges_to_current = comm_edges.get(current_comm, 0.0)

            # Find best community
            best_comm = current_comm
            best_gain = 0.0

            for comm, edges_to_comm in comm_edges.items():
                # Modularity gain from moving v to comm
                sigma_c = comm_degree[comm]
                gain = edges_to_comm - resolution * v_degree * sigma_c / (2 * total_weight)

                if gain > best_gain:
                    best_gain = gain
                    best_comm = comm

            # Also consider staying in current (now empty or not) community
            if current_comm != best_comm:
                sigma_curr = comm_degree[current_comm]
                stay_gain = edges_to_current - resolution * v_degree * sigma_curr / (2 * total_weight)
                if stay_gain >= best_gain:
                    best_comm = current_comm
                    best_gain = stay_gain

            # Move v to best community
            node_to_comm[v] = best_comm
            comm_nodes[best_comm].add(v)
            comm_degree[best_comm] += v_degree

            if best_comm != current_comm:
                improved = True

    # Collect non-empty communities
    communities = [c for c in comm_nodes.values() if c]

    # Calculate final modularity
    modularity = 0.0
    for comm in communities:
        edges_within = sum(
            adj[v].get(w, 0.0)
            for v in comm
            for w in comm
            if v < w
        )
        comm_deg = sum(degree[v] for v in comm)
        modularity += edges_within / total_weight - resolution * (comm_deg / (2 * total_weight)) ** 2

    return Result(communities, modularity, iterations, n)

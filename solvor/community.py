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
each node moves to the community that gives the largest modularity gain, until
no move improves it. Phase 2: build a new graph where communities become nodes
(edge weights summed, internal edges kept as self-loops). Repeat both phases
on the new graph until no node moves. Modularity measures the fraction of edges
within communities minus the expected fraction if edges were random.

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
directions are counted once). Results are deterministic when nodes are mutually
comparable (ints, strs, tuples of those): nodes and neighbors are processed in
sorted order, so set inputs give the same communities under any PYTHONHASHSEED.
Otherwise nodes are processed in the order given.
"""

from collections.abc import Callable, Iterable

from solvor.types import Result
from solvor.utils.helpers import canonical_order

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
    node_list = canonical_order(nodes)
    n = len(node_list)

    if n == 0:
        return Result([], 0.0, 0, 0)

    if n == 1:
        return Result([{node_list[0]}], 0.0, 0, 1)

    # Level-0 graph on integer ids, weight 1 per undirected edge
    index = {v: i for i, v in enumerate(node_list)}
    adj: list[dict[int, float]] = [{} for _ in range(n)]
    for i, v in enumerate(node_list):
        for w in canonical_order(neighbors(v)):
            j = index.get(w)
            if j is not None and j != i and j not in adj[i]:
                adj[i][j] = 1.0
                adj[j][i] = 1.0

    total_weight = sum(len(nbrs) for nbrs in adj) / 2.0

    if total_weight == 0:
        # No edges - each node is its own community
        return Result([{v} for v in node_list], 0.0, 0, n)

    level_adj, level_loops = adj, [0.0] * n
    membership = list(range(n))  # original node -> node of the current level
    iterations = 0

    while True:
        degree = [sum(nbrs.values()) + 2.0 * loop for nbrs, loop in zip(level_adj, level_loops)]
        comm, passes, moved = _local_moving(level_adj, degree, 2.0 * total_weight, resolution)
        iterations += passes
        if not moved:
            break
        next_adj, next_loops, to_next = _aggregate(level_adj, level_loops, comm)
        membership = [to_next[c] for c in membership]
        if len(next_adj) == len(level_adj):
            break
        level_adj, level_loops = next_adj, next_loops

    groups: dict[int, set[S]] = {}
    for v, c in zip(node_list, membership):
        groups.setdefault(c, set()).add(v)
    communities = list(groups.values())

    # Modularity on the original graph, one pass over the edges
    within: dict[int, float] = {}
    comm_degree: dict[int, float] = {}
    for i, nbrs in enumerate(adj):
        ci = membership[i]
        comm_degree[ci] = comm_degree.get(ci, 0.0) + len(nbrs)
        for j in nbrs:
            if membership[j] == ci:
                within[ci] = within.get(ci, 0.0) + 0.5  # each internal edge is seen from both ends
    modularity = 0.0
    for c in groups:
        share = comm_degree[c] / (2 * total_weight)  # squared as share * share: ** 2 goes through libm pow
        modularity += within.get(c, 0.0) / total_weight - resolution * share * share

    return Result(communities, modularity, iterations, n)


def _local_moving(
    adj: list[dict[int, float]], degree: list[float], two_m: float, resolution: float
) -> tuple[list[int], int, bool]:
    """Phase 1 on one level: move nodes between communities until no move increases modularity.

    Returns (community of each node, passes over all nodes, whether any node moved).
    """
    n = len(adj)
    comm = list(range(n))
    comm_degree = list(degree)
    passes = 0
    moved = False
    improved = True

    while improved:
        improved = False
        passes += 1

        for v in range(n):
            current = comm[v]
            k_v = degree[v]

            # Edge weight from v to each neighboring community
            edges_to: dict[int, float] = {}
            for w, weight in adj[v].items():
                c = comm[w]
                edges_to[c] = edges_to.get(c, 0.0) + weight

            # Remove v from its community, then find the best one to join
            comm_degree[current] -= k_v
            best, best_gain = current, 0.0
            for c, e in edges_to.items():
                gain = e - resolution * k_v * comm_degree[c] / two_m
                if gain > best_gain:
                    best, best_gain = c, gain

            # Staying put wins ties
            if best != current:
                stay_gain = edges_to.get(current, 0.0) - resolution * k_v * comm_degree[current] / two_m
                if stay_gain >= best_gain:
                    best = current

            comm[v] = best
            comm_degree[best] += k_v
            if best != current:
                improved = True
                moved = True

    return comm, passes, moved


def _aggregate(
    adj: list[dict[int, float]], loops: list[float], comm: list[int]
) -> tuple[list[dict[int, float]], list[float], list[int]]:
    """Phase 2: one node per community, numbered in order of first appearance.

    Edge weights between communities are summed; edges inside a community
    become its self-loop weight. Returns (adjacency, self-loops, node -> new node).
    """
    new_id: dict[int, int] = {}
    for c in comm:
        if c not in new_id:
            new_id[c] = len(new_id)

    k = len(new_id)
    new_adj: list[dict[int, float]] = [{} for _ in range(k)]
    new_loops = [0.0] * k
    for v, nbrs in enumerate(adj):
        cv = new_id[comm[v]]
        new_loops[cv] += loops[v]
        for w, weight in nbrs.items():
            cw = new_id[comm[w]]
            if cv == cw:
                new_loops[cv] += weight / 2.0  # each internal edge is seen from both ends
            else:
                new_adj[cv][cw] = new_adj[cv].get(cw, 0.0) + weight

    return new_adj, new_loops, [new_id[c] for c in comm]

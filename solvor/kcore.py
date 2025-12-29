r"""
K-core decomposition for identifying core vs periphery in graphs.

A k-core is the maximal subgraph where every node has degree at least k.
Higher k-cores represent the "core" of the graph - densely connected regions.
Lower k-cores are the "periphery" - loosely connected nodes.

    from solvor.kcore import kcore_decomposition, kcore

    # Collaboration network
    collabs = {
        "alice": ["bob", "carol", "dave"],
        "bob": ["alice", "carol", "dave"],
        "carol": ["alice", "bob", "dave"],
        "dave": ["alice", "bob", "carol", "eve"],
        "eve": ["dave"],
    }

    result = kcore_decomposition(collabs.keys(), lambda n: collabs.get(n, []))
    # result.solution = {"alice": 3, "bob": 3, "carol": 3, "dave": 3, "eve": 1}

    result = kcore(collabs.keys(), lambda n: collabs.get(n, []), k=3)
    # result.solution = {"alice", "bob", "carol", "dave"}  # core collaborators

How it works: Iteratively removes nodes with degree < k, updating remaining
degrees. A node's core number is the largest k for which it remains in the
k-core. Uses bucket sort for O(V + E) time complexity.

Use this for:

- Finding core vs peripheral components in codebases
- Identifying tightly-coupled subsystems
- Filtering noise to focus on essential structure
- Detecting influential groups in networks

Parameters:

    nodes: iterable of all nodes in the graph
    neighbors: function returning iterable of adjacent nodes (undirected)
    k: for kcore(), the specific core level to extract

Works with any hashable node type. Treats graph as undirected.
"""

from collections.abc import Callable, Iterable

from solvor.types import Result

__all__ = ["kcore_decomposition", "kcore"]


def kcore_decomposition[S](
    nodes: Iterable[S],
    neighbors: Callable[[S], Iterable[S]],
) -> Result[dict[S, int]]:
    """Compute the core number of each node.

    Returns a dict mapping each node to its core number (the largest k for
    which the node belongs to the k-core). Higher values = more central.
    """
    node_list = list(nodes)
    n = len(node_list)

    if n == 0:
        return Result({}, 0, 0, 0)

    # Build adjacency and compute degrees
    node_set = set(node_list)
    adj: dict[S, set[S]] = {v: set() for v in node_list}

    for v in node_list:
        for w in neighbors(v):
            if w in node_set and w != v:
                adj[v].add(w)
                adj[w].add(v)

    degree: dict[S, int] = {v: len(adj[v]) for v in node_list}
    max_degree = max(degree.values()) if degree else 0

    # Bucket sort by degree
    buckets: list[set[S]] = [set() for _ in range(max_degree + 1)]
    for v in node_list:
        buckets[degree[v]].add(v)

    core_number: dict[S, int] = {}
    iterations = 0

    # Process nodes in order of increasing degree
    for k in range(max_degree + 1):
        while buckets[k]:
            iterations += 1
            v = buckets[k].pop()
            core_number[v] = k

            # Update neighbors' degrees
            for w in adj[v]:
                if w not in core_number:
                    old_deg = degree[w]
                    if old_deg > k:
                        buckets[old_deg].remove(w)
                        degree[w] = old_deg - 1
                        new_bucket = max(k, degree[w])
                        buckets[new_bucket].add(w)

    max_core = max(core_number.values()) if core_number else 0
    return Result(core_number, max_core, iterations, n)


def kcore[S](
    nodes: Iterable[S],
    neighbors: Callable[[S], Iterable[S]],
    k: int,
) -> Result[set[S]]:
    """Extract the k-core subgraph.

    Returns the set of nodes in the k-core (nodes with core number >= k).
    The k-core is the maximal subgraph where every node has degree at least k.
    """
    decomp = kcore_decomposition(nodes, neighbors)
    core_nodes = {v for v, core in decomp.solution.items() if core >= k}
    return Result(core_nodes, len(core_nodes), decomp.iterations, decomp.evaluations)

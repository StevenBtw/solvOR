r"""
Strongly connected components and topological sorting for directed graphs.

SCC finds groups of nodes where every node can reach every other node in the
group. Topological sort orders nodes so dependencies come before dependents.
Together they're essential for dependency analysis and cycle detection.

    from solvor.scc import strongly_connected_components, topological_sort, condense

    # Find cycles/strongly coupled components
    result = strongly_connected_components(nodes, neighbors)
    # result.solution = [[a, b, c], [d], [e, f]]  # groups of tightly coupled nodes

    # Order by dependencies (fails if cycles exist)
    result = topological_sort(nodes, neighbors)
    # result.solution = [d, e, f, a, b, c]  # dependencies first

    # Create DAG by collapsing SCCs into single nodes
    result = condense(nodes, neighbors)
    # result.solution = (condensed_nodes, condensed_edges)

How it works: SCC uses Tarjan's algorithm with a single DFS pass, tracking
discovery times and low-links. Topological sort uses Kahn's algorithm,
repeatedly removing nodes with no incoming edges. Both are O(V + E).

Use this for:

- Detecting circular dependencies in codebases
- Finding tightly coupled modules that should be grouped
- Ordering build/compilation steps
- Layered architecture validation
- Dependency graph analysis

Parameters:

    nodes: iterable of all nodes in the graph
    neighbors: function returning iterable of outgoing edges (successors)

Works with any hashable node type. For incoming edges (predecessors), swap
the edge direction in your neighbors function.

Note: SCC uses recursion internally. For very deep graphs (>1000 nodes in a
single path), you may need to increase the recursion limit:

    import sys
    sys.setrecursionlimit(5000)  # adjust as needed
"""

from collections import defaultdict, deque
from collections.abc import Callable, Iterable

from solvor.types import Result, Status

__all__ = ["strongly_connected_components", "topological_sort", "condense"]


def strongly_connected_components[S](
    nodes: Iterable[S],
    neighbors: Callable[[S], Iterable[S]],
) -> Result:
    """Find strongly connected components using Tarjan's algorithm.

    Returns components in reverse topological order (sinks first).
    Each component is a list of nodes that can all reach each other.
    Single-node components with no self-loop are also returned.
    """
    node_list = list(nodes)
    index_counter = [0]
    stack: list[S] = []
    on_stack: set[S] = set()
    index: dict[S, int] = {}
    low_link: dict[S, int] = {}
    components: list[list[S]] = []
    iterations = 0

    def strongconnect(v: S) -> None:
        nonlocal iterations
        iterations += 1

        index[v] = index_counter[0]
        low_link[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)

        for w in neighbors(v):
            if w not in index:
                strongconnect(w)
                low_link[v] = min(low_link[v], low_link[w])
            elif w in on_stack:
                low_link[v] = min(low_link[v], index[w])

        if low_link[v] == index[v]:
            component: list[S] = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                component.append(w)
                if w == v:
                    break
            components.append(component)

    for v in node_list:
        if v not in index:
            strongconnect(v)

    return Result(components, len(components), iterations, len(node_list))


def topological_sort[S](
    nodes: Iterable[S],
    neighbors: Callable[[S], Iterable[S]],
) -> Result:
    """Topological sort using Kahn's algorithm.

    Returns nodes ordered so that for every edge u→v, u comes before v.
    Returns INFEASIBLE status if the graph contains a cycle.
    """
    node_list = list(nodes)
    node_set = set(node_list)

    # Build in-degree map
    in_degree: dict[S, int] = {v: 0 for v in node_list}
    adjacency: dict[S, list[S]] = {v: [] for v in node_list}

    for v in node_list:
        for w in neighbors(v):
            if w in node_set:
                adjacency[v].append(w)
                in_degree[w] += 1

    # Start with nodes that have no incoming edges
    queue: deque[S] = deque(v for v in node_list if in_degree[v] == 0)
    result: list[S] = []
    iterations = 0

    while queue:
        iterations += 1
        v = queue.popleft()
        result.append(v)

        for w in adjacency[v]:
            in_degree[w] -= 1
            if in_degree[w] == 0:
                queue.append(w)

    if len(result) != len(node_list):
        # Cycle detected - not all nodes processed
        return Result(None, 0, iterations, len(node_list), Status.INFEASIBLE)

    return Result(result, len(result), iterations, len(node_list))


def condense[S](
    nodes: Iterable[S],
    neighbors: Callable[[S], Iterable[S]],
) -> Result:
    """Condense graph by collapsing SCCs into single nodes.

    Returns a tuple of:
    - List of condensed nodes (each is a frozenset of original nodes)
    - Dict mapping each condensed node to its successors

    The resulting graph is a DAG (directed acyclic graph).
    Useful for simplifying dependency analysis after finding cycles.
    """
    node_list = list(nodes)

    # Get SCCs
    scc_result = strongly_connected_components(node_list, neighbors)
    components: list[list[S]] = scc_result.solution

    # Map each node to its component
    node_to_component: dict[S, int] = {}
    for i, component in enumerate(components):
        for node in component:
            node_to_component[node] = i

    # Build condensed graph
    condensed_nodes: list[frozenset[S]] = [frozenset(c) for c in components]
    condensed_edges: dict[int, set[int]] = defaultdict(set)

    iterations = scc_result.iterations
    for v in node_list:
        iterations += 1
        v_comp = node_to_component[v]
        for w in neighbors(v):
            if w in node_to_component:
                w_comp = node_to_component[w]
                if v_comp != w_comp:
                    condensed_edges[v_comp].add(w_comp)

    # Convert to adjacency dict with frozenset keys
    adjacency: dict[frozenset[S], list[frozenset[S]]] = {
        condensed_nodes[i]: [condensed_nodes[j] for j in condensed_edges[i]]
        for i in range(len(condensed_nodes))
    }

    return Result(
        (condensed_nodes, adjacency),
        len(condensed_nodes),
        iterations,
        len(node_list),
    )

"""
Network Flow - Max Flow and Min Cost Flow Algorithms

Solves network flow problems using Ford-Fulkerson (max flow) and
Successive Shortest Paths (min cost flow). Useful for assignment,
bipartite matching, transportation, and resource allocation.

Usage:
    from solvor.flow import max_flow, min_cost_flow, Status

    # Max flow: find maximum flow from source to sink
    flow_value, flows = max_flow(graph, source, sink)

    # Min cost flow: find cheapest way to send `demand` units
    cost, flows = min_cost_flow(graph, source, sink, demand)

Graph format:
    graph = {
        node: [(neighbor, capacity, cost), ...],
        ...
    }
    For max_flow, cost is ignored (can be 0).

Returns Result(solution, objective, iterations, evaluations, status)
    solution = flows dict where flows[(u,v)] = flow on edge
    objective = total flow (max_flow) or total cost (min_cost_flow)

Examples:
    # Bipartite matching: source->left, left->right, right->sink (cap=1)
    # Assignment problem: use min_cost_flow with costs
"""

from collections import defaultdict, deque
from collections.abc import Sequence
from solvor.types import Status, Result

__all__ = ["max_flow", "min_cost_flow", "solve_assignment", "Status", "Result"]

def max_flow[Node](
    graph: dict[Node, list[tuple[Node, int, ...]]],
    source: Node,
    sink: Node,
) -> Result:
    """(graph, source, sink) -> Result with max flow value and flows dict."""
    capacity = defaultdict(lambda: defaultdict(int))
    for u in graph:
        for v, cap, *_ in graph[u]:
            capacity[u][v] += cap

    flow = defaultdict(lambda: defaultdict(int))
    total_flow = 0

    def bfs():
        visited = {source}
        queue = deque([(source, [source])])

        while queue:
            node, path = queue.popleft()
            if node == sink:
                return path

            for neighbor in capacity[node]:
                residual = capacity[node][neighbor] - flow[node][neighbor] + flow[neighbor][node]
                if neighbor not in visited and residual > 0:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    while (path := bfs()):
        path_flow = float('inf')
        for u, v in zip(path, path[1:]):
            residual = capacity[u][v] - flow[u][v] + flow[v][u]
            path_flow = min(path_flow, residual)

        for u, v in zip(path, path[1:]):
            if flow[v][u] > 0:
                reduce = min(path_flow, flow[v][u])
                flow[v][u] -= reduce
                path_flow_remaining = path_flow - reduce
                flow[u][v] += path_flow_remaining
            else:
                flow[u][v] += path_flow

        total_flow += path_flow

    flows = {(u, v): flow[u][v] for u in flow for v in flow[u] if flow[u][v] > 0}
    return Result(flows, total_flow, 0, 0, Status.OPTIMAL)

def min_cost_flow[Node](
    graph: dict[Node, list[tuple[Node, int, int]]],
    source: Node,
    sink: Node,
    demand: int,
) -> Result:
    """(graph, source, sink, demand) -> Result using Successive Shortest Paths."""
    capacity = defaultdict(lambda: defaultdict(int))
    cost = defaultdict(lambda: defaultdict(lambda: float('inf')))
    nodes = set()

    for u in graph:
        nodes.add(u)
        for v, cap, c in graph[u]:
            nodes.add(v)
            capacity[u][v] += cap
            cost[u][v] = min(cost[u][v], c)
            if cost[v][u] == float('inf'):
                cost[v][u] = -c

    flow = defaultdict(lambda: defaultdict(int))
    total_cost = 0
    total_flow = 0

    def bellman_ford():
        dist = {n: float('inf') for n in nodes}
        parent = {n: None for n in nodes}
        dist[source] = 0

        for _ in range(len(nodes) - 1):
            updated = False
            for u in nodes:
                if dist[u] == float('inf'):
                    continue
                for v in nodes:
                    residual = capacity[u][v] - flow[u][v] + flow[v][u]
                    if residual > 0 and dist[u] + cost[u][v] < dist[v]:
                        dist[v] = dist[u] + cost[u][v]
                        parent[v] = u
                        updated = True
            if not updated:
                break

        if dist[sink] == float('inf'):
            return None, float('inf')

        path = []
        node = sink
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()

        return path, dist[sink]

    while total_flow < demand:
        path, path_cost = bellman_ford()
        if path is None:
            return Result({}, float('inf'), 0, 0, Status.INFEASIBLE)

        path_flow = demand - total_flow
        for u, v in zip(path, path[1:]):
            residual = capacity[u][v] - flow[u][v] + flow[v][u]
            path_flow = min(path_flow, residual)

        for u, v in zip(path, path[1:]):
            if flow[v][u] > 0:
                reduce = min(path_flow, flow[v][u])
                flow[v][u] -= reduce
                remaining = path_flow - reduce
                flow[u][v] += remaining
                total_cost += cost[u][v] * remaining - cost[v][u] * reduce
            else:
                flow[u][v] += path_flow
                total_cost += cost[u][v] * path_flow

        total_flow += path_flow

    flows = {(u, v): flow[u][v] for u in flow for v in flow[u] if flow[u][v] > 0}
    return Result(flows, total_cost, 0, 0, Status.OPTIMAL)

def solve_assignment(
    cost_matrix: Sequence[Sequence[float]],
) -> Result:
    """(cost_matrix) -> Result with minimum cost assignment."""
    n = len(cost_matrix)
    m = len(cost_matrix[0]) if n > 0 else 0

    graph = defaultdict(list)
    source, sink = 'source', 'sink'

    for i in range(n):
        graph[source].append((f'L{i}', 1, 0))
        for j in range(m):
            graph[f'L{i}'].append((f'R{j}', 1, cost_matrix[i][j]))

    for j in range(m):
        graph[f'R{j}'].append((sink, 1, 0))

    result = min_cost_flow(graph, source, sink, min(n, m))

    assignment = [-1] * n
    for (u, v), f in result.solution.items():
        if f > 0 and u.startswith('L') and v.startswith('R'):
            i = int(u[1:])
            j = int(v[1:])
            assignment[i] = j

    return Result(assignment, result.objective, 0, 0, result.status)

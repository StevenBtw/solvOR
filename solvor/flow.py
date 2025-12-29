r"""
Network Flow, for assignment, matching, and bottleneck analysis.

Use max_flow when you need "how much can I push through this network?"
Use min_cost_flow when cost matters: "what's the cheapest way to route X units?"

    from solvor.flow import max_flow, min_cost_flow

    # graph: {node: [(neighbor, capacity, cost), ...]}
    result = max_flow(graph, source='s', sink='t')
    result = min_cost_flow(graph, source='s', sink='t', demand=10)

How it works: max_flow uses Ford-Fulkerson with BFS (Edmonds-Karp), repeatedly
finding augmenting paths until none exist. min_cost_flow uses successive
shortest paths via Bellman-Ford, finding the cheapest augmenting path each time.
The max-flow min-cut theorem means you find bottlenecks for free.

Use this for:

- Assignment and matching problems
- Server-to-request allocation
- Transportation and logistics
- Finding network bottlenecks (max-flow = min-cut)

Parameters:

    graph: adjacency dict {node: [(neighbor, capacity, cost), ...]}
    source: source node
    sink: sink node
    demand: (min_cost_flow only) units to route

Works well with NetworkX for graph construction. For heavier graph work
(shortest paths, centrality, community detection), NetworkX is more extensive.
"""

from collections import defaultdict, deque
from collections.abc import Sequence

from solvor.types import Result, Status

__all__ = ["max_flow", "min_cost_flow", "solve_assignment"]


def max_flow[Node](
    graph: dict[Node, list[tuple[Node, int, ...]]],
    source: Node,
    sink: Node,
) -> Result:
    """Find maximum flow from source to sink using Ford-Fulkerson (FFA) with BFS."""
    capacity = defaultdict(lambda: defaultdict(int))
    for u in graph:
        for v, cap, *_ in graph[u]:
            capacity[u][v] += cap

    flow = defaultdict(lambda: defaultdict(int))
    total_flow = 0
    iterations = 0

    def bfs():
        visited = {source}
        queue = deque([(source, [source])])

        while queue:
            node, path = queue.popleft()
            if node == sink:
                return path

            for neighbor in capacity[node]:
                # Forward capacity minus used, plus reverse flow we can cancel
                residual = capacity[node][neighbor] - flow[node][neighbor] + flow[neighbor][node]
                if neighbor not in visited and residual > 0:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    while path := bfs():
        iterations += 1
        path_flow = float("inf")
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
    return Result(flows, total_flow, iterations, iterations)


def min_cost_flow[Node](
    graph: dict[Node, list[tuple[Node, int, int]]],
    source: Node,
    sink: Node,
    demand: int,
) -> Result:
    """Route demand units from source to sink at minimum total cost."""
    capacity = defaultdict(lambda: defaultdict(int))
    cost = defaultdict(lambda: defaultdict(lambda: float("inf")))
    nodes = set()

    for u in graph:
        nodes.add(u)
        for v, cap, c in graph[u]:
            nodes.add(v)
            capacity[u][v] += cap
            cost[u][v] = min(cost[u][v], c)
            if cost[v][u] == float("inf"):
                cost[v][u] = -c

    flow = defaultdict(lambda: defaultdict(int))
    total_cost = 0
    total_flow = 0
    iterations = 0

    def bellman_ford():
        dist = {n: float("inf") for n in nodes}
        parent = {n: None for n in nodes}
        dist[source] = 0

        for _ in range(len(nodes) - 1):
            updated = False
            for u in nodes:
                if dist[u] == float("inf"):
                    continue
                for v in nodes:
                    residual = capacity[u][v] - flow[u][v] + flow[v][u]
                    if residual > 0 and dist[u] + cost[u][v] < dist[v]:
                        dist[v] = dist[u] + cost[u][v]
                        parent[v] = u
                        updated = True
            if not updated:
                break

        if dist[sink] == float("inf"):
            return None, float("inf")

        path = []
        node = sink
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()

        return path, dist[sink]

    while total_flow < demand:
        iterations += 1
        path, path_cost = bellman_ford()
        if path is None:
            return Result({}, float("inf"), iterations, iterations, Status.INFEASIBLE)

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
    return Result(flows, total_cost, iterations, iterations)


def solve_assignment(
    cost_matrix: Sequence[Sequence[float]],
) -> Result:
    """Solve assignment problem via min-cost flow reduction."""
    n = len(cost_matrix)
    m = len(cost_matrix[0]) if n > 0 else 0

    graph = defaultdict(list)
    source, sink = "source", "sink"

    for i in range(n):
        graph[source].append((f"L{i}", 1, 0))
        for j in range(m):
            graph[f"L{i}"].append((f"R{j}", 1, cost_matrix[i][j]))

    for j in range(m):
        graph[f"R{j}"].append((sink, 1, 0))

    result = min_cost_flow(graph, source, sink, min(n, m))

    assignment = [-1] * n
    flow_solution: dict[tuple[str, str], float] = result.solution  # type: ignore[assignment]
    for (u, v), f in flow_solution.items():
        if f > 0 and u.startswith("L") and v.startswith("R"):
            i = int(u[1:])
            j = int(v[1:])
            assignment[i] = j

    return Result(assignment, result.objective, result.iterations, result.evaluations, result.status)

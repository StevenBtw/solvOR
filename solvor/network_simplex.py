"""
Network Simplex - simplex specialized for flow networks.

Remember simplex walking along edges of a crystal? Network simplex does the
same thing, but the crystal has special structure: it's a flow network. Instead
of a dense tableau, it maintains a spanning tree. Pivots become tree edge swaps.
This structure makes it fly on large networks where regular simplex would struggle.

    from solvor.network_simplex import network_simplex

    result = network_simplex(n_nodes, arcs, supplies)

    Source [+10] --(cap:10, cost:2)--> [0] Transship --(cap:15, cost:1)--> [-10] Sink
                  \__(cap:5, cost:3)_____________________/

    supplies: positive = produces flow, negative = consumes, zero = passes through
    arcs: (from, to, capacity, cost)

Use this over min_cost_flow when you have large networks. min_cost_flow uses
successive shortest paths which is simpler but slower. Network simplex scales.

Don't use this for: problems that aren't min-cost flow shaped (use solve_lp),
or tiny problems where config overhead isn't worth it (use min_cost_flow).
"""

from solvor.types import Result, Status

__all__ = ["network_simplex"]


def network_simplex(
    n_nodes: int,
    arcs: list[tuple[int, int, int, float]],
    supplies: list[float],
    *,
    max_iter: int = 1_000_000,
) -> Result:
    if abs(sum(supplies)) > 1e-9:
        return Result(None, float("inf"), 0, 0, Status.INFEASIBLE)

    if not arcs:
        if all(abs(s) < 1e-9 for s in supplies):
            return Result({}, 0.0, 0, 0)
        return Result(None, float("inf"), 0, 0, Status.INFEASIBLE)

    m = len(arcs)
    n = n_nodes

    source = [0] * (m + n)
    target = [0] * (m + n)
    cap = [0] * (m + n)
    cost = [0.0] * (m + n)
    flow = [0] * (m + n)

    for i, (u, v, c, w) in enumerate(arcs):
        source[i] = u
        target[i] = v
        cap[i] = c
        cost[i] = w

    big_m = sum(abs(cost[i]) for i in range(m)) * n + 1

    for i in range(n):
        arc_id = m + i
        if supplies[i] >= 0:
            source[arc_id] = i
            target[arc_id] = n
            cap[arc_id] = int(supplies[i]) + 1
            flow[arc_id] = int(supplies[i])
        else:
            source[arc_id] = n
            target[arc_id] = i
            cap[arc_id] = int(-supplies[i]) + 1
            flow[arc_id] = int(-supplies[i])
        cost[arc_id] = big_m

    total_arcs = m + n
    total_nodes = n + 1
    root = n

    parent = [root] * total_nodes
    parent[root] = -1
    pred = list(range(m, m + n)) + [-1]
    depth = [1] * total_nodes
    depth[root] = 0
    thread = list(range(1, total_nodes)) + [0]
    thread[n - 1] = root
    thread[root] = 0
    rev_thread = [root] + list(range(total_nodes - 1))
    rev_thread[root] = n - 1

    pi = [0.0] * total_nodes
    for i in range(n):
        arc = pred[i]
        if source[arc] == i:
            pi[i] = pi[root] + cost[arc]
        else:
            pi[i] = pi[root] - cost[arc]

    state = [0] * total_arcs
    for arc in range(total_arcs):
        if flow[arc] == 0:
            state[arc] = 1
        elif flow[arc] == cap[arc]:
            state[arc] = -1
        else:
            state[arc] = 0

    iterations = 0

    while iterations < max_iter:
        iterations += 1

        entering = -1
        best_cost = -1e-9

        for arc in range(total_arcs):
            if state[arc] == 0:
                continue
            u, v = source[arc], target[arc]
            rc = cost[arc] - pi[u] + pi[v]

            if state[arc] == 1 and rc < best_cost:
                best_cost = rc
                entering = arc
            elif state[arc] == -1 and -rc < best_cost:
                best_cost = -rc
                entering = arc

        if entering == -1:
            break

        u, v = source[entering], target[entering]
        rc = cost[entering] - pi[u] + pi[v]

        if rc < 0:
            delta = cap[entering] - flow[entering]
            first, second = u, v
        else:
            delta = flow[entering]
            first, second = v, u

        join = _find_join(first, second, depth, parent)

        leaving = entering
        leaving_first = True

        node = first
        while node != join:
            arc = pred[node]
            d = _residual(arc, node, source, flow, cap)
            if d < delta:
                delta = d
                leaving = arc
                leaving_first = True
            node = parent[node]

        node = second
        while node != join:
            arc = pred[node]
            d = _residual(arc, parent[node], source, flow, cap)
            if d < delta:
                delta = d
                leaving = arc
                leaving_first = False
            node = parent[node]

        if delta == 0 and leaving == entering:
            state[entering] = -state[entering]
            continue

        if rc < 0:
            flow[entering] += delta
        else:
            flow[entering] -= delta

        node = first
        while node != join:
            arc = pred[node]
            if source[arc] == node:
                flow[arc] -= delta
            else:
                flow[arc] += delta
            node = parent[node]

        node = second
        while node != join:
            arc = pred[node]
            if source[arc] == node:
                flow[arc] += delta
            else:
                flow[arc] -= delta
            node = parent[node]

        for arc in range(total_arcs):
            if flow[arc] == 0:
                state[arc] = 1
            elif flow[arc] == cap[arc]:
                state[arc] = -1
            else:
                state[arc] = 0

        if leaving != entering:
            if leaving_first:
                leaving_node = first
                while pred[leaving_node] != leaving:
                    leaving_node = parent[leaving_node]
                new_parent = second
            else:
                leaving_node = second
                while pred[leaving_node] != leaving:
                    leaving_node = parent[leaving_node]
                new_parent = first

            prev_thread = rev_thread[leaving_node]
            subtree_last = leaving_node
            node = thread[leaving_node]
            while depth[node] > depth[leaving_node]:
                subtree_last = node
                node = thread[node]

            thread[prev_thread] = thread[subtree_last]
            rev_thread[thread[subtree_last]] = prev_thread

            attach_point = new_parent
            node = thread[new_parent]
            while node != new_parent and depth[node] > depth[new_parent]:
                attach_point = node
                node = thread[node]

            thread[subtree_last] = thread[attach_point]
            if thread[attach_point] < total_nodes:
                rev_thread[thread[attach_point]] = subtree_last
            thread[attach_point] = leaving_node
            rev_thread[leaving_node] = attach_point

            parent[leaving_node] = new_parent
            pred[leaving_node] = entering

            diff = depth[new_parent] + 1 - depth[leaving_node]
            node = leaving_node
            while True:
                depth[node] += diff
                node = thread[node]
                if depth[node] <= depth[leaving_node] - diff or node == leaving_node:
                    break

            node = leaving_node
            while True:
                arc = pred[node]
                if source[arc] == parent[node]:
                    pi[node] = pi[parent[node]] - cost[arc]
                else:
                    pi[node] = pi[parent[node]] + cost[arc]
                node = thread[node]
                if depth[node] <= depth[new_parent] or node == leaving_node:
                    break

    for arc in range(m, total_arcs):
        if flow[arc] > 0:
            return Result(None, float("inf"), iterations, total_arcs, Status.INFEASIBLE)

    total_cost = sum(flow[i] * cost[i] for i in range(m))
    flow_dict = {(source[i], target[i]): flow[i] for i in range(m) if flow[i] > 0}

    return Result(flow_dict, total_cost, iterations, total_arcs)


def _find_join(u, v, depth, parent):
    while u != v:
        if depth[u] > depth[v]:
            u = parent[u]
        else:
            v = parent[v]
    return u


def _residual(arc, node, source, flow, cap):
    if source[arc] == node:
        return flow[arc]
    return cap[arc] - flow[arc]

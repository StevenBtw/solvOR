"""
Network Simplex Example

Solve min-cost flow problem on transportation network.
"""

from solvor import network_simplex

# Nodes: 0,1 are sources, 2,3 are sinks
# supplies[i] > 0 = source, < 0 = sink
supplies = [10, 10, -8, -12]

# Arcs: (from, to, capacity, cost)
arcs = [
    (0, 2, 8, 2),
    (0, 3, 5, 4),
    (1, 2, 5, 5),
    (1, 3, 10, 3),
]

result = network_simplex(4, arcs, supplies)
print(f"Flow on arcs: {result.solution}")
print(f"Total cost: {result.objective}")

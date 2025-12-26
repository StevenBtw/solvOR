"""
Max Flow Example

Find maximum flow from source to sink in network.
"""

from solvor import max_flow

# Graph as adjacency list: {node: [(neighbor, capacity), ...]}
graph = {
    0: [(1, 10), (2, 10)],
    1: [(2, 2), (3, 4), (4, 8)],
    2: [(4, 9)],
    3: [(5, 10)],
    4: [(3, 6), (5, 10)],
    5: [],
}

result = max_flow(graph, source=0, sink=5)
print(f"Maximum flow: {result.objective}")

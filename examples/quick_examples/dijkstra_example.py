"""
Dijkstra's Algorithm Example

Find shortest path from node 0 to node 4 in weighted graph.
"""

from solvor import dijkstra

# Graph as adjacency list: {node: [(neighbor, weight), ...]}
graph = {
    0: [(1, 4), (2, 1)],
    1: [(3, 1)],
    2: [(1, 2), (3, 5)],
    3: [(4, 3)],
    4: [],
}


def neighbors(node):
    return graph.get(node, [])


result = dijkstra(0, 4, neighbors)
print(f"Shortest path: {result.solution}")
print(f"Total distance: {result.objective}")

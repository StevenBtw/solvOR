"""
Breadth-First Search Example

Find shortest path (by edge count) in unweighted graph.
"""

from solvor import bfs

# Graph as adjacency list
graph = {
    0: [1, 2],
    1: [0, 3, 4],
    2: [0, 4],
    3: [1, 5],
    4: [1, 2, 5],
    5: [3, 4],
}


def neighbors(node):
    return graph.get(node, [])


result = bfs(0, 5, neighbors)
print(f"Path: {result.solution}")
print(f"Distance (edges): {result.objective}")

"""
Depth-First Search Example

Find a path (not necessarily shortest) in a graph.
"""

from solvor import dfs

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


result = dfs(0, 5, neighbors)
print(f"Path: {result.solution}")
print(f"Status: {result.status.name}")

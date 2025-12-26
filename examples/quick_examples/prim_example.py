"""
Prim's MST Example

Find minimum spanning tree using vertex-based approach.
"""

from solvor import prim

# Graph as adjacency list: {node: [(neighbor, weight), ...]}
graph = {
    0: [(1, 4), (2, 3)],
    1: [(0, 4), (2, 1), (3, 2)],
    2: [(0, 3), (1, 1), (3, 4), (4, 5)],
    3: [(1, 2), (2, 4), (4, 2)],
    4: [(2, 5), (3, 2)],
}

result = prim(graph)
print(f"MST edges: {result.solution}")
print(f"Total weight: {result.objective}")

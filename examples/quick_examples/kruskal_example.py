"""
Kruskal's MST Example

Find minimum spanning tree using edge-based approach.
"""

from solvor import kruskal

# Edges as (u, v, weight)
edges = [
    (0, 1, 4),
    (0, 2, 3),
    (1, 2, 1),
    (1, 3, 2),
    (2, 3, 4),
    (3, 4, 2),
    (2, 4, 5),
]

result = kruskal(5, edges)
print(f"MST edges: {result.solution}")
print(f"Total weight: {result.objective}")

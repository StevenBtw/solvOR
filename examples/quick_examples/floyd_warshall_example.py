"""
Floyd-Warshall Example

Compute all-pairs shortest paths.
"""

from solvor import floyd_warshall

# Edges as (u, v, weight)
edges = [
    (0, 1, 3),
    (0, 3, 7),
    (1, 0, 8),
    (1, 2, 2),
    (2, 0, 5),
    (2, 3, 1),
    (3, 0, 2),
]

result = floyd_warshall(4, edges)
print("Shortest path distances:")
INF = float("inf")
for row in result.solution:
    print([f"{x:3.0f}" if x != INF else "INF" for x in row])

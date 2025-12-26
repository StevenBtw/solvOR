"""
Bellman-Ford Example

Find shortest paths from source, handles negative weights.
"""

from solvor import bellman_ford

# Edges as (u, v, weight)
edges = [
    (0, 1, 4),
    (0, 2, 5),
    (1, 2, -3),
    (2, 3, 4),
]

result = bellman_ford(0, edges, 4)
print(f"Distances from node 0: {result.solution}")
# result.solution[i] = shortest distance to node i

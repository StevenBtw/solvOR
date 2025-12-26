"""
Min-Cost Flow Example

Find minimum cost way to send flow through a network.
"""

from solvor import min_cost_flow

# Graph: node -> [(neighbor, capacity, cost), ...]
graph = {
    "s": [("a", 10, 2), ("b", 5, 3)],
    "a": [("b", 15, 1), ("t", 10, 4)],
    "b": [("t", 10, 2)],
    "t": [],
}

result = min_cost_flow(graph, source="s", sink="t", demand=10)
print(f"Total cost: {result.objective}")
print(f"Flow: {result.solution}")

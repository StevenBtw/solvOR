"""
Articulation Points and Bridges Example

Find critical nodes and edges whose removal disconnects the graph.
"""

from solvor import articulation_points, bridges

# Network topology - servers and connections
network = {
    # Data center 1
    "dc1-web": ["dc1-app", "dc1-db"],
    "dc1-app": ["dc1-web", "dc1-db"],
    "dc1-db": ["dc1-web", "dc1-app", "gateway"],
    # Gateway connecting data centers
    "gateway": ["dc1-db", "dc2-db"],
    # Data center 2
    "dc2-db": ["gateway", "dc2-app", "dc2-web"],
    "dc2-app": ["dc2-db", "dc2-web"],
    "dc2-web": ["dc2-db", "dc2-app"],
}


def neighbors(server):
    return network.get(server, [])


# Find single points of failure
ap_result = articulation_points(network.keys(), neighbors)
print("Articulation points (single points of failure):")
for node in ap_result.solution:
    print(f"  {node}")

# Find critical connections
br_result = bridges(network.keys(), neighbors)
print("\nBridges (critical connections):")
for edge in br_result.solution:
    print(f"  {edge[0]} -- {edge[1]}")

"""
Louvain Community Detection Example

Find clusters of densely connected nodes.
"""

from solvor import louvain

# Social network - two friend groups
friendships = {
    # Group 1: work colleagues
    "alice": ["bob", "carol", "dave"],
    "bob": ["alice", "carol", "dave"],
    "carol": ["alice", "bob", "dave"],
    "dave": ["alice", "bob", "carol", "eve"],  # dave knows eve
    # Group 2: gaming buddies
    "eve": ["dave", "frank", "grace"],
    "frank": ["eve", "grace", "henry"],
    "grace": ["eve", "frank", "henry"],
    "henry": ["frank", "grace"],
}


def neighbors(person):
    return friendships.get(person, [])


result = louvain(friendships.keys(), neighbors)

print(f"Found {len(result.solution)} communities:")
for i, community in enumerate(result.solution, 1):
    print(f"  Community {i}: {sorted(community)}")
print(f"Modularity: {result.objective:.3f}")

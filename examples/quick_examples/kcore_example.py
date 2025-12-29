"""
K-Core Decomposition Example

Identify core vs periphery in a graph.
"""

from solvor import kcore, kcore_decomposition

# Collaboration network - some people collaborate frequently
collaborations = {
    # Core team - everyone works with everyone
    "alice": ["bob", "carol", "dave"],
    "bob": ["alice", "carol", "dave"],
    "carol": ["alice", "bob", "dave"],
    "dave": ["alice", "bob", "carol", "eve"],
    # Peripheral contributors
    "eve": ["dave", "frank"],
    "frank": ["eve"],
}


def neighbors(person):
    return collaborations.get(person, [])


# Get core number for each person
decomp = kcore_decomposition(collaborations.keys(), neighbors)
print("Core numbers (higher = more central):")
for person, core in sorted(decomp.solution.items(), key=lambda x: -x[1]):
    print(f"  {person}: {core}")

# Extract the 2-core (people with at least 2 strong collaborators)
core_team = kcore(collaborations.keys(), neighbors, k=2)
print(f"\nCore team (2-core): {sorted(core_team.solution)}")

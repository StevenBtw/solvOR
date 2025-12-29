"""
Strongly Connected Components Example

Find circular dependencies in a codebase.
"""

from solvor import strongly_connected_components

# Module imports (some are circular)
imports = {
    "auth": ["user", "session"],
    "user": ["database", "auth"],  # auth <-> user cycle
    "session": ["user"],
    "database": ["config"],
    "config": [],
    "api": ["auth", "user"],
}


def neighbors(module):
    return imports.get(module, [])


result = strongly_connected_components(imports.keys(), neighbors)

print("Strongly connected components:")
for scc in result.solution:
    if len(scc) > 1:
        print(f"  Circular: {scc}")
    else:
        print(f"  Single: {scc}")

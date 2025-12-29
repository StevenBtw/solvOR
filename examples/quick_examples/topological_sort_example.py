"""
Topological Sort Example

Order tasks by dependencies - no task starts before its dependencies.
"""

from solvor import topological_sort

# Build system: each module depends on others
dependencies = {
    "app": ["ui", "api"],
    "ui": ["utils", "config"],
    "api": ["utils", "database"],
    "database": ["config"],
    "utils": ["config"],
    "config": [],
}


def neighbors(module):
    return dependencies.get(module, [])


result = topological_sort(dependencies.keys(), neighbors)
print(f"Build order: {result.solution}")
# config first, then utils/database, then ui/api, finally app

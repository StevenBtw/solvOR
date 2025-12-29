# Dependency Analysis

Cycle detection, topological ordering, and graph condensation. Essential for build systems, compiler optimization, and architecture validation.

## topological_sort

Order nodes so dependencies come before dependents. Uses [Kahn's algorithm](https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm).

```python
def topological_sort[S](
    nodes: Iterable[S],
    neighbors: Callable[[S], Iterable[S]],
) -> Result[list[S] | None]
```

### Example

```python
from solvor import topological_sort

# Build dependencies: app needs ui and api, etc.
deps = {
    "app": ["ui", "api"],
    "ui": ["utils"],
    "api": ["utils"],
    "utils": [],
}

result = topological_sort(deps.keys(), lambda n: deps[n])
print(result.solution)  # ['app', 'ui', 'api', 'utils'] or similar valid order
```

**Complexity:** O(V + E)
**Returns:** `INFEASIBLE` if graph contains a cycle

## strongly_connected_components

Find groups of nodes where every node can reach every other. Uses [Tarjan's algorithm](https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm).

```python
def strongly_connected_components[S](
    nodes: Iterable[S],
    neighbors: Callable[[S], Iterable[S]],
) -> Result[list[list[S]]]
```

### Example

```python
from solvor import strongly_connected_components

# Circular imports: auth <-> user
imports = {
    "auth": ["user"],
    "user": ["auth", "db"],
    "db": [],
}

result = strongly_connected_components(imports.keys(), lambda n: imports[n])
for scc in result.solution:
    if len(scc) > 1:
        print(f"Circular dependency: {scc}")
# Output: Circular dependency: ['user', 'auth']
```

**Complexity:** O(V + E)
**Output order:** Components in reverse topological order (sinks first)

## condense

Collapse SCCs into single nodes, creating a DAG. Useful for simplifying dependency graphs after detecting cycles.

```python
def condense[S](
    nodes: Iterable[S],
    neighbors: Callable[[S], Iterable[S]],
) -> Result[tuple[list[frozenset[S]], dict[frozenset[S], list[frozenset[S]]]]]
```

### Example

```python
from solvor import condense

result = condense(imports.keys(), lambda n: imports[n])
condensed_nodes, adjacency = result.solution
# Cycles become single nodes, result is always a DAG
```

## When to Use

| Algorithm | Use When |
|-----------|----------|
| `topological_sort` | Build order, task scheduling, dependency resolution |
| `strongly_connected_components` | Detecting circular dependencies, finding tightly-coupled modules |
| `condense` | Simplifying graphs with cycles for further analysis |

## Tips

- **Cycle detection.** If `topological_sort` returns `INFEASIBLE`, use `strongly_connected_components` to find the cycles.
- **Build systems.** Process in reverse order of `topological_sort` result (dependencies first).
- **Architecture validation.** SCCs with multiple nodes indicate coupling that may need refactoring.
- **Recursion limit.** For very deep graphs (>1000 nodes in a single path), increase `sys.setrecursionlimit`.

## See Also

- [Pathfinding](pathfinding.md) - BFS/DFS for traversal
- [Shortest Paths](shortest-paths.md) - When edge weights matter
- [Wikipedia: Topological sorting](https://en.wikipedia.org/wiki/Topological_sorting)
- [Wikipedia: Strongly connected component](https://en.wikipedia.org/wiki/Strongly_connected_component)

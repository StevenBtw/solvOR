# Pathfinding (BFS / DFS)

Basic graph traversal. BFS finds shortest paths in unweighted graphs. DFS finds a path (not necessarily shortest).

## bfs

Breadth-first search. Explores level by level.

```python
def bfs[S](
    start: S,
    goal: S | Callable[[S], bool],
    neighbors: Callable[[S], Iterable[S]],
    *,
    max_iter: int = 1_000_000,
) -> Result[list[S] | None]
```

### Example

```python
from solvor import bfs

graph = {
    'A': ['B', 'C'],
    'B': ['D'],
    'C': ['D'],
    'D': []
}

result = bfs('A', 'D', lambda n: graph[n])
print(result.solution)  # ['A', 'B', 'D'] or ['A', 'C', 'D']
```

**Complexity:** O(V + E)
**Guarantees:** Optimal for unweighted graphs

## dfs

Depth-first search. Explores deeply before backtracking.

```python
def dfs[S](
    start: S,
    goal: S | Callable[[S], bool],
    neighbors: Callable[[S], Iterable[S]],
    *,
    max_depth: int | None = None,
) -> Result[list[S] | None]
```

### Example

```python
from solvor import dfs

result = dfs('A', 'D', lambda n: graph[n])
print(result.solution)  # Some path (not necessarily shortest)
```

**Complexity:** O(V + E)
**Guarantees:** Finds a path if one exists (not shortest)

## When to Use

| Algorithm | Use When |
|-----------|----------|
| BFS | Need shortest path (fewest edges) |
| DFS | Just need any path, or for connectivity/cycle detection |

## Tips

- **BFS for shortest paths.** DFS doesn't guarantee shortest.
- **DFS for memory efficiency.** O(depth) vs O(breadth).
- **Goal as function.** Pass `goal=lambda n: n.is_target()` for complex goals.

## See Also

- [Shortest Paths](shortest-paths.md) - For weighted graphs

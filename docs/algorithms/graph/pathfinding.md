# Pathfinding (BFS / DFS)

Basic graph traversal. [BFS](https://en.wikipedia.org/wiki/Breadth-first_search) finds shortest paths in unweighted graphs. [DFS](https://en.wikipedia.org/wiki/Depth-first_search) finds a path (not necessarily shortest).

## bfs

Breadth-first search. Explores level by level, guaranteeing the shortest path in unweighted graphs.

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
- **DFS for memory efficiency.** O(depth) vs O(breadth) space.
- **Goal as function.** Pass `goal=lambda n: n.is_target()` for complex goal conditions.
- **Iterative deepening.** For memory-efficient shortest paths, use iterative deepening DFS (not implemented, but easy to build on top of `dfs`).

## See Also

- [Shortest Paths](shortest-paths.md) - For weighted graphs (Dijkstra, A*, Bellman-Ford)
- [Wikipedia: Graph traversal](https://en.wikipedia.org/wiki/Graph_traversal) - Overview of traversal algorithms

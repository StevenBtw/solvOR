# dfs

**Depth-first search - finds *a* path, not necessarily shortest.**

## At a Glance

- **Category:** [Pathfinding](Pathfinding)
- **Problem Type:** Reachability, connectivity
- **Complexity:** O(V + E)
- **Guarantees:** Finds a path if one exists

## Quick Example

```python
from solvor import dfs

result = dfs('A', 'D', lambda n: graph[n])
print(result.solution)  # Some path from A to D
```

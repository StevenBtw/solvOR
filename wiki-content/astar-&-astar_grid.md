# astar & astar_grid

**A* search - Dijkstra with a heuristic compass.**

## Summary 

- **Category:** [Pathfinding](Pathfinding)
- **Problem Type:** Goal-directed shortest path
- **Complexity:** O((V + E) log V) best case
- **Guarantees:** Optimal with admissible heuristic

## Quick Example

```python
from solvor import astar, astar_grid

# A* with custom heuristic
def heuristic(node):
    # Straight-line distance to goal
    return distance_to_goal(node)

result = astar(start, goal, neighbors, heuristic)

# A* on 2D grid (built-in)
maze = [
    [0, 0, 1, 0],
    [0, 0, 0, 0],
    [1, 1, 1, 0],
    [0, 0, 0, 0]
]
result = astar_grid(maze, start=(0,0), goal=(3,3), blocked=1)
print(result.solution)  # Path through the maze
```

## See Also
- [Cookbook: Shortest Path Grid](Cookbook-Shortest-Path-Grid)

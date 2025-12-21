# Cookbook: Shortest Path in a Grid

Navigate a 2D grid maze from start to goal using A*.

## Example

```python
from solvor import astar_grid

# 0 = walkable, 1 = wall
maze = [
    [0, 0, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 1],
    [0, 0, 0, 1, 0, 0],
    [1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0]
]

result = astar_grid(
    maze,
    start=(0, 0),
    goal=(5, 5),
    blocked=1,
    directions=8  # 8-directional movement
)

if result.solution:
    print(f"Path: {result.solution}")
    print(f"Length: {result.objective}")
    
    # Visualize
    for i, row in enumerate(maze):
        for j, cell in enumerate(row):
            if (i, j) in result.solution:
                print('*', end='')
            elif cell == 1:
                print('#', end='')
            else:
                print('.', end='')
        print()
else:
    print("No path found!")
```

## See Also
- [`astar_grid`](astar-&-astar_grid)
- [Pathfinding](Pathfinding)

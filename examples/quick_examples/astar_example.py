"""
A* Search Example

Find shortest path in grid with obstacles.
Shows both astar_grid (simple) and astar (flexible) approaches.
"""

from solvor import astar, astar_grid

# 5x5 grid, 1 = blocked
grid = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0],
]

# Method 1: astar_grid (recommended for 2D grids)
result = astar_grid(grid, start=(0, 0), goal=(4, 4), blocked=1)
print("astar_grid:")
print(f"  Path: {result.solution}")
print(f"  Cost: {result.objective}")

# Method 2: astar with custom neighbors/heuristic (for non-grid problems)
GOAL = (4, 4)


def neighbors(pos):
    """4-directional neighbors with unit cost."""
    row, col = pos
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = row + dr, col + dc
        if 0 <= nr < 5 and 0 <= nc < 5 and grid[nr][nc] == 0:
            yield (nr, nc), 1


def heuristic(pos):
    """Manhattan distance to goal."""
    return abs(pos[0] - GOAL[0]) + abs(pos[1] - GOAL[1])


result = astar((0, 0), GOAL, neighbors, heuristic)
print("\nastar (custom):")
print(f"  Path: {result.solution}")
print(f"  Cost: {result.objective}")

"""
Grid Pathfinding - Comparing BFS, Dijkstra, and A*

Navigate a 2D grid with obstacles from start to goal.
Compare unweighted (BFS), weighted (Dijkstra), and heuristic (A*) search.

This is a fundamental problem in:
- Video game AI (NPC navigation)
- Robotics (path planning)
- Logistics (warehouse routing)

Source: Classic computer science problem
        Hart, Nilsson & Raphael (1968) "A* algorithm"
        https://en.wikipedia.org/wiki/A*_search_algorithm

Why compare: BFS is optimal for unweighted grids, Dijkstra for weighted,
A* uses heuristics to focus search and typically expands fewer nodes.
"""

from time import perf_counter

from solvor import astar, bfs, dijkstra

# 20x20 grid with obstacles (1 = wall, 0 = passable)
# S = Start (0, 0), G = Goal (19, 19)
GRID = [
    "S0000000000000000000",
    "01111111110000000000",
    "00000000010000000000",
    "00000000010000111110",
    "00000000010000100000",
    "00000000010000100000",
    "00000000000000100000",
    "00000011111111100000",
    "00000010000000000000",
    "00000010000000000000",
    "00000010000011111100",
    "00000010000010000000",
    "00000010000010000000",
    "00000000000010000000",
    "00111111111110000000",
    "00000000000000000000",
    "00000011111111111110",
    "00000010000000000000",
    "00000010000000000000",
    "00000000000000000000G",
]

HEIGHT = len(GRID)
WIDTH = len(GRID[0].replace("S", "0").replace("G", "0"))


def is_passable(row: int, col: int) -> bool:
    """Check if cell is passable."""
    if 0 <= row < HEIGHT and 0 <= col < WIDTH:
        cell = GRID[row][col]
        return cell != "1"
    return False


def get_neighbors(state: tuple[int, int]) -> list[tuple[int, int]]:
    """Get passable 4-connected neighbors (for BFS)."""
    row, col = state
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = row + dr, col + dc
        if is_passable(nr, nc):
            neighbors.append((nr, nc))
    return neighbors


def get_neighbors_weighted(state: tuple[int, int]) -> list[tuple[tuple[int, int], float]]:
    """Get neighbors with unit cost (for Dijkstra/A*)."""
    return [(n, 1.0) for n in get_neighbors(state)]


def manhattan_distance(state: tuple[int, int], goal: tuple[int, int]) -> float:
    """Manhattan distance heuristic."""
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])


def main():
    start = (0, 0)
    goal = (HEIGHT - 1, WIDTH - 1)

    print("Grid Pathfinding Comparison")
    print(f"  Grid size: {HEIGHT}x{WIDTH}")
    print(f"  Start: {start}")
    print(f"  Goal: {goal}")
    print()

    # BFS (unweighted)
    t0 = perf_counter()
    result_bfs = bfs(start, lambda s: s == goal, get_neighbors)
    t_bfs = perf_counter() - t0

    print("BFS (unweighted):")
    print(f"  Path length: {len(result_bfs.solution) - 1} steps")
    print(f"  Nodes explored: {result_bfs.evaluations}")
    print(f"  Time: {t_bfs * 1000:.2f}ms")
    print()

    # Dijkstra (weighted, unit costs)
    t0 = perf_counter()
    result_dijkstra = dijkstra(start, goal, get_neighbors_weighted)
    t_dijkstra = perf_counter() - t0

    print("Dijkstra (weighted, unit costs):")
    print(f"  Path length: {len(result_dijkstra.solution) - 1} steps")
    print(f"  Path cost: {result_dijkstra.objective:.0f}")
    print(f"  Nodes explored: {result_dijkstra.evaluations}")
    print(f"  Time: {t_dijkstra * 1000:.2f}ms")
    print()

    # A* with Manhattan heuristic
    t0 = perf_counter()
    result_astar = astar(
        start,
        lambda s: s == goal,
        get_neighbors_weighted,
        lambda s: manhattan_distance(s, goal),
    )
    t_astar = perf_counter() - t0

    print("A* (Manhattan heuristic):")
    print(f"  Path length: {len(result_astar.solution) - 1} steps")
    print(f"  Path cost: {result_astar.objective:.0f}")
    print(f"  Nodes explored: {result_astar.evaluations}")
    print(f"  Time: {t_astar * 1000:.2f}ms")
    print()

    # Visualize path
    print("Path visualization (. = path, # = wall, space = open):")
    path_set = set(result_astar.solution)
    for row in range(HEIGHT):
        line = ""
        for col in range(WIDTH):
            if (row, col) == start:
                line += "S"
            elif (row, col) == goal:
                line += "G"
            elif (row, col) in path_set:
                line += "."
            elif GRID[row][col] == "1":
                line += "#"
            else:
                line += " "
        print("  " + line)


if __name__ == "__main__":
    main()

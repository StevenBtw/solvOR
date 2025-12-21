# Pathfinding

Finding the shortest path from A to B is one of the oldest and most studied problems in computer science. Whether you're navigating a maze, routing packets, or planning a road trip, these algorithms have you covered.

## Solvers in this Category

### [bfs](bfs)
Breadth-first search for unweighted graphs. Explores level by level, guarantees shortest path in terms of number of edges. The simplest pathfinding algorithm.

**Complexity:** O(V + E)
**Guarantees:** Optimal for unweighted graphs

### [dfs](dfs)
Depth-first search. Explores deeply before backtracking. Finds *a* path, not necessarily the shortest. Useful for connectivity, reachability, topological sorting.

**Complexity:** O(V + E)
**Guarantees:** Finds a path if one exists (not necessarily shortest)

### [dijkstra](dijkstra)
Shortest path for weighted graphs with non-negative edges. The foundation for A*, greedily expands the closest unvisited node.

**Complexity:** O((V + E) log V) with binary heap
**Guarantees:** Optimal for non-negative weights

### [astar & astar_grid](astar-&-astar_grid)
A* search with heuristics. Like Dijkstra but with a compass: prioritizes directions toward the goal. `astar_grid` provides built-in heuristics for 2D grids.

**Complexity:** O((V + E) log V) best case, problem-dependent
**Guarantees:** Optimal with admissible heuristic

### [bellman_ford](bellman_ford)
Shortest paths with negative edge weights. Checks every edge V-1 times, detects negative cycles. The true edgelord.

**Complexity:** O(VE)
**Guarantees:** Optimal, detects negative cycles

### [floyd_warshall](floyd_warshall)
All-pairs shortest paths. Computes distances between every pair of nodes at once. Handles negative edges, but O(V³) means it's for smaller graphs.

**Complexity:** O(V³)
**Guarantees:** All-pairs distances in one shot

## When to Use This Category

**Perfect for:**
- Navigation (GPS, game AI, robotics)
- Network routing (internet packets, delivery routes)
- Maze solving and puzzle games
- Reachability queries ("can I get from A to B?")
- Finding connected components
- Dependency resolution (if edges = dependencies)

**The magic:** These are some of the most well-studied algorithms in CS. They're fast, reliable, and have decades of optimizations.

## When NOT to Use This Category

- **Traveling salesman** → Use metaheuristics or specialized TSP algorithms
- **Multiple destinations** → Depending on the problem, consider flow algorithms or TSP
- **Optimization beyond path length** → May need custom objectives with metaheuristics

## Comparing Solvers

| Solver | Edge Weights | Output | Use When |
|--------|--------------|--------|----------|
| `bfs` | Unweighted (or all weight 1) | Single path | Simplest case, mazes, grid navigation |
| `dfs` | Unweighted | Some path (not shortest) | Connectivity, cycle detection, topological sort |
| `dijkstra` | Non-negative | Single source, shortest paths | Weighted graphs, road networks |
| `astar` | Non-negative | Single path to goal | Have heuristic, goal-directed search |
| `astar_grid` | Non-negative (2D grid) | Single path to goal | 2D grid mazes, tile-based games |
| `bellman_ford` | Any (including negative) | Single source | Negative edges, detect negative cycles |
| `floyd_warshall` | Any (including negative) | All-pairs distances | Need all distances, smaller graphs |

## Quick Example

```python
from solvor import bfs, dijkstra, astar, astar_grid

# BFS: Unweighted graph
graph = {
    'A': ['B', 'C'],
    'B': ['D'],
    'C': ['D'],
    'D': []
}
result = bfs('A', 'D', lambda n: graph[n])
print(result.solution)  # ['A', 'B', 'D'] or ['A', 'C', 'D']

# Dijkstra: Weighted graph (neighbor, cost) pairs
graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('C', 2), ('D', 5)],
    'C': [('D', 1)],
    'D': []
}
result = dijkstra('A', 'D', lambda n: graph[n])
print(result.solution)  # ['A', 'B', 'C', 'D']
print(result.objective)  # 4

# A*: With heuristic (straight-line distance)
def heuristic(node):
    coords = {'A': (0,0), 'B': (1,0), 'C': (1,1), 'D': (2,1)}
    goal = coords['D']
    pos = coords[node]
    return ((pos[0]-goal[0])**2 + (pos[1]-goal[1])**2)**0.5

result = astar('A', 'D', lambda n: graph[n], heuristic)
print(result.solution)  # ['A', 'B', 'C', 'D']

# A* Grid: 2D maze
maze = [
    [0, 0, 1, 0],
    [0, 0, 0, 0],
    [1, 1, 1, 0],
    [0, 0, 0, 0]
]
result = astar_grid(maze, start=(0,0), goal=(3,3), blocked=1)
print(result.solution)  # Path like [(0,0), (0,1), (1,1), ...]
```

## Tips & Tricks

### BFS/DFS
- **Use BFS for shortest paths** - DFS doesn't guarantee shortest
- **DFS for memory efficiency** - O(depth) vs O(breadth) memory
- **Both find connected components** - Just don't pass a goal

### Dijkstra
- **Requires non-negative edges** - Negative edges break the greedy property
- **Single-source optimal** - Computes shortest paths from one start to all reachable nodes
- **Priority queue is critical** - Use heapq, not sorted lists

### A*
- **Heuristic must be admissible** - Never overestimate distance to goal, or you lose optimality
- **Manhattan distance for grids** - |dx| + |dy| for 4-directional movement
- **Octile distance for 8-directions** - max(|dx|, |dy|) + (√2-1) × min(|dx|, |dy|)
- **Weighted A* trades optimality for speed** - weight > 1 expands fewer nodes, may not be optimal

### Bellman-Ford
- **Only use if you have negative edges** - Otherwise Dijkstra is much faster
- **Detects negative cycles** - Returns UNBOUNDED status if one exists
- **Relaxes all edges V-1 times** - Simple but slow

### Floyd-Warshall
- **O(V³) limits graph size** - Works well up to ~500 nodes, gets slow after that
- **Precompute all distances** - Useful when you need many queries
- **Transitive closure** - Can also compute reachability (can I get from any i to any j?)

## Heuristics for A*

**2D Grids:**
- **Manhattan:** |dx| + |dy| - For 4-directional movement
- **Euclidean:** √(dx² + dy²) - Straight-line distance
- **Octile:** max(|dx|, |dy|) + (√2-1) × min(|dx|, |dy|) - For 8-directional
- **Chebyshev:** max(|dx|, |dy|) - For 8-directional when diagonal = horizontal cost

**General Graphs:**
- **Straight-line distance**, if you have coordinates
- **Landmark distances**, precompute distances to a few "landmark" nodes
- **Zero heuristic**, degrades to Dijkstra (still optimal, just slower)

## Real-World Applications

- **GPS Navigation:** Road networks with Dijkstra/A*
- **Game AI:** Pathfinding for NPCs (A* on tile grids)
- **Network Routing:** Internet packet routing (Dijkstra variants)
- **Robotics:** Motion planning around obstacles
- **Dependency Resolution:** Package managers, build systems (DFS for topological sort)

## Dijkstra's Negativity Was Legendary...

Just not in his algorithm. Dijkstra's algorithm assumes non-negative edge weights because it greedily commits to paths. With negative edges, a path that looks optimal might get beaten by a later detour through a negative edge.

**Example where Dijkstra fails:**
```
A --5--> B
|        |
1        -10
|        |
v        v
C --------> D
```

Dijkstra from A might commit to A→B (cost 5) before discovering A→C→B (cost 1-10 = -9 to C, but doesn't help reach B).

**Solution:** Use Bellman-Ford for negative edges.

## See Also

- [Network Flow & MST](Network-Flow-&-MST) - Related graph algorithms
- [Metaheuristics](Metaheuristics) - For traveling salesman (visiting all nodes)
- [Cookbook: Shortest Path Grid](Cookbook-Shortest-Path-Grid) - Full A* grid example

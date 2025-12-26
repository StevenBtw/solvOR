# Shortest Paths

Algorithms for finding shortest paths in weighted graphs.

## dijkstra

Shortest path for non-negative edge weights. Greedily expands the closest unvisited node.

```python
from solvor import dijkstra

graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('C', 2), ('D', 5)],
    'C': [('D', 1)],
    'D': []
}

result = dijkstra('A', 'D', lambda n: graph[n])
print(result.solution)  # ['A', 'B', 'C', 'D']
print(result.objective)  # 4
```

**Complexity:** O((V + E) log V)
**Guarantees:** Optimal for non-negative weights

## astar

A* search with heuristic. Faster than Dijkstra when you have a good distance estimate.

```python
from solvor import astar

def heuristic(node):
    coords = {'A': (0,0), 'B': (1,0), 'C': (1,1), 'D': (2,1)}
    goal = coords['D']
    pos = coords[node]
    return ((pos[0]-goal[0])**2 + (pos[1]-goal[1])**2)**0.5

result = astar('A', 'D', lambda n: graph[n], heuristic)
```

**Guarantees:** Optimal with admissible heuristic (never overestimates)

## astar_grid

A* for 2D grids with built-in heuristics.

```python
from solvor import astar_grid

maze = [
    [0, 0, 1, 0],
    [0, 0, 0, 0],
    [1, 1, 1, 0],
    [0, 0, 0, 0]
]

result = astar_grid(maze, start=(0, 0), goal=(3, 3), blocked=1)
print(result.solution)  # Path coordinates
```

## bellman_ford

Handles negative edge weights. Detects negative cycles.

```python
from solvor import bellman_ford

edges = [(0, 1, 4), (0, 2, 5), (1, 2, -3), (2, 3, 4)]
result = bellman_ford(n_nodes=4, edges=edges, start=0)
print(result.solution)  # Distances from node 0
```

**Complexity:** O(VE)
**Guarantees:** Optimal, detects negative cycles

## floyd_warshall

All-pairs shortest paths. O(V³) but gives everything at once.

```python
from solvor import floyd_warshall

edges = [(0, 1, 3), (1, 2, 1), (0, 2, 6)]
result = floyd_warshall(n_nodes=3, edges=edges)
print(result.solution[0][2])  # Shortest 0→2 = 4
```

**Complexity:** O(V³)

## Comparison

| Algorithm | Edge Weights | Output | Complexity |
|-----------|--------------|--------|------------|
| dijkstra | Non-negative | Single source | O((V+E) log V) |
| astar | Non-negative | Single path | Problem-dependent |
| bellman_ford | Any | Single source | O(VE) |
| floyd_warshall | Any | All-pairs | O(V³) |

## See Also

- [Pathfinding](pathfinding.md) - Unweighted graphs
- [Cookbook: Shortest Path Grid](../../cookbook/shortest-path-grid.md)

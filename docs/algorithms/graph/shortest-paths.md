# Shortest Paths

Algorithms for finding shortest paths in weighted graphs. For unweighted graphs, see [Pathfinding (BFS/DFS)](pathfinding.md).

## dijkstra

[Dijkstra's algorithm](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm) for non-negative edge weights. The classic greedy shortest path algorithm.

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

**How it works:** Maintain a priority queue of nodes by tentative distance. Always expand the closest unvisited node, with non-negative weights, this node's distance is final. Update neighbors and repeat until you reach the goal.

The key insight: if all edges are non-negative, the shortest path to the closest unexplored node can't possibly go through unexplored territory (that would only add distance). So we can "lock in" each node's distance as we visit it.

**Complexity:** O((V + E) log V) with a binary heap, where V = nodes, E = edges
**Guarantees:** Optimal for non-negative weights

## astar

[A* search](https://en.wikipedia.org/wiki/A*_search_algorithm) with heuristic. Faster than Dijkstra when you have a good distance estimate.

```python
from solvor import astar

def heuristic(node):
    coords = {'A': (0,0), 'B': (1,0), 'C': (1,1), 'D': (2,1)}
    goal = coords['D']
    pos = coords[node]
    return ((pos[0]-goal[0])**2 + (pos[1]-goal[1])**2)**0.5

result = astar('A', 'D', lambda n: graph[n], heuristic)
```

**Guarantees:** Optimal with admissible heuristic (never overestimates).

**How it works:** A* is Dijkstra with direction. Instead of expanding nodes by distance alone, it prioritizes by f(n) = g(n) + h(n):

- g(n): actual cost from start to n (known)
- h(n): estimated cost from n to goal (heuristic)

**The admissibility requirement:** The heuristic must never overestimate. If h(n) ≤ true_distance(n, goal) for all n, A* finds the optimal path. Common admissible heuristics:

- Manhattan distance: |dx| + |dy| (4-directional movement)
- Euclidean distance: √(dx² + dy²) (any direction)
- Octile distance: max(|dx|, |dy|) + (√2−1)·min(|dx|, |dy|) (8-directional)

**Why it's faster:** Dijkstra explores uniformly in all directions. A* uses the heuristic to focus exploration toward the goal. With a perfect heuristic (h = true distance), A* goes straight to the goal. With h=0, A* degrades to Dijkstra.

**Weighted A*:** Setting weight > 1 makes h more aggressive. Faster but may find suboptimal paths (bounded by weight factor).

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

[Bellman-Ford algorithm](https://en.wikipedia.org/wiki/Bellman%E2%80%93Ford_algorithm). Handles negative edge weights and detects negative cycles.

```python
from solvor import bellman_ford

edges = [(0, 1, 4), (0, 2, 5), (1, 2, -3), (2, 3, 4)]
result = bellman_ford(n_nodes=4, edges=edges, start=0)
print(result.solution)  # Distances from node 0
```

**How it works:** Relax all edges V−1 times (where V is the number of nodes). "Relaxing" an edge means: if going through this edge gives a shorter path, update the distance.

Each pass guarantees we've found shortest paths using at most k edges. After V−1 passes, we've found all shortest paths, because a simple path visits each node at most once, so it uses at most V−1 edges.

To detect negative cycles: do one more pass. If any distance still improves, there's a negative cycle, you can keep going around it forever, reducing the distance infinitely.

**Complexity:** O(V × E) where V = nodes, E = edges
**Guarantees:** Optimal, detects negative cycles

## floyd_warshall

[Floyd-Warshall algorithm](https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm). All-pairs shortest paths in O(V³), giving you every shortest path at once.

```python
from solvor import floyd_warshall

edges = [(0, 1, 3), (1, 2, 1), (0, 2, 6)]
result = floyd_warshall(n_nodes=3, edges=edges)
print(result.solution[0][2])  # Shortest 0→2 = 4
```

**How it works:** The key insight is beautifully simple: build up shortest paths by considering intermediate nodes one at a time.

**The recurrence:** Let dist\[i\]\[j\]\[k\] = shortest path from i to j using only nodes {0, 1, ..., k-1} as intermediates. Then:

```text
dist[i][j][k] = min(
    dist[i][j][k-1],           # don't use node k
    dist[i][k][k-1] + dist[k][j][k-1]  # go through node k
)
```

Either the shortest path avoids node k entirely, or it goes i→k→j.

**The algorithm:** We don't need to store all k values—just update in place:

```text
for k in 0..n:
    for i in 0..n:
        for j in 0..n:
            dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
```

After considering all k, dist\[i\]\[j\] is the shortest path using any intermediate nodes.

**Negative cycle detection:** If dist\[i\]\[i\] < 0 for any i after the algorithm, there's a negative cycle (you can reach i from i with negative cost).

**When to use:** All-pairs queries, small dense graphs, or when you need the full distance matrix. For sparse graphs or single-source, Dijkstra/Bellman-Ford are faster.

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

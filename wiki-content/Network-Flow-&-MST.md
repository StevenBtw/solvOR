# Network Flow & MST

Graphs aren't just about finding paths, sometimes you need to push stuff through them (flow) or connect them all at minimum cost (spanning trees). These algorithms handle resource routing, bottleneck analysis, and network design.

## Solvers in this Category

### [max_flow](max_flow)
Maximum flow from source to sink. "How much can I push through this network?" Uses Edmonds-Karp (BFS-based augmenting paths). Finds bottlenecks for free via max-flow min-cut theorem.

**Complexity:** O(VE²)
**Guarantees:** Optimal maximum flow

### [min_cost_flow](min_cost_flow)
Minimum cost flow. "Route this much flow at minimum total cost." Uses successive shortest paths. Extends max flow with per-edge costs.

**Complexity:** O(demand × Bellman-Ford)
**Guarantees:** Optimal minimum cost routing

### [network_simplex](network_simplex)
Network simplex for min-cost flow. Like regular simplex but exploits flow network structure. Uses spanning trees instead of dense tableaus. Much faster on large networks.

**Complexity:** O(V²E) typical, exponential worst case
**Guarantees:** Optimal minimum cost routing

### [solve_assignment](solve_assignment)
Assignment via min-cost flow. Match workers to tasks optimally using flow formulation. Simple wrapper around min_cost_flow.

**Complexity:** O(n³) via flow reduction
**Guarantees:** Optimal assignment

### [kruskal](kruskal)
Minimum spanning tree. Greedily picks cheapest edges that don't create cycles. Uses union-find for near-O(1) cycle detection. One of those rare greedy = optimal algorithms.

**Complexity:** O(E log E) for sorting edges
**Guarantees:** Optimal MST

### [prim](prim)
Minimum spanning tree by growing a tree from a start node. Like Dijkstra but for MST. Returns same result as Kruskal, different approach.

**Complexity:** O((V + E) log V) with binary heap
**Guarantees:** Optimal MST

## When to Use This Category

**Perfect for:**
- **Flow:** Resource routing, matching, bottleneck analysis
- **MST:** Network cabling, clustering, circuit layout
- **Assignment:** Worker-task matching, server-request allocation
- **Transportation:** Shipping goods at minimum cost
- **Capacity planning:** Finding network bottlenecks

**The magic:** These algorithms exploit graph structure for efficiency. They're polynomial where naive approaches would be exponential.

## When NOT to Use This Category

- **Shortest paths** → Use Dijkstra, A*, or Bellman-Ford
- **Traveling salesman** → Use metaheuristics or specialized TSP solvers
- **General optimization** → Use MILP or constraint programming

## Comparing Solvers

### Flow Algorithms

| Solver | Objective | Use When |
|--------|-----------|----------|
| `max_flow` | Maximize throughput | Finding maximum capacity, bottlenecks |
| `min_cost_flow` | Minimize cost for fixed demand | Transportation, cost matters |
| `network_simplex` | Minimize cost for fixed demand | Large networks, need speed |
| `solve_assignment` | Optimal matching | Bipartite matching, worker-task allocation |

**Flow rule of thumb:** Small network or simple needs? Use `max_flow` or `min_cost_flow`. Large network with costs? Use `network_simplex`.

### MST Algorithms

| Solver | Approach | Use When |
|--------|----------|----------|
| `kruskal` | Edge-centric, union-find | Sparse graphs, all edges available |
| `prim` | Node-centric, priority queue | Dense graphs, neighbor iteration |

Both guarantee the same optimal MST. Kruskal is simpler, Prim integrates better with adjacency list representations.

## Quick Example

```python
from solvor import max_flow, min_cost_flow, kruskal, prim, hungarian

# Max flow: How much can flow from source to sink?
graph = {
    's': [('a', 10, 0), ('b', 5, 0)],  # (neighbor, capacity, cost)
    'a': [('t', 5, 0), ('b', 15, 0)],
    'b': [('t', 10, 0)],
    't': []
}
result = max_flow(graph, source='s', sink='t')
print(result.objective)  # 15 (total flow)
print(result.solution)  # {('s','a'): 10, ('s','b'): 5, ...}

# Min cost flow: Route 10 units at minimum cost
graph_with_costs = {
    's': [('a', 10, 2), ('b', 10, 3)],  # (neighbor, capacity, cost)
    'a': [('t', 10, 1)],
    'b': [('t', 10, 1)],
    't': []
}
result = min_cost_flow(graph_with_costs, source='s', sink='t', demand=10)
print(result.objective)  # Total cost

# MST: Connect all nodes at minimum cost
n_nodes = 4
edges = [(0, 1, 4), (0, 2, 3), (1, 2, 2), (1, 3, 5), (2, 3, 6)]
result = kruskal(n_nodes, edges)
print(result.solution)  # [(1,2,2), (0,2,3), (0,1,4)] or similar
print(result.objective)  # 9 (total weight)
```

## Tips & Tricks

### Max Flow
- **Max-flow = min-cut**, the maximum flow equals the minimum capacity cut
- **Finding bottlenecks**, look at saturated edges in the solution
- **Bipartite matching**, model as flow: source → left nodes → right nodes → sink

### Min Cost Flow
- **Supply/demand formulation**, nodes can produce or consume flow (see network_simplex)
- **Not just point-to-point**, can model multiple sources/sinks
- **Successive shortest paths**, the algorithm finds cheapest augmenting paths

### Network Simplex
- **Balanced flow required**, sum of supplies must equal sum of demands
- **Scales better**, use this for large networks instead of min_cost_flow
- **Artificial arcs**, the algorithm adds high-cost artificial arcs to get an initial feasible tree

### Kruskal vs Prim
- **Kruskal:** Sort all edges first, pick cheapest that don't create cycles
- **Prim:** Grow tree from one node, always add cheapest edge to a new node
- **Same result:** Both find the minimum spanning tree

### Assignment as Flow
- `solve_assignment` is just min_cost_flow in disguise
- For large assignments, `hungarian` algorithm is faster (O(n³))
- Use flow formulation when you need additional constraints

## The Max-Flow Min-Cut Theorem

**One of the most beautiful results in graph theory:**

The maximum amount of flow you can push from source to sink equals the minimum capacity of any cut separating them.

**Why it matters:**
- Solving max flow gives you the bottleneck for free
- Cut capacity = sum of edge capacities crossing from source side to sink side
- Practical use: network capacity planning, vulnerability analysis

## Real-World Applications

### Flow
- **Bipartite Matching:** Jobs to workers, students to projects
- **Network Throughput:** Maximum packets through a network
- **Image Segmentation:** Min-cut for foreground/background separation
- **Supply Chain:** Shipping goods from warehouses to stores

### MST
- **Network Design:** Laying fiber optic cable to connect cities
- **Clustering:** Stop early at k-1 edges = k clusters
- **Circuit Board Layout:** Minimum wire length to connect components
- **Approximate TSP:** MST gives a 2-approximation lower bound

## Advanced Topics

### Flow Variants
- **Multi-commodity flow:** Multiple flow types sharing capacity
- **Circulation:** No source/sink, flow circulates
- **Parametric flow:** How does max flow change as capacities scale?

### MST Variants
- **Steiner tree:** Connect subset of nodes at minimum cost (NP-hard)
- **Degree-constrained MST:** MST where node degrees are bounded
- **Dynamic MST:** Update MST as edges are added/removed

## See Also

- [Assignment](Assignment) - Dedicated assignment solvers
- [Pathfinding](Pathfinding) - Shortest paths instead of flows
- [Linear & Integer Programming](Linear-&-Integer-Programming) - Flow can be modeled as LP
- [Cookbook: Max Flow Network](Cookbook-Max-Flow-Network) - Full flow example

# Centrality & Structural Analysis

Algorithms for measuring node importance and identifying critical structures in graphs.

## pagerank

[PageRank](https://en.wikipedia.org/wiki/PageRank) measures node importance based on link structure. Nodes linked to by many important nodes get higher scores.

```python
from solvor import pagerank

# Web pages linking to each other
links = {
    "home": ["about", "products"],
    "about": ["home", "contact"],
    "products": ["home"],
    "contact": ["home"],
}

result = pagerank(links.keys(), lambda n: links.get(n, []))
for page, score in sorted(result.solution.items(), key=lambda x: -x[1]):
    print(f"{page}: {score:.3f}")
# home has highest rank (many pages link to it)
```

**How it works:** Iteratively updates each node's score as a weighted sum of scores from nodes linking to it, plus a damping factor allowing random jumps. Converges when scores stop changing. Equivalent to finding the dominant eigenvector of the transition matrix.

**Parameters:**

- `damping`: Probability of following links vs random jump (default 0.85)
- `max_iter`: Maximum iterations (default 100)
- `tol`: Convergence tolerance (default 1e-6)

**Complexity:** O(E × iterations) where E = edges

## articulation_points

Find [articulation points](https://en.wikipedia.org/wiki/Biconnected_component) (cut vertices) whose removal disconnects the graph.

```python
from solvor import articulation_points

network = {
    "server1": ["router", "server2"],
    "server2": ["router", "server1"],
    "router": ["server1", "server2", "database"],
    "database": ["router"],
}

result = articulation_points(network.keys(), lambda n: network.get(n, []))
print(result.solution)  # {'router'} - single point of failure
```

**How it works:** Uses Tarjan's algorithm with DFS, tracking discovery times and low-links. A node is an articulation point if any subtree cannot reach back above it. O(V + E) time.

## bridges

Find [bridges](https://en.wikipedia.org/wiki/Bridge_(graph_theory)) (cut edges) whose removal disconnects the graph.

```python
from solvor import bridges

result = bridges(network.keys(), lambda n: network.get(n, []))
for edge in result.solution:
    print(f"Critical: {edge[0]} -- {edge[1]}")
```

**How it works:** Similar to articulation points. An edge (u, v) is a bridge if the subtree rooted at v cannot reach u's ancestors. O(V + E) time.

## kcore_decomposition

[K-core decomposition](https://en.wikipedia.org/wiki/Degeneracy_(graph_theory)) assigns each node a core number—the largest k for which it belongs to the k-core. Higher = more central.

```python
from solvor import kcore_decomposition

# Collaboration network
collabs = {
    "alice": ["bob", "carol", "dave"],
    "bob": ["alice", "carol", "dave"],
    "carol": ["alice", "bob", "dave"],
    "dave": ["alice", "bob", "carol", "eve"],
    "eve": ["dave"],
}

result = kcore_decomposition(collabs.keys(), lambda n: collabs.get(n, []))
# alice, bob, carol, dave have core 3 (tight group)
# eve has core 1 (peripheral)
```

**How it works:** Iteratively remove nodes with degree < k, updating remaining degrees. A node's core number is the last k at which it survived. O(V + E) time with bucket sort.

## kcore

Extract the k-core subgraph (nodes with core number ≥ k).

```python
from solvor import kcore

core_team = kcore(collabs.keys(), lambda n: collabs.get(n, []), k=3)
print(core_team.solution)  # {'alice', 'bob', 'carol', 'dave'}
```

## When to Use

| Algorithm | Use When |
|-----------|----------|
| `pagerank` | Finding influential nodes, ranking by importance |
| `articulation_points` | Finding single points of failure |
| `bridges` | Finding critical connections |
| `kcore_decomposition` | Core vs periphery analysis |
| `kcore` | Extracting densely connected subgraphs |

## See Also

- [Dependency Analysis](dependency-analysis.md) - Cycle detection, topological sort
- [Community Detection](community-detection.md) - Clustering nodes into groups

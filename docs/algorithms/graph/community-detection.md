# Community Detection

Algorithms for finding clusters of densely connected nodes.

## louvain

[Louvain algorithm](https://en.wikipedia.org/wiki/Louvain_method) for community detection. Finds groups of nodes that are more densely connected to each other than to the rest of the graph.

```python
from solvor import louvain

# Social network
friendships = {
    "alice": ["bob", "carol"],
    "bob": ["alice", "carol"],
    "carol": ["alice", "bob", "dave"],
    "dave": ["carol", "eve", "frank"],
    "eve": ["dave", "frank"],
    "frank": ["dave", "eve"],
}

result = louvain(friendships.keys(), lambda n: friendships.get(n, []))
for community in result.solution:
    print(f"Community: {sorted(community)}")
# Two communities: alice/bob/carol and dave/eve/frank
print(f"Modularity: {result.objective:.3f}")
```

**How it works:** Optimizes modularity in two phases:

1. **Local moves:** Each node moves to the community that gives the largest modularity gain
2. **Aggregation:** Build a new graph where communities become nodes

Repeat until no improvement. Modularity measures the fraction of edges within communities minus the expected fraction if edges were random.

**Parameters:**

- `resolution`: Controls community size (default 1.0). Higher values produce smaller communities.

**Complexity:** Near-linear in practice, O(n log n) for most graphs

## Tips

- **Resolution parameter:** Start with default (1.0). Increase for finer-grained communities, decrease for larger ones.
- **Undirected graphs:** Louvain treats the graph as undirected. For directed community detection, consider other methods.
- **Determinism:** Results may vary slightly between runs due to node ordering. For reproducible results, sort nodes first.

## See Also

- [Centrality](centrality.md) - PageRank, k-core for node importance
- [Dependency Analysis](dependency-analysis.md) - Strongly connected components

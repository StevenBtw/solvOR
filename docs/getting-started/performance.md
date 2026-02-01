# Performance

solvOR gives you both readable Python implementations and optional Rust backends for performance-critical algorithms.

## Design Philosophy

**Have your cake and eat it too:**

- **Readable Python** - Every algorithm is implemented in clear, documented Python you can study, debug, and modify
- **Rust Performance** - Optional drop-in Rust backends with 5-60x speedup for compute-heavy algorithms

The Rust extensions are pre-built for Linux, macOS, and Windows. If Rust isn't available, solvOR falls back to Python automatically.

## The `backend` Parameter

All Rust-accelerated functions accept a `backend` parameter:

```python
from solvor import floyd_warshall

# "auto" (default) - Uses Rust if available, falls back to Python
result = floyd_warshall(n_nodes, edges)

# "python" - Force pure Python (for learning, debugging, or extending)
result = floyd_warshall(n_nodes, edges, backend="python")

# "rust" - Force Rust (raises error if Rust not available)
result = floyd_warshall(n_nodes, edges, backend="rust")
```

### When to Use Each Backend

| Backend | Use Case |
|---------|----------|
| `"auto"` | Default for most users, best performance available |
| `"python"` | Learning how the algorithm works, debugging, or extending |
| `"rust"` | Strict performance requirements, CI validation |

## Algorithms with Rust Backends

The following algorithms have Rust implementations:

| Algorithm | Function | Typical Speedup | Notes |
|-----------|----------|-----------------|-------|
| Floyd-Warshall | `floyd_warshall` | **45-60x** | All-pairs shortest paths |
| Bellman-Ford | `bellman_ford` | **10-20x** | Negative weight edges |
| Dijkstra | `dijkstra_edges` | **5-10x** | Edge-list variant only |
| BFS | `bfs_edges` | **3-5x** | Edge-list variant only |
| DFS | `dfs_edges` | **3-5x** | Edge-list variant only |
| PageRank | `pagerank_edges` | **10-15x** | Edge-list variant only |
| SCC | `strongly_connected_components_edges` | **5-10x** | Edge-list variant only |
| Topological Sort | `topological_sort_edges` | **5-10x** | Edge-list variant only |
| Kruskal MST | `kruskal` | **5-10x** | Minimum spanning tree |

## Two API Styles

Some graph algorithms have two variants:

### Callback-based (Flexible)

Works with any hashable node type - strings, tuples, objects, custom classes:

```python
from solvor import dijkstra, bfs, pagerank

# Nodes can be strings
graph = {"A": [("B", 1), ("C", 4)], "B": [("C", 2)], "C": []}
result = dijkstra("A", "C", lambda n: graph.get(n, []))

# Nodes can be tuples (grid coordinates)
def get_neighbors(pos):
    x, y = pos
    return [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

result = bfs((0, 0), (5, 5), get_neighbors)

# Works with any hashable type
result = pagerank(nodes, lambda n: adjacency[n])
```

These functions are **pure Python** and prioritize flexibility and readability.

### Edge-list (`*_edges`) - Fast

Integer nodes 0..n-1 with edge lists - optimized for performance:

```python
from solvor import dijkstra_edges, bfs_edges, pagerank_edges

# Nodes are integers 0 to n_nodes-1
edges = [(0, 1, 3), (1, 2, 1), (0, 2, 5)]  # (from, to, weight)
result = dijkstra_edges(n_nodes=3, edges=edges, source=0, target=2)

# Unweighted edges for BFS
edges = [(0, 1), (1, 2), (0, 2)]  # (from, to)
result = bfs_edges(n_nodes=3, edges=edges, source=0, target=2)

# PageRank on edge list
result = pagerank_edges(n_nodes=100, edges=edges)
```

These functions use **Rust backends when available** for maximum performance.

!!! tip "Which to Choose?"
    - Use **callback-based** when your nodes aren't integers or when you want to understand/modify the algorithm
    - Use **edge-list** (`*_edges`) when you need performance on large graphs with integer nodes

## Benchmarks

Measured on a typical laptop (Macbook Pro M1, 16GB RAM):

| Algorithm | Graph Size | Python | Rust | Speedup |
|-----------|------------|--------|------|---------|
| `floyd_warshall` | 400 nodes | 4.0s | 69ms | **58x** |
| `pagerank_edges` | 2000 nodes, 8000 edges | 36ms | 3.3ms | **11x** |
| `dijkstra_edges` | 500 nodes, 2000 edges | 4.9ms | 1.0ms | **5x** |
| `bellman_ford` | 500 nodes, 2000 edges | 15ms | 1.2ms | **12x** |
| `kruskal` | 1000 nodes, 5000 edges | 8ms | 1.5ms | **5x** |

## Checking Backend Availability

```python
from solvor._rust import RUST_AVAILABLE, get_backend

# Check if Rust extension is available
print(f"Rust available: {RUST_AVAILABLE}")

# See which backend will be used
print(f"Auto selects: {get_backend('auto')}")  # "rust" or "python"
```

## Building from Source with Rust

If you want to build the Rust extension yourself:

```bash
# Install Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone and build
git clone https://github.com/StevenBtw/solvOR.git
cd solvOR
uv sync
uv run maturin develop --release
```

The `maturin develop` command compiles the Rust code and installs it as a Python extension.

## Pure Python Fallback

Even without Rust, solvOR is fully functional. The Python implementations are:

- **Complete** - Every feature works without Rust
- **Readable** - Clear code you can study and learn from
- **Correct** - Same results as Rust, just slower
- **Debuggable** - Step through with any Python debugger

This makes solvOR ideal for:

- Learning algorithms
- Prototyping and experimentation
- Environments where installing Rust isn't practical
- Extending and customizing algorithms

# Maximum Flow

Find the maximum flow through a network from source to sink.

## Example

```python
from solvor import max_flow

# Network: s -> a,b -> t
graph = {
    's': [('a', 10, 0), ('b', 5, 0)],
    'a': [('t', 5, 0), ('b', 15, 0)],
    'b': [('t', 10, 0)],
    't': []
}

result = max_flow(graph, source='s', sink='t')

print(f"Maximum flow: {result.objective}")
print(f"Flow on edges: {result.solution}")
```

## Applications

- **Network capacity** - Maximum data throughput
- **Bipartite matching** - Assign workers to tasks
- **Image segmentation** - Min-cut max-flow duality

## See Also

- [Network Flow](../algorithms/graph/network-flow.md)

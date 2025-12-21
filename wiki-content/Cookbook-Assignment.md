# Cookbook: Assignment Problem

Optimally assign workers to tasks minimizing total cost.

## Complete Example

```python
from solvor import hungarian

# Cost matrix: cost[worker][task]
costs = [
    [9, 2, 7, 8],   # Worker 0
    [6, 4, 3, 7],   # Worker 1
    [5, 8, 1, 8],   # Worker 2
    [7, 6, 9, 4]    # Worker 3
]

result = hungarian(costs)

print(f"Assignment: {result.solution}")
print(f"Total cost: {result.objective}")

# Decode assignment
for worker, task in enumerate(result.solution):
    cost = costs[worker][task]
    print(f"Worker {worker} -> Task {task} (cost {cost})")
```

## See Also
- [`hungarian`](hungarian)
- [Assignment](Assignment)

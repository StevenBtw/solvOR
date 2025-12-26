# Assignment Problem

Optimally assign workers to tasks minimizing total cost.

## The Problem

Given n workers and n tasks with a cost matrix, assign each worker to exactly one task (and vice versa) to minimize total cost.

## Example

```python
from solvor import solve_hungarian

# Cost matrix: cost[worker][task]
costs = [
    [9, 2, 7, 8],   # Worker 0
    [6, 4, 3, 7],   # Worker 1
    [5, 8, 1, 8],   # Worker 2
    [7, 6, 9, 4]    # Worker 3
]

result = solve_hungarian(costs)

print(f"Assignment: {result.solution}")
print(f"Total cost: {result.objective}")

# Decode assignment
for worker, task in enumerate(result.solution):
    cost = costs[worker][task]
    print(f"Worker {worker} -> Task {task} (cost {cost})")
```

## Maximization

For maximization (e.g., profit instead of cost):

```python
result = solve_hungarian(profits, minimize=False)
```

## See Also

- [Resource Allocation](resource-allocation.md) - When workers can do multiple tasks

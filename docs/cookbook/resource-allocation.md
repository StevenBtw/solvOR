# Resource Allocation

Allocate limited resources to tasks to maximize value using MILP.

## The Problem

Assign workers to tasks where each worker has limited capacity, maximizing total value.

## Example

```python
from solvor import solve_milp

# 3 workers, 4 tasks
# values[worker][task] = value if worker does task
values = [
    [10, 5, 8, 12],
    [7, 9, 6, 11],
    [8, 6, 10, 9]
]

n_workers, n_tasks = 3, 4
n_vars = n_workers * n_tasks

# Objective: maximize sum of values
c = [-values[i][j] for i in range(n_workers) for j in range(n_tasks)]

A = []
b = []

# Each task assigned to exactly one worker
for j in range(n_tasks):
    row = []
    for i in range(n_workers):
        for j2 in range(n_tasks):
            row.append(1 if j2 == j else 0)
    A.append(row)
    b.append(1)

# Each worker does at most 2 tasks
for i in range(n_workers):
    row = []
    for i2 in range(n_workers):
        for j in range(n_tasks):
            row.append(1 if i2 == i else 0)
    A.append(row)
    b.append(2)

result = solve_milp(c, A, b, integers=list(range(n_vars)), minimize=False)
print(f"Total value: {result.objective}")
```

## See Also

- [Assignment Problem](assignment.md) - One-to-one matching
- [solve_milp](../algorithms/linear-programming/solve-milp.md)

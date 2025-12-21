# Cookbook: Resource Allocation

Allocate limited resources to tasks to maximize value using MILP.

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

# Each task needs exactly 1 worker
# Maximize total value

# Variables: x[i][j] = 1 if worker i does task j
# 12 variables total (3*4)

n_workers, n_tasks = 3, 4
n_vars = n_workers * n_tasks

# Objective: maximize sum of values
c = [-values[i][j] for i in range(n_workers) for j in range(n_tasks)]

# Constraints: each task assigned to exactly one worker
A = []
b = []
for j in range(n_tasks):  # For each task
    row = [1 if (i * n_tasks + j2 == idx) and j2 == j else 0 
           for i in range(n_workers) for j2 in range(n_tasks) 
           for idx in range(n_vars)][:n_vars]
    row = []
    for i in range(n_workers):
        for j2 in range(n_tasks):
            row.append(1 if j2 == j else 0)
    A.append(row)
    b.append(1)

# Each worker does at most 2 tasks (capacity constraint)
for i in range(n_workers):
    row = []
    for i2 in range(n_workers):
        for j in range(n_tasks):
            row.append(1 if i2 == i else 0)
    A.append(row)
    b.append(2)

# Binary variables, upper bounds
for var in range(n_vars):
    row = [1 if i == var else 0 for i in range(n_vars)]
    A.append(row)
    b.append(1)

result = solve_milp(c, A, b, integers=list(range(n_vars)), minimize=False)
print(f"Allocation: {result.solution}")
print(f"Total value: {result.objective}")
```

## See Also
- [`solve_milp`](solve_milp)
- [Linear & Integer Programming](Linear-&-Integer-Programming)

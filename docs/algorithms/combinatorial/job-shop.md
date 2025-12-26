# solve_job_shop

Job shop scheduling. Minimize makespan for jobs on machines.

## Example

```python
from solvor import solve_job_shop

# jobs[i] = [(machine, duration), ...] - operations for job i
jobs = [
    [(0, 3), (1, 2), (2, 2)],  # Job 0: machine 0 for 3, then machine 1 for 2, etc.
    [(0, 2), (2, 1), (1, 4)],
    [(1, 4), (2, 3)]
]

result = solve_job_shop(jobs)
print(result.objective)  # Makespan
print(result.solution)   # Schedule
```

## The Problem

Each job consists of ordered operations. Each operation runs on a specific machine for a duration. Find a schedule that:

- Respects operation order within jobs
- No two operations on the same machine overlap
- Minimizes total time (makespan)

## Complexity

NP-hard. Uses constraint-based approach.

## See Also

- [Cookbook: Job Shop](../../cookbook/job-shop.md) - Full example
- [Nurse Scheduling](../../cookbook/nurse-scheduling.md) - Related scheduling

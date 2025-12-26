# Job Shop Scheduling

Schedule jobs on machines to minimize total completion time (makespan).

Each job has a sequence of operations, each requiring a specific machine for a specific duration. A machine can only process one operation at a time.

## The Problem

- n jobs, each with a sequence of operations
- m machines
- Each operation needs a specific machine for a specific time
- Operations in a job must run in order
- Goal: minimize makespan (when the last operation finishes)

## Example

```python
from solvor import solve_job_shop

# Job = [(machine, duration), ...]
jobs = [
    [(0, 3), (1, 2), (2, 2)],  # Job 0: M0 for 3, then M1 for 2, then M2 for 2
    [(0, 2), (2, 1), (1, 4)],  # Job 1
    [(1, 4), (2, 3)],          # Job 2
]

result = solve_job_shop(jobs, rule="spt", local_search=True)
print(f"Makespan: {result.objective}")

# Show schedule
for (job, op), (start, end) in sorted(result.solution.items()):
    machine = jobs[job][op][0]
    print(f"  Job {job} Op {op}: M{machine} @ {start}-{end}")
```

## Dispatching Rules

- `spt` - Shortest Processing Time first
- `lpt` - Longest Processing Time first
- `mwkr` - Most Work Remaining
- `fifo` - First In First Out

Use `local_search=True` to improve the initial dispatch solution.

## See Also

- [Nurse Scheduling](nurse-scheduling.md)

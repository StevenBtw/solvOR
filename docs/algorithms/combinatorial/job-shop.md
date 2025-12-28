# solve_job_shop

Job shop scheduling. Minimize makespan for jobs on machines. Uses dispatching rules with local search.

## Signature

```python
def solve_job_shop(
    jobs: Sequence[Job],
    *,
    rule: str = "spt",
    local_search: bool = True,
    max_iter: int = 1000,
    seed: int | None = None,
    on_progress: ProgressCallback | None = None,
    progress_interval: int = 0,
) -> Result[dict[tuple[int, int], tuple[int, int]]]
```

Where `Job = Sequence[tuple[int, int]]` (list of (machine, duration) tuples).

## Parameters

| Parameter | Description |
|-----------|-------------|
| `jobs` | List of jobs, each job is a sequence of (machine, duration) operations |
| `rule` | Dispatching rule: `"spt"` (shortest processing time), `"lpt"` (longest), `"fifo"`, `"mwkr"` (most work remaining), `"random"` |
| `local_search` | If True, improve initial schedule with swap moves |
| `max_iter` | Maximum local search iterations |
| `seed` | Random seed for reproducibility |
| `on_progress` | Progress callback (return True to stop early) |
| `progress_interval` | Call progress every N iterations (0 = disabled) |

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

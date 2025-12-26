"""
Job Shop Scheduling Example

Schedule jobs with machine-specific operations to minimize makespan.
"""

from solvor import solve_job_shop

# Jobs: list of (machine, duration) tuples for each operation
jobs = [
    [(0, 3), (1, 2)],  # Job 0
    [(1, 4), (0, 2)],  # Job 1
    [(0, 2), (1, 3)],  # Job 2
]

result = solve_job_shop(jobs, rule="spt", local_search=True, seed=42)
print(f"Schedule: {result.solution}")
print(f"Makespan: {result.objective}")

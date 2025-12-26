"""
Job Shop Scheduling - Fisher & Thompson ft06 Benchmark

Classic 6x6 job shop instance: 6 jobs, 6 machines, 36 operations.
One of the most studied benchmark instances in scheduling research.

Source: Fisher & Thompson (1963), later used in:
        - Muth & Thompson (1963) "Industrial Scheduling"
        - Adams, Balas & Zawack (1988)
        http://jobshop.jjvh.nl/

Optimal makespan: 55 time units

Problem: Each job consists of 6 operations that must be processed in order.
Each operation requires a specific machine for a specific duration.
No machine can process more than one operation at a time.
Goal: Minimize makespan (completion time of last operation).

Why this solver: Dispatching rules + local search is fast and effective
for medium instances. For proven optimal, use CP-SAT formulation.
"""

from time import perf_counter

from solvor import solve_job_shop

# ft06 instance: 6 jobs x 6 machines
# Format: job = [(machine, duration), (machine, duration), ...]
# Machine indices are 0-based

FT06_JOBS = [
    # Job 0
    [(2, 1), (0, 3), (1, 6), (3, 7), (5, 3), (4, 6)],
    # Job 1
    [(1, 8), (2, 5), (4, 10), (5, 10), (0, 10), (3, 4)],
    # Job 2
    [(2, 5), (3, 4), (5, 8), (0, 9), (1, 1), (4, 7)],
    # Job 3
    [(1, 5), (0, 5), (2, 5), (3, 3), (4, 8), (5, 9)],
    # Job 4
    [(2, 9), (1, 3), (4, 5), (5, 4), (0, 3), (3, 1)],
    # Job 5
    [(1, 3), (3, 3), (5, 9), (0, 10), (4, 4), (2, 1)],
]

OPTIMAL_MAKESPAN = 55
N_JOBS = 6
N_MACHINES = 6


def main():
    print("Job Shop Scheduling - Fisher & Thompson ft06")
    print(f"  Jobs: {N_JOBS}")
    print(f"  Machines: {N_MACHINES}")
    print(f"  Operations: {sum(len(job) for job in FT06_JOBS)}")
    print(f"  Optimal makespan: {OPTIMAL_MAKESPAN}")
    print()

    # Try different dispatching rules
    rules = ["fifo", "spt", "lpt", "mwkr", "random"]

    print("Dispatching rules (no local search):")
    for rule in rules:
        result = solve_job_shop(FT06_JOBS, rule=rule, local_search=False, seed=42)
        gap = (result.objective - OPTIMAL_MAKESPAN) / OPTIMAL_MAKESPAN * 100
        print(f"  {rule:8s}: makespan = {int(result.objective)} (gap: {gap:+.1f}%)")

    print()
    print("With local search improvement:")

    start = perf_counter()
    result = solve_job_shop(
        FT06_JOBS,
        rule="spt",
        local_search=True,
        max_iter=1000,
        seed=42,
    )
    elapsed = perf_counter() - start

    makespan = int(result.objective)
    gap = (makespan - OPTIMAL_MAKESPAN) / OPTIMAL_MAKESPAN * 100

    print(f"  SPT + local search: makespan = {makespan} (gap: {gap:+.1f}%)")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Iterations: {result.iterations}")
    print(f"  Evaluations: {result.evaluations}")

    # Print schedule
    schedule = result.solution
    print()
    print("Schedule (job, op): (start, end)")

    # Group by machine
    by_machine = {m: [] for m in range(N_MACHINES)}
    for (job, op), (start, end) in schedule.items():
        machine = FT06_JOBS[job][op][0]
        by_machine[machine].append((start, end, job, op))

    for m in range(N_MACHINES):
        ops = sorted(by_machine[m])
        ops_str = ", ".join(f"J{j}O{o}@{s}-{e}" for s, e, j, o in ops)
        print(f"  Machine {m}: {ops_str}")


if __name__ == "__main__":
    main()

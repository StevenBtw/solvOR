"""
Job Shop Scheduling solver using dispatching rules and local search.

Classic scheduling problem: n jobs, each with a sequence of operations on
different machines. Each operation has a specific machine and duration.
Goal is to minimize makespan (total completion time).

    from solvor import solve_job_shop

    jobs = [
        [(0, 3), (1, 2), (2, 2)],  # Job 0: machine 0 for 3, machine 1 for 2, machine 2 for 2
        [(0, 2), (2, 1), (1, 4)],  # Job 1: machine 0 for 2, machine 2 for 1, machine 1 for 4
    ]
    result = solve_job_shop(jobs)
    print(result.solution)  # Schedule with start times

Use this for:
- Manufacturing scheduling
- Production planning
- Resource allocation with precedence constraints

For optimal solutions on small instances (<10 jobs), consider using CP-SAT
with the encoding in this module. For larger instances, dispatching rules
with local search provide good approximate solutions.
"""

from collections.abc import Sequence
from random import Random

from solvor.types import Progress, ProgressCallback, Result, Status

__all__ = ["solve_job_shop"]

# Type aliases
Operation = tuple[int, int]  # (machine, duration)
Job = Sequence[Operation]


def solve_job_shop(
    jobs: Sequence[Job],
    *,
    rule: str = "spt",
    local_search: bool = True,
    max_iter: int = 1000,
    seed: int | None = None,
    on_progress: ProgressCallback | None = None,
    progress_interval: int = 0,
) -> Result:
    """Solve job shop scheduling using dispatching rules and local search."""
    n_jobs = len(jobs)
    if n_jobs == 0:
        return Result({}, 0.0, 0, 0, Status.OPTIMAL)

    # Validate and count machines
    n_machines = 0
    for j, job in enumerate(jobs):
        if not job:
            raise ValueError(f"Job {j} has no operations")
        for op_idx, (machine, duration) in enumerate(job):
            if machine < 0:
                raise ValueError(f"Job {j} operation {op_idx} has negative machine index")
            if duration < 0:
                raise ValueError(f"Job {j} operation {op_idx} has negative duration")
            n_machines = max(n_machines, machine + 1)

    rng = Random(seed)
    evals = 0

    # Generate initial schedule using dispatching rule
    schedule = _dispatch(jobs, n_machines, rule, rng)
    makespan = _compute_makespan(jobs, schedule)
    evals += 1

    best_schedule = schedule
    best_makespan = makespan

    if not local_search:
        return Result(best_schedule, float(best_makespan), 0, evals, Status.FEASIBLE)

    # Local search: swap adjacent operations on same machine
    no_improve = 0
    max_no_improve = 100

    for iteration in range(1, max_iter + 1):
        # Try random swap on random machine
        improved = False
        machine = rng.randrange(n_machines)

        # Get operations on this machine in current order
        ops_on_machine = []
        for j, job in enumerate(jobs):
            for op_idx, (m, _) in enumerate(job):
                if m == machine:
                    ops_on_machine.append((j, op_idx))

        if len(ops_on_machine) < 2:
            continue

        # Sort by start time
        ops_on_machine.sort(key=lambda x: schedule[(x[0], x[1])][0])

        # Try swapping adjacent pairs
        for i in range(len(ops_on_machine) - 1):
            j1, op1 = ops_on_machine[i]
            j2, op2 = ops_on_machine[i + 1]

            # Check if swap is valid (doesn't violate job precedence)
            new_schedule = _try_swap(jobs, schedule, j1, op1, j2, op2)
            if new_schedule is None:
                continue

            new_makespan = _compute_makespan(jobs, new_schedule)
            evals += 1

            if new_makespan < makespan:
                schedule = new_schedule
                makespan = new_makespan
                improved = True

                if makespan < best_makespan:
                    best_schedule = schedule
                    best_makespan = makespan
                break

        if improved:
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= max_no_improve:
            break

        if on_progress and progress_interval > 0 and iteration % progress_interval == 0:
            progress = Progress(iteration, float(best_makespan), None, evals)
            if on_progress(progress) is True:
                return Result(best_schedule, float(best_makespan), iteration, evals, Status.FEASIBLE)

    return Result(best_schedule, float(best_makespan), iteration, evals, Status.FEASIBLE)


def _dispatch(
    jobs: Sequence[Job],
    n_machines: int,
    rule: str,
    rng: Random,
) -> dict[tuple[int, int], tuple[int, int]]:
    """Generate schedule using dispatching rule. Returns {(job, op): (start, end)}."""
    n_jobs = len(jobs)

    # Track next operation index for each job
    next_op = [0] * n_jobs

    # Track when each machine becomes free
    machine_free = [0] * n_machines

    # Track when each job's last operation ends
    job_free = [0] * n_jobs

    schedule: dict[tuple[int, int], tuple[int, int]] = {}

    # Compute total remaining work for each job (for MWKR rule)
    remaining_work = [sum(d for _, d in job) for job in jobs]

    total_ops = sum(len(job) for job in jobs)

    for _ in range(total_ops):
        # Find ready operations (next operation of each job that's ready)
        ready = []
        for j in range(n_jobs):
            if next_op[j] < len(jobs[j]):
                machine, duration = jobs[j][next_op[j]]
                ready.append((j, next_op[j], machine, duration))

        if not ready:
            break

        # Select operation based on rule
        rule_lower = rule.lower()
        if rule_lower == "fifo":
            # First ready (by job index)
            selected = ready[0]
        elif rule_lower == "spt":
            # Shortest Processing Time
            selected = min(ready, key=lambda x: x[3])
        elif rule_lower == "lpt":
            # Longest Processing Time
            selected = max(ready, key=lambda x: x[3])
        elif rule_lower == "mwkr":
            # Most Work Remaining
            selected = max(ready, key=lambda x: remaining_work[x[0]])
        elif rule_lower == "random":
            selected = rng.choice(ready)
        else:
            raise ValueError(f"Unknown dispatching rule: {rule}")

        j, op_idx, machine, duration = selected

        # Schedule this operation
        start = max(machine_free[machine], job_free[j])
        end = start + duration

        schedule[(j, op_idx)] = (start, end)

        # Update state
        machine_free[machine] = end
        job_free[j] = end
        next_op[j] += 1
        remaining_work[j] -= duration

    return schedule


def _compute_makespan(
    jobs: Sequence[Job],
    schedule: dict[tuple[int, int], tuple[int, int]],
) -> int:
    """Compute makespan (maximum end time) of a schedule."""
    return max(end for _, end in schedule.values()) if schedule else 0


def _try_swap(
    jobs: Sequence[Job],
    schedule: dict[tuple[int, int], tuple[int, int]],
    j1: int,
    op1: int,
    j2: int,
    op2: int,
) -> dict[tuple[int, int], tuple[int, int]] | None:
    """Try swapping two operations on the same machine. Returns new schedule or None."""
    # This is a simplified swap - we rebuild the schedule with swapped priority
    # A full implementation would do critical path analysis

    machine = jobs[j1][op1][0]
    assert jobs[j2][op2][0] == machine

    # Get all operations on this machine
    ops_on_machine = []
    for j, job in enumerate(jobs):
        for op_idx, (m, _) in enumerate(job):
            if m == machine:
                ops_on_machine.append((j, op_idx, schedule[(j, op_idx)][0]))

    # Sort by original start time
    ops_on_machine.sort(key=lambda x: x[2])

    # Find positions to swap
    pos1 = pos2 = -1
    for i, (j, op_idx, _) in enumerate(ops_on_machine):
        if j == j1 and op_idx == op1:
            pos1 = i
        if j == j2 and op_idx == op2:
            pos2 = i

    if pos1 == -1 or pos2 == -1 or abs(pos1 - pos2) != 1:
        return None

    # Swap in the order
    ops_on_machine[pos1], ops_on_machine[pos2] = ops_on_machine[pos2], ops_on_machine[pos1]

    # Rebuild schedule with new order constraint
    new_schedule = _rebuild_schedule(jobs, schedule, machine, ops_on_machine)
    return new_schedule


def _rebuild_schedule(
    jobs: Sequence[Job],
    old_schedule: dict[tuple[int, int], tuple[int, int]],
    target_machine: int,
    machine_order: list[tuple[int, int, int]],
) -> dict[tuple[int, int], tuple[int, int]]:
    """Rebuild schedule respecting new machine order."""
    n_jobs = len(jobs)
    n_machines = max(jobs[j][op][0] for j in range(n_jobs) for op in range(len(jobs[j]))) + 1

    # Track constraints
    machine_free = [0] * n_machines
    job_free = [0] * n_jobs

    new_schedule: dict[tuple[int, int], tuple[int, int]] = {}

    # Process all operations in topological order respecting:
    # 1. Job precedence (earlier ops before later in same job)
    # 2. New machine order for target_machine

    # Build operation list with dependencies
    all_ops = []
    for j, job in enumerate(jobs):
        for op_idx in range(len(job)):
            all_ops.append((j, op_idx))

    # Sort by: job precedence first, then by specified machine order for target machine
    machine_order_map = {(j, op): i for i, (j, op, _) in enumerate(machine_order)}

    def op_priority(op_tuple):
        j, op_idx = op_tuple
        machine = jobs[j][op_idx][0]
        # Primary: operation index in job (precedence)
        # Secondary: position in machine order if on target machine
        if machine == target_machine:
            return (op_idx, machine_order_map.get((j, op_idx), 0))
        return (op_idx, old_schedule[(j, op_idx)][0])

    # Schedule operations greedily
    scheduled = set()
    while len(scheduled) < len(all_ops):
        # Find ready operations
        ready = []
        for j, op_idx in all_ops:
            if (j, op_idx) in scheduled:
                continue
            # Check job precedence
            if op_idx > 0 and (j, op_idx - 1) not in scheduled:
                continue
            ready.append((j, op_idx))

        if not ready:
            break

        # Sort ready ops by priority
        ready.sort(key=op_priority)

        # Schedule first ready
        j, op_idx = ready[0]
        machine, duration = jobs[j][op_idx]

        start = max(machine_free[machine], job_free[j])
        end = start + duration

        new_schedule[(j, op_idx)] = (start, end)
        machine_free[machine] = end
        job_free[j] = end
        scheduled.add((j, op_idx))

    return new_schedule

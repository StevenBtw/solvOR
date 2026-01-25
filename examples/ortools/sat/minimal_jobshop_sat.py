"""
OR-Tools Example: minimal_jobshop_sat.py

This is a solvOR implementation of the Google OR-Tools example.
Original: https://github.com/google/or-tools/blob/stable/ortools/sat/samples/minimal_jobshop_sat.py

solvOR provides pure Python solvers for learning and prototyping.
For production-scale problems, consider using Google OR-Tools which
offers compiled C++ solvers with significantly better performance.

Comparison:
- solvOR: Pure Python, readable, educational, no dependencies
- OR-Tools: C++ backend, production-ready, 10-100x faster
"""

#!/usr/bin/env python3
# Copyright 2010-2025 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START program]
"""Minimal jobshop example."""
# [START import]
import collections

from solvor import Status, solve_job_shop

# [END import]


def main() -> None:
    """Minimal jobshop problem."""
    # Data.
    # [START data]
    jobs_data = [  # task = (machine_id, processing_time).
        [(0, 3), (1, 2), (2, 2)],  # Job0
        [(0, 2), (2, 1), (1, 4)],  # Job1
        [(1, 4), (2, 3)],  # Job2
    ]

    machines_count = 1 + max(task[0] for job in jobs_data for task in job)
    all_machines = range(machines_count)
    # [END data]

    # [START solve]
    # Solve using solvOR's job shop solver
    result = solve_job_shop(jobs_data, rule="spt", local_search=True, max_iter=1000)
    # [END solve]

    # [START print_solution]
    if result.status == Status.OPTIMAL or result.status == Status.FEASIBLE:
        print("Solution:")
        # Named tuple to manipulate solution information.
        assigned_task_type = collections.namedtuple("assigned_task_type", "start job index duration")

        # Create one list of assigned tasks per machine.
        assigned_jobs = collections.defaultdict(list)
        for job_id, job in enumerate(jobs_data):
            for task_id, task in enumerate(job):
                machine = task[0]
                start, _ = result.solution[(job_id, task_id)]
                assigned_jobs[machine].append(
                    assigned_task_type(
                        start=start,
                        job=job_id,
                        index=task_id,
                        duration=task[1],
                    )
                )

        # Create per machine output lines.
        output = ""
        for machine in all_machines:
            # Sort by starting time.
            assigned_jobs[machine].sort()
            sol_line_tasks = "Machine " + str(machine) + ": "
            sol_line = "           "

            for assigned_task in assigned_jobs[machine]:
                name = f"job_{assigned_task.job}_task_{assigned_task.index}"
                # add spaces to output to align columns.
                sol_line_tasks += f"{name:15}"

                start = assigned_task.start
                duration = assigned_task.duration
                sol_tmp = f"[{start},{start + duration}]"
                # add spaces to output to align columns.
                sol_line += f"{sol_tmp:15}"

            sol_line += "\n"
            sol_line_tasks += "\n"
            output += sol_line_tasks
            output += sol_line

        # Finally print the solution found.
        print(f"Optimal Schedule Length: {int(result.objective)}")
        print(output)
    else:
        print("No solution found.")
    # [END print_solution]

    # Statistics.
    # [START statistics]
    print("\nStatistics")
    print(f"  - iterations: {result.iterations}")
    print(f"  - evaluations: {result.evaluations}")
    # [END statistics]


if __name__ == "__main__":
    main()
# [END program]

"""
OR-Tools Example: assignment_task_sizes_mip.py

This is a solvOR implementation of the Google OR-Tools example.
Original: https://github.com/google/or-tools/blob/stable/ortools/linear_solver/samples/assignment_task_sizes_mip.py

solvOR provides pure Python solvers for learning and prototyping.
For production-scale problems, consider using Google OR-Tools which
offers compiled C++ solvers with significantly better performance.

Comparison:
- solvOR: Pure Python, readable, educational, no dependencies
- OR-Tools: C++ backend, production-ready, 10-100x faster

Note: This is a Generalized Assignment Problem (GAP) where workers can handle
multiple tasks up to a capacity limit. Uses MILP with 80 binary variables.
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
"""MIP example that solves an assignment problem."""
# [START import]
from solvor import Status, solve_milp

# [END import]


def main():
    # Data
    # [START data]
    costs = [
        [90, 76, 75, 70, 50, 74, 12, 68],
        [35, 85, 55, 65, 48, 101, 70, 83],
        [125, 95, 90, 105, 59, 120, 36, 73],
        [45, 110, 95, 115, 104, 83, 37, 71],
        [60, 105, 80, 75, 59, 62, 93, 88],
        [45, 65, 110, 95, 47, 31, 81, 34],
        [38, 51, 107, 41, 69, 99, 115, 48],
        [47, 85, 57, 71, 92, 77, 109, 36],
        [39, 63, 97, 49, 118, 56, 92, 61],
        [47, 101, 71, 60, 88, 109, 52, 90],
    ]
    num_workers = len(costs)
    num_tasks = len(costs[0])

    task_sizes = [10, 7, 3, 12, 15, 4, 11, 5]
    # Maximum total of task sizes for any worker
    total_size_max = 15
    # [END data]

    # [START solver]
    # Variables: x[worker, task] flattened to worker * num_tasks + task
    n_vars = num_workers * num_tasks  # 80 binary variables

    def x_idx(worker, task):
        return worker * num_tasks + task

    print("Solving with solvOR MILP")
    # [END solver]

    # [START constraints]
    A = []
    b_rhs = []

    # The total size of tasks each worker takes on is at most total_size_max
    for worker in range(num_workers):
        row = [0] * n_vars
        for task in range(num_tasks):
            row[x_idx(worker, task)] = task_sizes[task]
        A.append(row)
        b_rhs.append(total_size_max)

    # Each task is assigned to exactly one worker: sum_w x[w,t] == 1
    for task in range(num_tasks):
        # sum <= 1
        row = [0] * n_vars
        for worker in range(num_workers):
            row[x_idx(worker, task)] = 1
        A.append(row)
        b_rhs.append(1)
        # sum >= 1 → -sum <= -1
        row = [0] * n_vars
        for worker in range(num_workers):
            row[x_idx(worker, task)] = -1
        A.append(row)
        b_rhs.append(-1)
    # [END constraints]

    # [START objective]
    # Minimize total cost
    c = [0] * n_vars
    for worker in range(num_workers):
        for task in range(num_tasks):
            c[x_idx(worker, task)] = costs[worker][task]
    # [END objective]

    # [START solve]
    integers = list(range(n_vars))  # All variables are binary
    # Use branch and bound - simpler constraints work well with B&B
    result = solve_milp(c, A, b_rhs, integers, minimize=True, max_nodes=100000)
    # [END solve]

    # [START print_solution]
    if result.status in (Status.OPTIMAL, Status.FEASIBLE):
        print(f"Total cost = {result.objective}\n")
        for worker in range(num_workers):
            for task in range(num_tasks):
                if result.solution[x_idx(worker, task)] > 0.5:
                    print(f"Worker {worker} assigned to task {task}." + f" Cost: {costs[worker][task]}")
    else:
        print("No solution found.")
    # [END print_solution]


if __name__ == "__main__":
    main()
# [END program]

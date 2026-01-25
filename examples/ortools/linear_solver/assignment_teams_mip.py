"""
OR-Tools Example: assignment_teams_mip.py

This is a solvOR implementation of the Google OR-Tools example.
Original: https://github.com/google/or-tools/blob/stable/ortools/linear_solver/samples/assignment_teams_mip.py

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
"""MIP example that solves an assignment problem."""
# [START import]
from solvor import Status, solve_milp

# [END import]


def main():
    # Data
    # [START data]
    costs = [
        [90, 76, 75, 70],
        [35, 85, 55, 65],
        [125, 95, 90, 105],
        [45, 110, 95, 115],
        [60, 105, 80, 75],
        [45, 65, 110, 95],
    ]
    num_workers = len(costs)
    num_tasks = len(costs[0])

    team1 = [0, 2, 4]
    team2 = [1, 3, 5]
    # Maximum total of tasks for any team
    team_max = 2
    # [END data]

    # [START solver]
    # Variables: x[worker, task] flattened
    n_vars = num_workers * num_tasks  # 24 binary variables

    def x_idx(worker, task):
        return worker * num_tasks + task

    print("Solving with solvOR MILP")
    # [END solver]

    # [START constraints]
    A = []
    b_rhs = []

    # Each worker is assigned at most 1 task
    for worker in range(num_workers):
        row = [0] * n_vars
        for task in range(num_tasks):
            row[x_idx(worker, task)] = 1
        A.append(row)
        b_rhs.append(1)

    # Each task is assigned to exactly one worker: sum == 1
    for task in range(num_tasks):
        # sum <= 1
        row = [0] * n_vars
        for worker in range(num_workers):
            row[x_idx(worker, task)] = 1
        A.append(row)
        b_rhs.append(1)
        # sum >= 1
        row = [0] * n_vars
        for worker in range(num_workers):
            row[x_idx(worker, task)] = -1
        A.append(row)
        b_rhs.append(-1)

    # Each team takes at most team_max tasks
    # Team 1
    row = [0] * n_vars
    for worker in team1:
        for task in range(num_tasks):
            row[x_idx(worker, task)] = 1
    A.append(row)
    b_rhs.append(team_max)

    # Team 2
    row = [0] * n_vars
    for worker in team2:
        for task in range(num_tasks):
            row[x_idx(worker, task)] = 1
    A.append(row)
    b_rhs.append(team_max)
    # [END constraints]

    # [START objective]
    c = [0] * n_vars
    for worker in range(num_workers):
        for task in range(num_tasks):
            c[x_idx(worker, task)] = costs[worker][task]
    # [END objective]

    # [START solve]
    integers = list(range(n_vars))
    result = solve_milp(c, A, b_rhs, integers, minimize=True, max_nodes=10000)
    # [END solve]

    # [START print_solution]
    if result.status in (Status.OPTIMAL, Status.FEASIBLE):
        print(f"Total cost = {result.objective}\n")
        for worker in range(num_workers):
            for task in range(num_tasks):
                if result.solution[x_idx(worker, task)] > 0.5:
                    print(f"Worker {worker} assigned to task {task}." + f" Cost = {costs[worker][task]}")
    else:
        print("No solution found.")
    # [END print_solution]


if __name__ == "__main__":
    main()
# [END program]

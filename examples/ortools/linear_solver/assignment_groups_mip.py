"""
OR-Tools Example: assignment_groups_mip.py

This is a solvOR implementation of the Google OR-Tools example.
Original: https://github.com/google/or-tools/blob/stable/ortools/linear_solver/samples/assignment_groups_mip.py

solvOR provides pure Python solvers for learning and prototyping.
For production-scale problems, consider using Google OR-Tools which
offers compiled C++ solvers with significantly better performance.

Comparison:
- solvOR: Pure Python, readable, educational, no dependencies
- OR-Tools: C++ backend, production-ready, 10-100x faster

Note: This problem requires exactly one allowed pair from each group to work.
Solved by enumerating all 125 group configurations (5x5x5) and using Hungarian
algorithm for each 6-worker assignment. Much faster than general MILP.
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
"""Solve assignment problem for given group of workers."""
# [START import]
from solvor import solve_hungarian
# [END import]


def main():
    # Data
    # [START data]
    costs = [
        [90, 76, 75, 70, 50, 74],
        [35, 85, 55, 65, 48, 101],
        [125, 95, 90, 105, 59, 120],
        [45, 110, 95, 115, 104, 83],
        [60, 105, 80, 75, 59, 62],
        [45, 65, 110, 95, 47, 31],
        [38, 51, 107, 41, 69, 99],
        [47, 85, 57, 71, 92, 77],
        [39, 63, 97, 49, 118, 56],
        [47, 101, 71, 60, 88, 109],
        [17, 39, 103, 64, 61, 92],
        [101, 45, 83, 59, 92, 27],
    ]
    num_workers = len(costs)
    num_tasks = len(costs[0])
    # [END data]

    # Allowed groups of workers:
    # [START allowed_groups]
    group1 = [  # Subgroups of workers 0 - 3
        [2, 3],
        [1, 3],
        [1, 2],
        [0, 1],
        [0, 2],
    ]

    group2 = [  # Subgroups of workers 4 - 7
        [6, 7],
        [5, 7],
        [5, 6],
        [4, 5],
        [4, 7],
    ]

    group3 = [  # Subgroups of workers 8 - 11
        [10, 11],
        [9, 11],
        [9, 10],
        [8, 10],
        [8, 11],
    ]
    # [END allowed_groups]

    # [START solver]
    # Enumerate all 125 valid group configurations (5 × 5 × 5)
    # For each, solve 6×6 assignment with Hungarian algorithm
    print("Solving with solvOR Hungarian algorithm (enumeration)")

    best_cost = float('inf')
    best_assignment = None
    best_workers = None
    # [END solver]

    # [START solve]
    for pair1 in group1:
        for pair2 in group2:
            for pair3 in group3:
                # Selected workers from this configuration
                workers = pair1 + pair2 + pair3  # 6 workers

                # Build 6×6 cost submatrix for these workers
                submatrix = [[costs[w][t] for t in range(num_tasks)] for w in workers]

                # Solve assignment for this configuration
                result = solve_hungarian(submatrix)

                if result.objective < best_cost:
                    best_cost = result.objective
                    best_assignment = result.solution
                    best_workers = workers
    # [END solve]

    # [START print_solution]
    if best_assignment is not None:
        print(f"Total cost = {best_cost}\n")
        for i, worker in enumerate(best_workers):
            task = best_assignment[i]
            if task != -1:
                print(
                    f"Worker {worker} assigned to task {task}."
                    + f" Cost: {costs[worker][task]}"
                )
    else:
        print("No solution found.")
    # [END print_solution]


if __name__ == "__main__":
    main()
# [END program]

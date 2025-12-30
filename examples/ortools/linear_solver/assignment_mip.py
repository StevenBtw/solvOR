"""
OR-Tools Example: assignment_mip.py

This is a solvOR implementation of the Google OR-Tools example.
Original: https://github.com/google/or-tools/blob/stable/ortools/linear_solver/samples/assignment_mip.py

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

"""MIP example that solves an assignment problem."""
# [START program]
# [START import]
from solvor import solve_hungarian
# [END import]


def main():
    # Data
    # [START data_model]
    costs = [
        [90, 80, 75, 70],
        [35, 85, 55, 65],
        [125, 95, 90, 95],
        [45, 110, 95, 115],
        [50, 100, 90, 100],
    ]
    num_workers = len(costs)
    num_tasks = len(costs[0])
    # [END data_model]

    # Solver
    # [START solver]
    # solvOR uses solve_hungarian for assignment problems
    # [END solver]

    # Variables and Constraints handled internally by Hungarian algorithm
    # [START variables]
    # [END variables]
    # [START constraints]
    # [END constraints]

    # Objective
    # [START objective]
    # Hungarian algorithm minimizes total cost by default
    # [END objective]

    # Solve
    # [START solve]
    print("Solving with solvOR Hungarian algorithm")
    result = solve_hungarian(costs)
    # [END solve]

    # Print solution.
    # [START print_solution]
    if result.objective > 0:
        print(f"Total cost = {result.objective}\n")
        for i in range(num_workers):
            j = result.solution[i]
            if j != -1:  # Worker is assigned
                print(f"Worker {i} assigned to task {j}. Cost: {costs[i][j]}")
    else:
        print("No solution found.")
    # [END print_solution]


if __name__ == "__main__":
    main()
# [END program]

"""
OR-Tools Example: assignment_linear_sum_assignment.py

This is a solvOR implementation of the Google OR-Tools example.
Original: https://github.com/google/or-tools/blob/stable/ortools/graph/samples/assignment_linear_sum_assignment.py

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
"""Solve assignment problem using linear assignment solver."""
# [START import]
from solvor import solve_hungarian

# [END import]


def main():
    """Linear Sum Assignment example."""
    # [START data]
    costs = [
        [90, 76, 75, 70],
        [35, 85, 55, 65],
        [125, 95, 90, 105],
        [45, 110, 95, 115],
    ]
    # [END data]

    # [START solve]
    result = solve_hungarian(costs)
    # [END solve]

    # [START print_solution]
    if result.objective is not None:
        print(f"Total cost = {int(result.objective)}\n")
        for worker, task in enumerate(result.solution):
            print(
                f"Worker {worker} assigned to task {task}."
                + f"  Cost = {costs[worker][task]}"
            )
    else:
        print("No assignment is possible.")
    # [END print_solution]


if __name__ == "__main__":
    main()
# [END Program]

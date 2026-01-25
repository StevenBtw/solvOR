"""
OR-Tools Example: simple_lp_program.py

This is a solvOR implementation of the Google OR-Tools example.
Original: https://github.com/google/or-tools/blob/stable/ortools/linear_solver/samples/simple_lp_program.py

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

"""Minimal example to call the GLOP solver."""
# [START program]
# [START import]
from solvor import Status, solve_lp

# [END import]


def main():
    # [START solver]
    # solvOR uses solve_lp with matrix representation
    # Variables: x (index 0), y (index 1)
    # [END solver]

    # [START variables]
    # Variables are implicit in solvOR - defined by constraint matrix dimensions
    # x >= 0 and y >= 0 are default (non-negative)
    n_vars = 2
    print("Number of variables =", n_vars)
    # [END variables]

    # [START constraints]
    # x + 7 * y <= 17.5
    # x <= 3.5
    A = [
        [1, 7],  # x + 7*y <= 17.5
        [1, 0],  # x <= 3.5
    ]
    b = [17.5, 3.5]

    print("Number of constraints =", len(b))
    # [END constraints]

    # [START objective]
    # Maximize x + 10 * y
    c = [1, 10]
    # [END objective]

    # [START solve]
    print("Solving with solvOR simplex")
    result = solve_lp(c, A, b, minimize=False)
    # [END solve]

    # [START print_solution]
    if result.status == Status.OPTIMAL:
        print("Solution:")
        print("Objective value =", result.objective)
        print("x =", result.solution[0])
        print("y =", result.solution[1])
    else:
        print("The problem does not have an optimal solution.")
    # [END print_solution]

    # [START advanced]
    print("\nAdvanced usage:")
    print(f"Problem solved in {result.iterations:d} iterations")
    # [END advanced]


if __name__ == "__main__":
    main()
# [END program]

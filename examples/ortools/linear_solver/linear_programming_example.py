"""
OR-Tools Example: linear_programming_example.py

This is a solvOR implementation of the Google OR-Tools example.
Original: https://github.com/google/or-tools/blob/stable/ortools/linear_solver/samples/linear_programming_example.py

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

"""Linear optimization example."""
# [START program]
# [START import]
from solvor import Status, solve_lp

# [END import]


def LinearProgrammingExample():
    """Linear programming sample."""
    # solvOR uses solve_lp with matrix representation
    # [START solver]
    # Variables: x (index 0), y (index 1)
    # [END solver]

    # Create the two variables and let them take on any non-negative value.
    # [START variables]
    n_vars = 2
    print("Number of variables =", n_vars)
    # [END variables]

    # [START constraints]
    # Constraint 0: x + 2y <= 14.
    # Constraint 1: 3x - y >= 0  →  -3x + y <= 0
    # Constraint 2: x - y <= 2.
    A = [
        [1, 2],  # x + 2y <= 14
        [-3, 1],  # -3x + y <= 0 (flipped from 3x - y >= 0)
        [1, -1],  # x - y <= 2
    ]
    b = [14.0, 0.0, 2.0]

    print("Number of constraints =", len(b))
    # [END constraints]

    # [START objective]
    # Objective function: 3x + 4y.
    c = [3, 4]
    # [END objective]

    # Solve the system.
    # [START solve]
    print("Solving with solvOR simplex")
    result = solve_lp(c, A, b, minimize=False)
    # [END solve]

    # [START print_solution]
    if result.status == Status.OPTIMAL:
        print("Solution:")
        print(f"Objective value = {result.objective:0.1f}")
        print(f"x = {result.solution[0]:0.1f}")
        print(f"y = {result.solution[1]:0.1f}")
    else:
        print("The problem does not have an optimal solution.")
    # [END print_solution]

    # [START advanced]
    print("\nAdvanced usage:")
    print(f"Problem solved in {result.iterations:d} iterations")
    # [END advanced]


LinearProgrammingExample()
# [END program]

"""
OR-Tools Example: integer_programming_example.py

This is a solvOR implementation of the Google OR-Tools example.
Original: https://github.com/google/or-tools/blob/stable/ortools/linear_solver/samples/integer_programming_example.py

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

"""Small example to illustrate solving a MIP problem."""
# [START program]
# [START import]
from solvor import solve_milp, Status
# [END import]


def IntegerProgrammingExample():
    """Integer programming sample."""
    # [START solver]
    # solvOR uses solve_milp with matrix representation
    # Variables: x (index 0), y (index 1), z (index 2)
    # [END solver]

    # [START variables]
    # x, y, and z are non-negative integer variables.
    n_vars = 3
    integers = [0, 1, 2]  # All three are integers
    var_names = ["x", "y", "z"]

    print("Number of variables =", n_vars)
    # [END variables]

    # [START constraints]
    # 2*x + 7*y + 3*z <= 50
    # 3*x - 5*y + 7*z <= 45
    # 5*x + 2*y - 6*z <= 37
    A = [
        [2, 7, 3],    # 2x + 7y + 3z <= 50
        [3, -5, 7],   # 3x - 5y + 7z <= 45
        [5, 2, -6],   # 5x + 2y - 6z <= 37
    ]
    b = [50, 45, 37]

    print("Number of constraints =", len(b))
    # [END constraints]

    # [START objective]
    # Maximize 2*x + 2*y + 3*z
    c = [2, 2, 3]
    # [END objective]

    # Solve the problem.
    # [START solve]
    print("Solving with solvOR MILP (branch and bound)")
    result = solve_milp(c, A, b, integers, minimize=False)
    # [END solve]

    # Print the solution.
    # [START print_solution]
    if result.status == Status.OPTIMAL:
        print("Solution:")
        print(f"Objective value = {result.objective}")
        # Print the value of each variable in the solution.
        for i, name in enumerate(var_names):
            print(f"{name} = {result.solution[i]}")
    else:
        print("The problem does not have an optimal solution.")
    # [END print_solution]

    # [START advanced]
    print("\nAdvanced usage:")
    print(f"Problem solved in {result.iterations:d} iterations")
    # [END advanced]


IntegerProgrammingExample()
# [END program]

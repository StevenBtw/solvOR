"""
OR-Tools Example: basic_example.py

This is a solvOR implementation of the Google OR-Tools example.
Original: https://github.com/google/or-tools/blob/stable/ortools/linear_solver/samples/basic_example.py

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
import solvor
from solvor import solve_lp, Status
# [END import]


def main():
    print("solvOR version:", solvor.__version__)

    # [START solver]
    # solvOR uses solve_lp with matrix representation
    # Variables: x (index 0), y (index 1)
    # [END solver]

    # [START variables]
    # Create the variables x and y.
    # x in [0, 1], y in [0, 2] - bounds expressed as constraints
    n_vars = 2
    print("Number of variables =", n_vars)
    # [END variables]

    # [START constraints]
    # x + y <= 2 (original constraint)
    # x <= 1 (upper bound on x)
    # y <= 2 (upper bound on y)
    A = [
        [1, 1],  # x + y <= 2
        [1, 0],  # x <= 1
        [0, 1],  # y <= 2
    ]
    b = [2, 1, 2]

    print("Number of constraints =", 1)  # Only counting the original constraint
    # [END constraints]

    # [START objective]
    # Create the objective function, 3 * x + y.
    c = [3, 1]
    # [END objective]

    # [START solve]
    print("Solving with solvOR simplex")
    result = solve_lp(c, A, b, minimize=False)
    # [END solve]

    # [START print_solution]
    print(f"Status: {result.status.name}")
    if result.status != Status.OPTIMAL:
        print("The problem does not have an optimal solution!")
        if result.status == Status.FEASIBLE:
            print("A potentially suboptimal solution was found")
        else:
            print("The solver could not solve the problem.")
            return

    print("Solution:")
    print("Objective value =", result.objective)
    print("x =", result.solution[0])
    print("y =", result.solution[1])
    # [END print_solution]

    # [START advanced]
    print("Advanced usage:")
    print(f"Problem solved in {result.iterations:d} iterations")
    # [END advanced]


if __name__ == "__main__":
    main()
# [END program]

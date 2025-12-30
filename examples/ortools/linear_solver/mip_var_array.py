"""
OR-Tools Example: mip_var_array.py

This is a solvOR implementation of the Google OR-Tools example.
Original: https://github.com/google/or-tools/blob/stable/ortools/linear_solver/samples/mip_var_array.py

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

"""MIP example that uses a variable array."""

# [START program]
# [START import]
from solvor import solve_milp, Status
# [END import]


# [START program_part1]
# [START data_model]
def create_data_model():
    """Stores the data for the problem."""
    data = {}
    data["constraint_coeffs"] = [
        [5, 7, 9, 2, 1],
        [18, 4, -9, 10, 12],
        [4, 7, 3, 8, 5],
        [5, 13, 16, 3, -7],
    ]
    data["bounds"] = [250, 285, 211, 315]
    data["obj_coeffs"] = [7, 8, 2, 9, 6]
    data["num_vars"] = 5
    data["num_constraints"] = 4
    return data

# [END data_model]


def main():
    # [START data]
    data = create_data_model()
    # [END data]
    # [END program_part1]

    # [START solver]
    n = data["num_vars"]
    print("Number of variables =", n)
    # [END solver]

    # [START constraints]
    # RowConstraint(0, bound) means: 0 <= sum <= bound
    # Convert to: sum <= bound AND -sum <= 0
    A = []
    b = []

    for i in range(data["num_constraints"]):
        # sum <= bound
        A.append(data["constraint_coeffs"][i])
        b.append(data["bounds"][i])
        # sum >= 0 → -sum <= 0
        A.append([-c for c in data["constraint_coeffs"][i]])
        b.append(0)

    print("Number of constraints =", data["num_constraints"])
    # [END constraints]

    # [START objective]
    c = data["obj_coeffs"]
    # [END objective]

    # [START solve]
    print("Solving with solvOR MILP")
    integers = list(range(n))
    result = solve_milp(c, A, b, integers, minimize=False, max_nodes=10000)
    # [END solve]

    # [START print_solution]
    if result.status == Status.OPTIMAL:
        print("Objective value =", result.objective)
        for j in range(n):
            print(f"x[{j}]  =  {result.solution[j]}")
        print()
        print(f"Problem solved in {result.iterations} iterations")
    else:
        print("The problem does not have an optimal solution.")
    # [END print_solution]


if __name__ == "__main__":
    main()
# [END program_part2]
# [END program]

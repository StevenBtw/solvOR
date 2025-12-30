"""
OR-Tools Example: search_for_all_solutions_sample_sat.py

This is a solvOR implementation of the Google OR-Tools example.
Original: https://github.com/google/or-tools/blob/stable/ortools/sat/samples/search_for_all_solutions_sample_sat.py

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

"""Code sample that solves a model and displays all solutions."""

# [START program]
from solvor import Model, Status

# [END import]


def search_for_all_solutions_sample_sat():
    """Showcases calling the solver to search for all solutions."""
    # Creates the model.
    # [START model]
    model = Model()
    # [END model]

    # Creates the variables.
    # [START variables]
    num_vals = 3
    x = model.int_var(0, num_vals - 1, "x")
    y = model.int_var(0, num_vals - 1, "y")
    z = model.int_var(0, num_vals - 1, "z")
    # [END variables]

    # Create the constraints.
    # [START constraints]
    model.add(x != y)
    # [END constraints]

    # Create a solver and solve.
    # [START solve]
    # Enumerate all solutions (use large limit to get all)
    result = model.solve(solution_limit=1000)
    # [END solve]

    # [START print_solution]
    if result.status == Status.OPTIMAL and hasattr(result, "solutions") and result.solutions:
        for sol in result.solutions:
            print(f"x={sol['x']} y={sol['y']} z={sol['z']}")
        print(f"Status = OPTIMAL")
        print(f"Number of solutions found: {len(result.solutions)}")
    elif result.status == Status.OPTIMAL and result.solution:
        print(f"x={result.solution['x']} y={result.solution['y']} z={result.solution['z']}")
        print(f"Status = OPTIMAL")
        print(f"Number of solutions found: 1")
    else:
        print("No solution found.")
    # [END print_solution]


search_for_all_solutions_sample_sat()
# [END program]

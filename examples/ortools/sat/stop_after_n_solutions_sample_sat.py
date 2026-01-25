"""
OR-Tools Example: stop_after_n_solutions_sample_sat.py

This is a solvOR implementation of the Google OR-Tools example.
Original: https://github.com/google/or-tools/blob/stable/ortools/sat/samples/stop_after_n_solutions_sample_sat.py

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
"""Code sample that solves a model and displays a small number of solutions."""

from solvor import Model, Status


def stop_after_n_solutions_sample_sat():
    """Showcases calling the solver to search for small number of solutions."""
    # Creates the model.
    model = Model()
    # Creates the variables.
    num_vals = 3
    x = model.int_var(0, num_vals - 1, "x")
    y = model.int_var(0, num_vals - 1, "y")
    z = model.int_var(0, num_vals - 1, "z")

    # Create a solver and solve with limit of 5 solutions.
    result = model.solve(solution_limit=5)

    # Print solutions
    if result.status == Status.OPTIMAL and hasattr(result, "solutions") and result.solutions:
        for sol in result.solutions:
            print(f"x={sol['x']} y={sol['y']} z={sol['z']}")
        print(f"Stop search after {len(result.solutions)} solutions")
        print("Status = OPTIMAL")
        print(f"Number of solutions found: {len(result.solutions)}")
        assert len(result.solutions) == 5
    elif result.status == Status.OPTIMAL and result.solution:
        print(f"x={result.solution['x']} y={result.solution['y']} z={result.solution['z']}")
        print("Status = OPTIMAL")
        print("Number of solutions found: 1")
    else:
        print("No solution found.")


stop_after_n_solutions_sample_sat()
# [END program]

"""
OR-Tools Example: multiple_knapsack_mip.py

This is a solvOR implementation of the Google OR-Tools example.
Original: https://github.com/google/or-tools/blob/stable/ortools/linear_solver/samples/multiple_knapsack_mip.py

solvOR provides pure Python solvers for learning and prototyping.
For production-scale problems, consider using Google OR-Tools which
offers compiled C++ solvers with significantly better performance.

Comparison:
- solvOR: Pure Python, readable, educational, no dependencies
- OR-Tools: C++ backend, production-ready, 10-100x faster

Note: This problem (75 binary variables) is at the edge of what pure Python
branch-and-bound can handle efficiently. solvOR uses LP-based heuristics
to find good solutions quickly (~95% of optimal).
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
"""Solve a multiple knapsack problem using a MIP solver."""
# [START import]
from solvor import solve_milp, Status
# [END import]


def main():
    # [START data]
    weights = [48, 30, 42, 36, 36, 48, 42, 42, 36, 24, 30, 30, 42, 36, 36]
    values = [10, 30, 25, 50, 35, 30, 15, 40, 30, 35, 45, 10, 20, 30, 25]
    bin_capacities = [100, 100, 100, 100, 100]
    n_items, n_bins = len(weights), len(bin_capacities)
    # [END data]

    # [START solver]
    # Variables: x[i,b] = 1 if item i in bin b (flattened: i*n_bins + b)
    n_vars = n_items * n_bins
    print("Solving with solvOR MILP (branch and bound with heuristics)")
    # [END solver]

    # [START constraints]
    A, b_rhs = [], []
    # Each item in at most one bin
    for i in range(n_items):
        row = [0] * n_vars
        for b in range(n_bins):
            row[i * n_bins + b] = 1
        A.append(row)
        b_rhs.append(1)
    # Bin capacity constraints
    for b in range(n_bins):
        row = [0] * n_vars
        for i in range(n_items):
            row[i * n_bins + b] = weights[i]
        A.append(row)
        b_rhs.append(bin_capacities[b])
    # [END constraints]

    # [START objective]
    c = [0] * n_vars
    for i in range(n_items):
        for b in range(n_bins):
            c[i * n_bins + b] = values[i]
    # [END objective]

    # [START solve]
    # Heuristics + LNS finds near-optimal solution for large binary problems
    result = solve_milp(c, A, b_rhs, list(range(n_vars)), minimize=False,
                        max_nodes=0, lns_iterations=50, lns_destroy_frac=0.4, seed=42)
    # [END solve]

    # [START print_solution]
    if result.status in (Status.OPTIMAL, Status.FEASIBLE):
        print(f"Total packed value: {result.objective}")
        total_weight = 0
        for b in range(n_bins):
            print(f"Bin {b}")
            bin_weight, bin_value = 0, 0
            for i in range(n_items):
                if result.solution[i * n_bins + b] > 0.5:
                    print(f"Item {i} weight: {weights[i]} value: {values[i]}")
                    bin_weight += weights[i]
                    bin_value += values[i]
            print(f"Packed bin weight: {bin_weight}")
            print(f"Packed bin value: {bin_value}\n")
            total_weight += bin_weight
        print(f"Total packed weight: {total_weight}")
    else:
        print("The problem does not have an optimal solution.")
    # [END print_solution]


if __name__ == "__main__":
    main()
# [END program]

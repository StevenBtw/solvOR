"""
OR-Tools Example: bin_packing_mip.py

This is a solvOR implementation of the Google OR-Tools example.
Original: https://github.com/google/or-tools/blob/stable/ortools/linear_solver/samples/bin_packing_mip.py

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

"""Solve a simple bin packing problem using a MIP solver."""
# [START program]
# [START import]
from solvor import solve_bin_pack

# [END import]


# [START program_part1]
# [START data_model]
def create_data_model():
    """Create the data for the example."""
    data = {}
    weights = [48, 30, 19, 36, 36, 27, 42, 42, 36, 24, 30]
    data["weights"] = weights
    data["items"] = list(range(len(weights)))
    data["bins"] = data["items"]
    data["bin_capacity"] = 100
    return data


# [END data_model]


def main():
    # [START data]
    data = create_data_model()
    # [END data]
    # [END program_part1]

    # [START solver]
    # solvOR uses solve_bin_pack for bin packing problems
    # Note: This uses a fast BFD heuristic. Full MIP formulation would be too slow.
    # [END solver]

    # [START variables]
    # Variables handled internally by bin packing solver
    # [END variables]

    # [START constraints]
    # Constraints handled internally by bin packing solver
    # [END constraints]

    # [START objective]
    # Objective: minimize the number of bins used (default)
    # [END objective]

    # [START solve]
    print("Solving with solvOR bin packing (best-fit-decreasing)")
    result = solve_bin_pack(data["weights"], data["bin_capacity"])
    # [END solve]

    # [START print_solution]
    if result.objective > 0:
        # Group items by bin
        bins_content = {}
        for i, bin_id in enumerate(result.solution):
            if bin_id not in bins_content:
                bins_content[bin_id] = []
            bins_content[bin_id].append(i)

        for bin_id in sorted(bins_content.keys()):
            bin_items = bins_content[bin_id]
            bin_weight = sum(data["weights"][i] for i in bin_items)
            print("Bin number", bin_id)
            print("  Items packed:", bin_items)
            print("  Total weight:", bin_weight)
            print()

        print()
        print("Number of bins used:", int(result.objective))
    else:
        print("The problem does not have an optimal solution.")
    # [END print_solution]


if __name__ == "__main__":
    main()
# [END program_part2]
# [END program]

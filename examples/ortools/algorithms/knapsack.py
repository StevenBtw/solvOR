"""
OR-Tools Example: knapsack.py

This is a solvOR implementation of the Google OR-Tools example.
Original: https://github.com/google/or-tools/blob/stable/ortools/algorithms/samples/knapsack.py

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

"""A simple knapsack problem."""
# [START program]
# [START import]
from solvor import solve_knapsack
# [END import]


def main():
    # [START data]
    values = [
        # fmt:off
      360, 83, 59, 130, 431, 67, 230, 52, 93, 125, 670, 892, 600, 38, 48, 147,
      78, 256, 63, 17, 120, 164, 432, 35, 92, 110, 22, 42, 50, 323, 514, 28,
      87, 73, 78, 15, 26, 78, 210, 36, 85, 189, 274, 43, 33, 10, 19, 389, 276,
      312
        # fmt:on
    ]
    weights = [
        # fmt: off
      7, 0, 30, 22, 80, 94, 11, 81, 70, 64, 59, 18, 0, 36, 3, 8, 15, 42, 9, 0,
      42, 47, 52, 32, 26, 48, 55, 6, 29, 84, 2, 4, 18, 56, 7, 29, 93, 44, 71,
      3, 86, 66, 31, 65, 0, 79, 20, 65, 52, 13
        # fmt: on
    ]
    capacity = 850
    # [END data]

    # [START solve]
    result = solve_knapsack(values, weights, capacity)
    # [END solve]

    # [START print_solution]
    packed_items = [i for i, selected in enumerate(result.solution) if selected]
    packed_weights = [weights[i] for i in packed_items]
    total_weight = sum(packed_weights)

    print("Total value =", int(result.objective))
    print("Total weight:", total_weight)
    print("Packed items:", packed_items)
    print("Packed_weights:", packed_weights)
    # [END print_solution]


if __name__ == "__main__":
    main()
# [END program]

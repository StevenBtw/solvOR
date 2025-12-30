"""
OR-Tools Example: simple_knapsack_program.py

This is a solvOR implementation of the Google OR-Tools example.
Original: https://github.com/google/or-tools/blob/stable/ortools/algorithms/samples/simple_knapsack_program.py

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
"""A simple knapsack problem."""
# [START import]
from solvor import solve_knapsack

# [END import]


def main():
    # [START data]
    # In this example, values = weights (maximize weight in capacity)
    weights = [565, 406, 194, 130, 435, 367, 230, 315, 393, 125, 670, 892, 600, 293, 712, 147, 421, 255]
    capacity = 850
    values = weights  # Same as weights in this example
    # [END data]

    # [START solve]
    result = solve_knapsack(values, weights, capacity)
    # [END solve]

    # [START print_solution]
    packed_items = [i for i, selected in enumerate(result.solution) if selected]
    packed_weights = [weights[i] for i in packed_items]

    print("Packed items: ", packed_items)
    print("Packed weights: ", packed_weights)
    print("Total weight (same as total value): ", int(result.objective))
    # [END print_solution]


if __name__ == "__main__":
    main()
# [END program]

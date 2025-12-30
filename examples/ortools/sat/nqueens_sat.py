"""
OR-Tools Example: nqueens_sat.py

This is a solvOR implementation of the Google OR-Tools example.
Original: https://github.com/google/or-tools/blob/stable/ortools/sat/samples/nqueens_sat.py

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
"""OR-Tools solution to the N-queens problem."""
# [START import]
import sys
import time
from solvor import Model, Status

# [END import]


def main(board_size: int) -> None:
    start_time = time.time()

    # Creates the solver.
    # [START model]
    model = Model()
    # [END model]

    # Creates the variables.
    # [START variables]
    # There are `board_size` number of variables, one for a queen in each column
    # of the board. The value of each variable is the row that the queen is in.
    queens = [model.int_var(0, board_size - 1, f"x_{i}") for i in range(board_size)]
    # [END variables]

    # Creates the constraints.
    # [START constraints]
    # All rows must be different.
    model.add(model.all_different(queens))

    # No two queens can be on the same diagonal.
    for i in range(board_size):
        for j in range(i + 1, board_size):
            model.add(queens[i] + i != queens[j] + j)  # Forward diagonal /
            model.add(queens[i] - i != queens[j] - j)  # Backward diagonal \
    # [END constraints]

    # Solve the model.
    # [START solve]
    result = model.solve(solution_limit=1000)
    # [END solve]

    # Print solutions
    # [START solution_printer]
    if result.status == Status.OPTIMAL and hasattr(result, "solutions") and result.solutions:
        solutions = result.solutions
    elif result.status == Status.OPTIMAL and result.solution:
        solutions = [result.solution]
    else:
        solutions = []

    for idx, sol in enumerate(solutions):
        current_time = time.time()
        print(f"Solution {idx}, time = {current_time - start_time} s")
        for i in range(board_size):
            for j in range(board_size):
                if sol[f"x_{j}"] == i:
                    print("Q", end=" ")
                else:
                    print("_", end=" ")
            print()
        print()
    # [END solution_printer]

    # Statistics.
    # [START statistics]
    print("\nStatistics")
    print(f"  iterations     : {result.iterations}")
    print(f"  wall time      : {time.time() - start_time} s")
    print(f"  solutions found: {len(solutions)}")
    # [END statistics]


if __name__ == "__main__":
    # By default, solve the 8x8 problem.
    size = 8
    if len(sys.argv) > 1:
        size = int(sys.argv[1])
    main(size)
# [END program]

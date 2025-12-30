"""
OR-Tools Example: nurses_sat.py

This is a solvOR implementation of the Google OR-Tools example.
Original: https://github.com/google/or-tools/blob/stable/ortools/sat/samples/nurses_sat.py

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
"""Example of a simple nurse scheduling problem."""
# [START import]
from solvor import Model, Status

# [END import]


def main() -> None:
    # Data.
    # [START data]
    num_nurses = 4
    num_shifts = 3
    num_days = 3
    all_nurses = range(num_nurses)
    all_shifts = range(num_shifts)
    all_days = range(num_days)
    # [END data]

    # Creates the model.
    # [START model]
    model = Model()
    # [END model]

    # Creates shift variables.
    # shifts[(n, d, s)]: nurse 'n' works shift 's' on day 'd'.
    # [START variables]
    shifts = {}
    for n in all_nurses:
        for d in all_days:
            for s in all_shifts:
                shifts[(n, d, s)] = model.int_var(0, 1, f"shift_n{n}_d{d}_s{s}")
    # [END variables]

    # Each shift is assigned to exactly one nurse in the schedule period.
    # [START exactly_one_nurse]
    # Use pairwise constraints: at most one + at least one = exactly one
    for d in all_days:
        for s in all_shifts:
            # At most one (pairwise exclusion)
            for n1 in all_nurses:
                for n2 in range(n1 + 1, num_nurses):
                    model.add(model.sum_le([shifts[(n1, d, s)], shifts[(n2, d, s)]], 1))
            # At least one
            model.add(model.sum_ge([shifts[(n, d, s)] for n in all_nurses], 1))
    # [END exactly_one_nurse]

    # Each nurse works at most one shift per day.
    # [START at_most_one_shift]
    for n in all_nurses:
        for d in all_days:
            # Pairwise exclusion for at most one
            for s1 in all_shifts:
                for s2 in range(s1 + 1, num_shifts):
                    model.add(model.sum_le([shifts[(n, d, s1)], shifts[(n, d, s2)]], 1))
    # [END at_most_one_shift]

    # [START assign_nurses_evenly]
    # Try to distribute the shifts evenly, so that each nurse works
    # min_shifts_per_nurse shifts. If this is not possible, because the total
    # number of shifts is not divisible by the number of nurses, some nurses will
    # be assigned one more shift.
    min_shifts_per_nurse = (num_shifts * num_days) // num_nurses
    if num_shifts * num_days % num_nurses == 0:
        max_shifts_per_nurse = min_shifts_per_nurse
    else:
        max_shifts_per_nurse = min_shifts_per_nurse + 1
    for n in all_nurses:
        shifts_worked = []
        for d in all_days:
            for s in all_shifts:
                shifts_worked.append(shifts[(n, d, s)])
        model.add(model.sum_ge(shifts_worked, min_shifts_per_nurse))
        model.add(model.sum_le(shifts_worked, max_shifts_per_nurse))
    # [END assign_nurses_evenly]

    # Creates the solver and solve.
    # [START solve]
    result = model.solve(solution_limit=5)
    # [END solve]

    # [START solution_printer]
    if result.status == Status.OPTIMAL and hasattr(result, "solutions") and result.solutions:
        solutions = result.solutions
    elif result.status == Status.OPTIMAL and result.solution:
        solutions = [result.solution]
    else:
        solutions = []

    for sol_idx, sol in enumerate(solutions):
        print(f"Solution {sol_idx + 1}")
        for d in all_days:
            print(f"Day {d}")
            for n in all_nurses:
                is_working = False
                for s in all_shifts:
                    if sol[f"shift_n{n}_d{d}_s{s}"] == 1:
                        is_working = True
                        print(f"  Nurse {n} works shift {s}")
                if not is_working:
                    print(f"  Nurse {n} does not work")
        if sol_idx + 1 >= 5:
            print(f"Stop search after 5 solutions")
            break
    # [END solution_printer]

    # Statistics.
    # [START statistics]
    print("\nStatistics")
    print(f"  - iterations     : {result.iterations}")
    print(f"  - solutions found: {len(solutions)}")
    # [END statistics]


if __name__ == "__main__":
    main()
# [END program]

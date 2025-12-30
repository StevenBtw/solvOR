"""
OR-Tools Example: literal_sample_sat.py

This is a solvOR implementation of the Google OR-Tools example.
Original: https://github.com/google/or-tools/blob/stable/ortools/sat/samples/literal_sample_sat.py

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

"""Code sample to demonstrate Boolean variable and literals."""


from solvor import Model


def literal_sample_sat():
    model = Model()
    # Boolean variable: int_var with domain [0, 1]
    x = model.int_var(0, 1, "x")
    # In solvOR, negation is represented as (1 - x) in constraints
    # For display, we show the variable and its logical complement
    print(f"x (bool var with domain [0,1])")
    print(f"not_x = 1 - x (logical negation)")


literal_sample_sat()

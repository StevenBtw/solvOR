"""
OR-Tools Example: simple_max_flow_program.py

This is a solvOR implementation of the Google OR-Tools example.
Original: https://github.com/google/or-tools/blob/stable/ortools/graph/samples/simple_max_flow_program.py

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
"""From Taha 'Introduction to Operations Research', example 6.4-2."""
# [START import]
from collections import defaultdict

from solvor import max_flow

# [END import]


def main():
    """MaxFlow simple interface example."""
    # [START data]
    # Define three parallel arrays: start_nodes, end_nodes, and the capacities
    # between each pair. For instance, the arc from node 0 to node 1 has a
    # capacity of 20.
    start_nodes = [0, 0, 0, 1, 1, 2, 2, 3, 3]
    end_nodes = [1, 2, 3, 2, 4, 3, 4, 2, 4]
    capacities = [20, 30, 10, 40, 30, 10, 20, 5, 20]
    # [END data]

    # [START constraints]
    # Build graph: {node: [(neighbor, capacity), ...]}
    graph = defaultdict(list)
    for start, end, cap in zip(start_nodes, end_nodes, capacities):
        graph[start].append((end, cap))
    # [END constraints]

    # [START solve]
    # Find the maximum flow between node 0 and node 4.
    result = max_flow(dict(graph), 0, 4)
    # [END solve]

    # [START print_solution]
    if result.objective > 0:
        print("Max flow:", int(result.objective))
        print("")
        print(" Arc    Flow / Capacity")
        for (start, end), flow in sorted(result.solution.items()):
            # Find original capacity for this arc
            cap = next(c for s, e, c in zip(start_nodes, end_nodes, capacities) if s == start and e == end)
            print(f"{start} / {end}   {flow:3}  / {cap:3}")
    else:
        print("There was an issue with the max flow input.")
    # [END print_solution]


if __name__ == "__main__":
    main()
# [END program]

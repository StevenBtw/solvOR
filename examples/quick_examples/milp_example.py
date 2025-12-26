"""
Mixed-Integer Linear Programming Example

Minimize: 2x + 3y
Subject to: x + y >= 4
            x <= 3
            x integer, y continuous
            x, y >= 0

Optimal: x=3, y=1, objective=9
"""

from solvor import solve_milp

c = [2, 3]
A = [
    [-1, -1],  # x + y >= 4
    [1, 0],  # x <= 3
]
b = [-4, 3]
integers = [0]  # x (index 0) must be integer

result = solve_milp(c, A, b, integers)
print(f"Optimal: x={result.solution[0]:.0f}, y={result.solution[1]:.1f}")
print(f"Objective: {result.objective:.1f}")

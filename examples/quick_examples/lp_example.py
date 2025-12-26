"""
Linear Programming Example

Minimize: 2x + 3y
Subject to: x + y >= 4
            x <= 3
            x, y >= 0

Optimal: x=3, y=1, objective=9
"""

from solvor import solve_lp

# Minimize c @ x subject to A @ x <= b
c = [2, 3]  # Objective: 2x + 3y
A = [
    [-1, -1],  # -x - y <= -4 (i.e., x + y >= 4)
    [1, 0],  # x <= 3
]
b = [-4, 3]

result = solve_lp(c, A, b)
print(f"Optimal: x={result.solution[0]:.1f}, y={result.solution[1]:.1f}")
print(f"Objective: {result.objective:.1f}")

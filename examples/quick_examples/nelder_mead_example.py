"""
Nelder-Mead Simplex Example

Minimize the Booth function (derivative-free).
Global minimum at (1, 3) with f(1,3) = 0.
"""

from solvor import nelder_mead


def booth(x):
    return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2


x0 = [0.0, 0.0]
result = nelder_mead(booth, x0)
print(f"Solution: ({result.solution[0]:.4f}, {result.solution[1]:.4f})")
print(f"Objective: {result.objective:.6f}")

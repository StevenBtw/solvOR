"""
Powell's Conjugate Direction Example

Minimize Rosenbrock function without derivatives.
Global minimum at (1, 1).
"""

from solvor import powell


def f(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


x0 = [0.0, 0.0]
result = powell(f, x0)
print(f"Solution: ({result.solution[0]:.4f}, {result.solution[1]:.4f})")
print(f"Objective: {result.objective:.6f}")
print(f"Evaluations: {result.evaluations}")

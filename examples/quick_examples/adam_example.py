"""
Adam Optimizer Example

Minimize Rosenbrock function with adaptive learning rates.
Global minimum at (1, 1).
"""

from solvor import adam


def grad(x):
    dx0 = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2)
    dx1 = 200 * (x[1] - x[0] ** 2)
    return [dx0, dx1]


x0 = [0.0, 0.0]
result = adam(grad, x0, lr=0.1, max_iter=5000)
print(f"Solution: ({result.solution[0]:.4f}, {result.solution[1]:.4f})")
print(f"Objective: {result.objective:.6f}")

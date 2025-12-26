"""
BFGS Quasi-Newton Example

Minimize Rosenbrock function using inverse Hessian approximation.
Global minimum at (1, 1).
"""

from solvor import bfgs


def f(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def grad(x):
    dx0 = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2)
    dx1 = 200 * (x[1] - x[0] ** 2)
    return [dx0, dx1]


x0 = [0.0, 0.0]
result = bfgs(grad, x0, objective_fn=f)
print(f"Solution: ({result.solution[0]:.4f}, {result.solution[1]:.4f})")
print(f"Objective: {result.objective:.6f}")
print(f"Iterations: {result.iterations}")

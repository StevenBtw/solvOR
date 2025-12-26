"""
Gradient Descent Example

Minimize x^2 + y^2.
Global minimum at origin with f(0,0) = 0.
"""

from solvor import gradient_descent


def grad(x):
    return [2 * x[0], 2 * x[1]]


x0 = [5.0, 3.0]
result = gradient_descent(grad, x0, lr=0.1)
print(f"Solution: ({result.solution[0]:.4f}, {result.solution[1]:.4f})")
print(f"Objective: {result.objective:.6f}")

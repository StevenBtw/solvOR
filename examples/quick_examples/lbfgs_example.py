"""
L-BFGS Example

Limited-memory quasi-Newton optimization for large-scale problems.
"""

from solvor import lbfgs


def gradient(x):
    """Gradient of Rosenbrock: f(x,y) = (1-x)^2 + 100*(y-x^2)^2."""
    dx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2)
    dy = 200 * (x[1] - x[0] ** 2)
    return [dx, dy]


x0 = [-1.0, 1.0]
result = lbfgs(gradient, x0, m=10, max_iter=1000)
print(f"Minimum at: {[round(v, 4) for v in result.solution]}")
print(f"Iterations: {result.iterations}")

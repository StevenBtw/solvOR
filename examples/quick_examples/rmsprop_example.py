"""
RMSprop Example

Gradient-based optimization with adaptive learning rates.
"""

from solvor import rmsprop


def gradient(x):
    """Gradient of f(x,y) = x^2 + 2*y^2."""
    return [2 * x[0], 4 * x[1]]


x0 = [5.0, 5.0]
result = rmsprop(gradient, x0, lr=0.1, max_iter=500)
print(f"Minimum at: {[round(v, 6) for v in result.solution]}")
print(f"Objective: {result.objective:.6f}")

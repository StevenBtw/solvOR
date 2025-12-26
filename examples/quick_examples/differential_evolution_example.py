"""
Differential Evolution Example

Minimize the Rosenbrock function.
Global minimum at (1, 1) with f(1,1) = 0.
"""

from solvor import differential_evolution


def rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


bounds = [(-5, 5), (-5, 5)]
result = differential_evolution(rosenbrock, bounds, seed=42)
print(f"Solution: ({result.solution[0]:.4f}, {result.solution[1]:.4f})")
print(f"Objective: {result.objective:.6f}")

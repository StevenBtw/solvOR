"""
Bayesian Optimization Example

Minimize a black-box function with expensive evaluations.
Uses Gaussian process surrogate with acquisition function.
"""

from solvor import bayesian_opt


def black_box(x):
    # Simulated expensive function
    return (x[0] - 0.5) ** 2 + (x[1] + 0.3) ** 2


bounds = [(-2, 2), (-2, 2)]
result = bayesian_opt(black_box, bounds, n_initial=5, max_iter=20, seed=42)
print(f"Solution: ({result.solution[0]:.4f}, {result.solution[1]:.4f})")
print(f"Objective: {result.objective:.6f}")
print(f"Evaluations: {result.evaluations}")

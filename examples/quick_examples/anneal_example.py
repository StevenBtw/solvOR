"""
Simulated Annealing Example

Minimize the Rastrigin function (multimodal with many local minima).
Global minimum at origin with f(0,0) = 0.
"""

from math import cos, pi

from solvor import anneal


def rastrigin(x):
    return 10 * len(x) + sum(xi**2 - 10 * cos(2 * pi * xi) for xi in x)


def neighbor(x):
    import random

    i = random.randint(0, len(x) - 1)
    delta = random.gauss(0, 0.5)
    new_x = list(x)
    new_x[i] = max(-5.12, min(5.12, new_x[i] + delta))
    return new_x


x0 = [2.0, 2.0]
result = anneal(x0, rastrigin, neighbor, seed=42)
print(f"Solution: ({result.solution[0]:.4f}, {result.solution[1]:.4f})")
print(f"Objective: {result.objective:.6f}")

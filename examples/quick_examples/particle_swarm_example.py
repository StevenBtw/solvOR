"""
Particle Swarm Optimization Example

Minimize the sphere function.
Global minimum at origin with f(0,0) = 0.
"""

from solvor import particle_swarm


def sphere(x):
    return sum(xi**2 for xi in x)


bounds = [(-5, 5), (-5, 5)]
result = particle_swarm(sphere, bounds, seed=42)
print(f"Solution: ({result.solution[0]:.4f}, {result.solution[1]:.4f})")
print(f"Objective: {result.objective:.6f}")

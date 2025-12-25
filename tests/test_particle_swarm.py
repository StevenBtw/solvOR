"""Tests for Particle Swarm Optimization."""

import pytest

from solvor.particle_swarm import particle_swarm
from solvor.types import Status


class TestBasicOptimization:
    def test_sphere(self):
        """Sphere function, minimum at origin."""

        def sphere(x):
            return sum(xi**2 for xi in x)

        bounds = [(-5, 5), (-5, 5)]
        result = particle_swarm(sphere, bounds, seed=42)
        assert result.objective < 1e-4
        assert all(abs(xi) < 0.1 for xi in result.solution)

    def test_rosenbrock_2d(self):
        """Rosenbrock function, minimum at (1, 1)."""

        def rosenbrock(x):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

        bounds = [(-5, 5), (-5, 5)]
        result = particle_swarm(rosenbrock, bounds, max_iter=2000, seed=42)
        assert result.objective < 0.1
        assert abs(result.solution[0] - 1.0) < 0.5
        assert abs(result.solution[1] - 1.0) < 0.5

    def test_maximize(self):
        """Test minimize=False."""

        def negative_sphere(x):
            return -(x[0] ** 2 + x[1] ** 2)

        bounds = [(-5, 5), (-5, 5)]
        result = particle_swarm(negative_sphere, bounds, minimize=False, seed=42)
        # Maximum is at origin where f(x) = 0
        assert result.objective > -1e-4

    def test_1d(self):
        """Single dimension optimization."""

        def f(x):
            return (x[0] - 3) ** 2

        bounds = [(0, 10)]
        result = particle_swarm(f, bounds, seed=42)
        assert abs(result.solution[0] - 3.0) < 0.1


class TestParameters:
    def test_more_particles(self):
        """More particles for better exploration."""

        def rastrigin(x):
            import math

            return 10 * len(x) + sum(xi**2 - 10 * math.cos(2 * math.pi * xi) for xi in x)

        bounds = [(-5.12, 5.12)] * 2
        result = particle_swarm(rastrigin, bounds, n_particles=50, max_iter=500, seed=42)
        assert result.objective < 5

    def test_high_inertia(self):
        """High inertia for more exploration."""

        def sphere(x):
            return sum(xi**2 for xi in x)

        bounds = [(-5, 5)] * 2
        result = particle_swarm(sphere, bounds, inertia=0.9, seed=42)
        assert result.objective < 0.1

    def test_high_cognitive(self):
        """High cognitive weight emphasizes personal best."""

        def sphere(x):
            return sum(xi**2 for xi in x)

        bounds = [(-5, 5)] * 2
        result = particle_swarm(sphere, bounds, cognitive=2.5, social=0.5, seed=42)
        assert result.objective < 0.1

    def test_high_social(self):
        """High social weight emphasizes global best."""

        def sphere(x):
            return sum(xi**2 for xi in x)

        bounds = [(-5, 5)] * 2
        result = particle_swarm(sphere, bounds, cognitive=0.5, social=2.5, seed=42)
        assert result.objective < 0.1


class TestProgressCallback:
    def test_callback_called(self):
        """Progress callback is invoked."""
        calls = []

        def callback(progress):
            calls.append(progress)

        def sphere(x):
            return sum(xi**2 for xi in x)

        bounds = [(-5, 5)] * 2
        particle_swarm(sphere, bounds, on_progress=callback, progress_interval=50, seed=42)
        assert len(calls) > 0
        assert all(p.iteration % 50 == 0 for p in calls)

    def test_early_stop(self):
        """Callback can stop optimization early."""

        def callback(progress):
            return True  # Stop immediately

        def sphere(x):
            return sum(xi**2 for xi in x)

        bounds = [(-5, 5)] * 2
        result = particle_swarm(sphere, bounds, on_progress=callback, progress_interval=1, seed=42)
        assert result.status == Status.FEASIBLE
        assert result.iterations <= 2


class TestEdgeCases:
    def test_single_particle(self):
        """Single particle still works (no swarm effect)."""

        def sphere(x):
            return sum(xi**2 for xi in x)

        bounds = [(-5, 5)] * 2
        result = particle_swarm(sphere, bounds, n_particles=1, seed=42)
        # Won't converge well but shouldn't crash
        assert result.evaluations > 0

    def test_single_dimension(self):
        """Works with 1D problems."""

        def f(x):
            return (x[0] - 2) ** 2

        bounds = [(-10, 10)]
        result = particle_swarm(f, bounds, seed=42)
        assert abs(result.solution[0] - 2.0) < 0.5

    def test_evaluations_counted(self):
        """Function evaluations are tracked."""

        def sphere(x):
            return sum(xi**2 for xi in x)

        bounds = [(-5, 5)] * 2
        result = particle_swarm(sphere, bounds, n_particles=10, max_iter=50, seed=42)
        # Initial evals + iterations * particles
        assert result.evaluations >= 10 + 50 * 10


class TestValidation:
    def test_empty_bounds(self):
        """Empty bounds raises error."""

        def sphere(x):
            return sum(xi**2 for xi in x)

        with pytest.raises(ValueError, match="bounds cannot be empty"):
            particle_swarm(sphere, [])


class TestClassicFunctions:
    def test_ackley(self):
        """Ackley function, minimum at origin."""
        from math import cos, exp, pi, sqrt

        def ackley(x):
            n = len(x)
            sum_sq = sum(xi**2 for xi in x)
            sum_cos = sum(cos(2 * pi * xi) for xi in x)
            return -20 * exp(-0.2 * sqrt(sum_sq / n)) - exp(sum_cos / n) + 20 + exp(1)

        bounds = [(-5, 5)] * 2
        result = particle_swarm(ackley, bounds, n_particles=40, max_iter=500, seed=42)
        assert result.objective < 0.5

    def test_booth(self):
        """Booth function, minimum at (1, 3)."""

        def booth(x):
            return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2

        bounds = [(-10, 10), (-10, 10)]
        result = particle_swarm(booth, bounds, seed=42)
        assert result.objective < 1e-4
        assert abs(result.solution[0] - 1.0) < 0.1
        assert abs(result.solution[1] - 3.0) < 0.1

    def test_himmelblau(self):
        """Himmelblau function, has multiple minima at f(x) = 0."""

        def himmelblau(x):
            return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

        bounds = [(-5, 5), (-5, 5)]
        result = particle_swarm(himmelblau, bounds, seed=42)
        assert result.objective < 1e-3


class TestInitialPositions:
    def test_initial_positions_used(self):
        """Initial positions seed the search."""

        def sphere(x):
            return sum(xi**2 for xi in x)

        bounds = [(-5, 5), (-5, 5)]
        # Start near optimum
        initial = [[0.1, 0.1], [0.2, -0.1], [-0.1, 0.2]]
        result = particle_swarm(sphere, bounds, initial_positions=initial, max_iter=100, seed=42)
        assert result.objective < 0.1

    def test_initial_positions_clipped(self):
        """Initial positions values are clipped to bounds."""

        def sphere(x):
            return sum(xi**2 for xi in x)

        bounds = [(-5, 5), (-5, 5)]
        # Start with out-of-bounds values
        initial = [[10.0, 10.0]]  # Way outside bounds
        result = particle_swarm(sphere, bounds, initial_positions=initial, seed=42)
        assert result.ok
        # Should still find solution within bounds
        assert all(-5 <= xi <= 5 for xi in result.solution)

    def test_initial_positions_partial(self):
        """Partial initial positions is filled with random particles."""

        def sphere(x):
            return sum(xi**2 for xi in x)

        bounds = [(-5, 5), (-5, 5)]
        initial = [[0.1, 0.1]]  # Only 1 particle
        result = particle_swarm(sphere, bounds, initial_positions=initial, n_particles=10, max_iter=100, seed=42)
        assert result.ok

    def test_initial_positions_empty(self):
        """Empty initial positions falls back to random."""

        def sphere(x):
            return sum(xi**2 for xi in x)

        bounds = [(-5, 5), (-5, 5)]
        result = particle_swarm(sphere, bounds, initial_positions=[], seed=42)
        assert result.ok

    def test_warm_start_from_previous(self):
        """Warm starting from a previous solution."""

        def sphere(x):
            return sum(xi**2 for xi in x)

        bounds = [(-5, 5), (-5, 5)]

        # First run
        result1 = particle_swarm(sphere, bounds, max_iter=50, seed=42)

        # Warm start second run with first solution
        result2 = particle_swarm(sphere, bounds, initial_positions=[result1.solution], max_iter=50, seed=42)
        assert result2.ok


class TestHigherDimensions:
    def test_3d_sphere(self):
        """3D sphere function."""

        def sphere(x):
            return sum(xi**2 for xi in x)

        bounds = [(-5, 5)] * 3
        result = particle_swarm(sphere, bounds, seed=42)
        assert result.objective < 0.01

    def test_5d_sphere(self):
        """5D sphere function."""

        def sphere(x):
            return sum(xi**2 for xi in x)

        bounds = [(-5, 5)] * 5
        result = particle_swarm(sphere, bounds, n_particles=50, max_iter=1500, seed=42)
        assert result.objective < 0.1

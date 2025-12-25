"""Tests for Differential Evolution global optimizer."""

import pytest

from solvor.differential_evolution import differential_evolution
from solvor.types import Status


class TestBasicOptimization:
    def test_sphere(self):
        """Sphere function, minimum at origin."""

        def sphere(x):
            return sum(xi**2 for xi in x)

        bounds = [(-5, 5), (-5, 5)]
        result = differential_evolution(sphere, bounds, seed=42)
        assert result.objective < 1e-6
        assert all(abs(xi) < 0.01 for xi in result.solution)

    def test_rosenbrock_2d(self):
        """Rosenbrock function, minimum at (1, 1)."""

        def rosenbrock(x):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

        bounds = [(-5, 5), (-5, 5)]
        result = differential_evolution(rosenbrock, bounds, max_iter=2000, seed=42)
        assert result.objective < 0.01
        assert abs(result.solution[0] - 1.0) < 0.1
        assert abs(result.solution[1] - 1.0) < 0.1

    def test_maximize(self):
        """Test minimize=False."""

        def negative_sphere(x):
            return -(x[0] ** 2 + x[1] ** 2)

        bounds = [(-5, 5), (-5, 5)]
        result = differential_evolution(negative_sphere, bounds, minimize=False, seed=42)
        # Maximum is at origin where f(x) = 0
        assert result.objective > -1e-6

    def test_1d(self):
        """Single dimension optimization."""

        def f(x):
            return (x[0] - 3) ** 2

        bounds = [(0, 10)]
        result = differential_evolution(f, bounds, seed=42)
        assert abs(result.solution[0] - 3.0) < 0.1


class TestStrategies:
    def test_rand_1(self):
        """Default rand/1 strategy."""

        def sphere(x):
            return sum(xi**2 for xi in x)

        bounds = [(-5, 5)] * 3
        result = differential_evolution(sphere, bounds, strategy="rand/1", seed=42)
        assert result.objective < 1e-4

    def test_best_1(self):
        """Best/1 strategy converges faster on unimodal."""

        def sphere(x):
            return sum(xi**2 for xi in x)

        bounds = [(-5, 5)] * 3
        result = differential_evolution(sphere, bounds, strategy="best/1", seed=42)
        assert result.objective < 1e-4

    def test_rand_2(self):
        """rand/2 uses two difference vectors."""

        def sphere(x):
            return sum(xi**2 for xi in x)

        bounds = [(-5, 5)] * 3
        result = differential_evolution(sphere, bounds, strategy="rand/2", population_size=20, seed=42)
        assert result.objective < 1e-3


class TestParameters:
    def test_mutation_rate(self):
        """Different mutation rates."""

        def sphere(x):
            return sum(xi**2 for xi in x)

        bounds = [(-5, 5)] * 2

        # Low mutation - more exploitation
        result_low = differential_evolution(sphere, bounds, mutation=0.3, seed=42)
        # High mutation - more exploration
        result_high = differential_evolution(sphere, bounds, mutation=1.5, seed=42)

        # Both should still find good solutions
        assert result_low.objective < 0.1
        assert result_high.objective < 0.1

    def test_crossover_rate(self):
        """Different crossover rates."""

        def sphere(x):
            return sum(xi**2 for xi in x)

        bounds = [(-5, 5)] * 2
        result = differential_evolution(sphere, bounds, crossover=0.9, seed=42)
        assert result.objective < 1e-4

    def test_population_size(self):
        """Larger population for harder problems."""

        def rastrigin(x):
            return 10 * len(x) + sum(xi**2 - 10 * __import__("math").cos(2 * 3.14159 * xi) for xi in x)

        bounds = [(-5.12, 5.12)] * 3
        result = differential_evolution(rastrigin, bounds, population_size=50, max_iter=500, seed=42)
        # Rastrigin is hard, just check we get a reasonable result
        assert result.objective < 10


class TestConvergence:
    def test_tolerance(self):
        """Population convergence stops search."""

        def sphere(x):
            return sum(xi**2 for xi in x)

        bounds = [(-5, 5)] * 2
        result = differential_evolution(sphere, bounds, tol=1e-6, seed=42)
        assert result.status == Status.OPTIMAL

    def test_max_iter(self):
        """Respects max_iter limit."""

        def sphere(x):
            return sum(xi**2 for xi in x)

        bounds = [(-5, 5)] * 2
        result = differential_evolution(sphere, bounds, max_iter=10, tol=1e-20, seed=42)
        assert result.iterations <= 10


class TestProgressCallback:
    def test_callback_called(self):
        """Progress callback is invoked."""
        calls = []

        def callback(progress):
            calls.append(progress)

        def sphere(x):
            return sum(xi**2 for xi in x)

        bounds = [(-5, 5)] * 2
        differential_evolution(sphere, bounds, on_progress=callback, progress_interval=10, seed=42)
        assert len(calls) > 0
        assert all(p.iteration % 10 == 0 for p in calls)

    def test_early_stop(self):
        """Callback can stop optimization early."""

        def callback(progress):
            return True  # Stop immediately

        def sphere(x):
            return sum(xi**2 for xi in x)

        bounds = [(-5, 5)] * 2
        result = differential_evolution(sphere, bounds, on_progress=callback, progress_interval=1, seed=42)
        assert result.status == Status.FEASIBLE
        assert result.iterations <= 2


class TestEdgeCases:
    def test_small_population(self):
        """Minimum population size is enforced."""

        def sphere(x):
            return sum(xi**2 for xi in x)

        bounds = [(-5, 5)] * 2
        # population_size=2 should be bumped up to 4
        result = differential_evolution(sphere, bounds, population_size=2, seed=42)
        assert result.objective < 0.1

    def test_single_dimension(self):
        """Works with 1D problems."""

        def f(x):
            return (x[0] - 2) ** 2

        bounds = [(-10, 10)]
        result = differential_evolution(f, bounds, seed=42)
        assert abs(result.solution[0] - 2.0) < 0.1

    def test_evaluations_counted(self):
        """Function evaluations are tracked."""

        def sphere(x):
            return sum(xi**2 for xi in x)

        bounds = [(-5, 5)] * 2
        result = differential_evolution(sphere, bounds, max_iter=50, seed=42)
        # At least pop_size * iterations evaluations
        assert result.evaluations >= 50


class TestValidation:
    def test_empty_bounds(self):
        """Empty bounds raises error."""

        def sphere(x):
            return sum(xi**2 for xi in x)

        with pytest.raises(ValueError, match="bounds cannot be empty"):
            differential_evolution(sphere, [])

    def test_invalid_strategy_format(self):
        """Invalid strategy format raises error."""

        def sphere(x):
            return sum(xi**2 for xi in x)

        bounds = [(-5, 5)] * 2
        with pytest.raises(ValueError, match="Invalid strategy format"):
            differential_evolution(sphere, bounds, strategy="invalid")

    def test_invalid_base_type(self):
        """Unknown base vector type raises error."""

        def sphere(x):
            return sum(xi**2 for xi in x)

        bounds = [(-5, 5)] * 2
        with pytest.raises(ValueError, match="Unknown base vector type"):
            differential_evolution(sphere, bounds, strategy="foo/1")


class TestInitialPopulation:
    def test_initial_population_used(self):
        """Initial population seeds the search."""

        def sphere(x):
            return sum(xi**2 for xi in x)

        bounds = [(-5, 5), (-5, 5)]
        # Start near optimum
        initial = [[0.1, 0.1], [0.2, -0.1], [-0.1, 0.2]]
        result = differential_evolution(sphere, bounds, initial_population=initial, max_iter=100, seed=42)
        assert result.objective < 0.1  # Should converge quickly from near-optimal start

    def test_initial_population_clipped(self):
        """Initial population values are clipped to bounds."""

        def sphere(x):
            return sum(xi**2 for xi in x)

        bounds = [(-5, 5), (-5, 5)]
        # Start with out-of-bounds values
        initial = [[10.0, 10.0]]  # Way outside bounds
        result = differential_evolution(sphere, bounds, initial_population=initial, seed=42)
        assert result.ok
        # Should still find solution within bounds
        assert all(-5 <= xi <= 5 for xi in result.solution)

    def test_initial_population_partial(self):
        """Partial initial population is filled with random individuals."""

        def sphere(x):
            return sum(xi**2 for xi in x)

        bounds = [(-5, 5), (-5, 5)]
        initial = [[0.1, 0.1]]  # Only 1 individual
        result = differential_evolution(
            sphere, bounds, initial_population=initial, population_size=10, max_iter=100, seed=42
        )
        assert result.ok

    def test_initial_population_empty(self):
        """Empty initial population falls back to random."""

        def sphere(x):
            return sum(xi**2 for xi in x)

        bounds = [(-5, 5), (-5, 5)]
        result = differential_evolution(sphere, bounds, initial_population=[], seed=42)
        assert result.ok

    def test_warm_start_from_previous(self):
        """Warm starting from a previous solution."""

        def sphere(x):
            return sum(xi**2 for xi in x)

        bounds = [(-5, 5), (-5, 5)]

        # First run
        result1 = differential_evolution(sphere, bounds, max_iter=50, seed=42)

        # Warm start second run with first solution
        result2 = differential_evolution(sphere, bounds, initial_population=[result1.solution], max_iter=50, seed=42)
        assert result2.ok


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
        result = differential_evolution(ackley, bounds, max_iter=500, seed=42)
        assert result.objective < 0.1

    def test_booth(self):
        """Booth function, minimum at (1, 3)."""

        def booth(x):
            return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2

        bounds = [(-10, 10), (-10, 10)]
        result = differential_evolution(booth, bounds, seed=42)
        assert result.objective < 1e-6
        assert abs(result.solution[0] - 1.0) < 0.01
        assert abs(result.solution[1] - 3.0) < 0.01

    def test_beale(self):
        """Beale function, minimum at (3, 0.5)."""

        def beale(x):
            return (
                (1.5 - x[0] + x[0] * x[1]) ** 2
                + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2
                + (2.625 - x[0] + x[0] * x[1] ** 3) ** 2
            )

        bounds = [(-4.5, 4.5), (-4.5, 4.5)]
        result = differential_evolution(beale, bounds, max_iter=1000, seed=42)
        assert result.objective < 0.001

    def test_himmelblau(self):
        """Himmelblau function, has multiple minima at f(x) = 0."""

        def himmelblau(x):
            return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

        bounds = [(-5, 5), (-5, 5)]
        result = differential_evolution(himmelblau, bounds, seed=42)
        assert result.objective < 1e-4

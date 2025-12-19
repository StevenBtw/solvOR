"""Tests for the Bayesian optimization solver."""

import pytest
from solvor.bayesian import bayesian_opt, Status


class TestBasicBayesian:
    def test_1d_optimization(self):
        # Minimize (x-0.5)^2
        def objective(x):
            return (x[0] - 0.5) ** 2

        result = bayesian_opt(objective, bounds=[(0, 1)], max_iter=20, seed=42)
        assert result.status == Status.FEASIBLE
        assert abs(result.solution[0] - 0.5) < 0.3

    def test_2d_optimization(self):
        # Minimize (x-0.3)^2 + (y-0.7)^2
        def objective(x):
            return (x[0] - 0.3) ** 2 + (x[1] - 0.7) ** 2

        result = bayesian_opt(objective, bounds=[(0, 1), (0, 1)], max_iter=30, seed=42)
        assert result.status == Status.FEASIBLE

    def test_maximize(self):
        # Maximize -(x-0.5)^2 (peak at 0.5)
        def objective(x):
            return -(x[0] - 0.5) ** 2

        result = bayesian_opt(
            objective,
            bounds=[(0, 1)],
            max_iter=20,
            minimize=False,
            seed=42
        )
        assert result.status == Status.FEASIBLE
        assert abs(result.solution[0] - 0.5) < 0.3


class TestBoundHandling:
    def test_wide_bounds(self):
        # Optimum at center of wide bounds
        def objective(x):
            return (x[0] - 50) ** 2

        result = bayesian_opt(objective, bounds=[(0, 100)], max_iter=25, seed=42)
        assert result.status == Status.FEASIBLE
        assert abs(result.solution[0] - 50) < 20

    def test_narrow_bounds(self):
        # Very narrow search space
        def objective(x):
            return (x[0] - 0.5) ** 2

        result = bayesian_opt(objective, bounds=[(0.4, 0.6)], max_iter=15, seed=42)
        assert result.status == Status.FEASIBLE
        assert 0.4 <= result.solution[0] <= 0.6

    def test_asymmetric_bounds(self):
        # Non-centered optimum
        def objective(x):
            return (x[0] - 0.1) ** 2

        result = bayesian_opt(objective, bounds=[(0, 1)], max_iter=20, seed=42)
        assert result.status == Status.FEASIBLE


class TestMultiDimensional:
    def test_3d_optimization(self):
        def objective(x):
            return (x[0] - 0.3) ** 2 + (x[1] - 0.5) ** 2 + (x[2] - 0.7) ** 2

        result = bayesian_opt(
            objective,
            bounds=[(0, 1), (0, 1), (0, 1)],
            max_iter=40,
            seed=42
        )
        assert result.status == Status.FEASIBLE

    def test_different_scales(self):
        # Variables with different scales
        def objective(x):
            return (x[0] - 5) ** 2 + (x[1] - 0.5) ** 2

        result = bayesian_opt(
            objective,
            bounds=[(0, 10), (0, 1)],
            max_iter=30,
            seed=42
        )
        assert result.status == Status.FEASIBLE


class TestMultiModal:
    def test_simple_multimodal(self):
        # Function with multiple local minima
        import math

        def objective(x):
            return math.sin(5 * x[0]) + (x[0] - 0.5) ** 2

        result = bayesian_opt(objective, bounds=[(0, 1)], max_iter=25, seed=42)
        assert result.status == Status.FEASIBLE
        # Should find a good local minimum
        assert result.objective < 1.0


class TestParameters:
    def test_initial_points(self):
        def objective(x):
            return x[0] ** 2

        result = bayesian_opt(
            objective,
            bounds=[(0, 1)],
            max_iter=15,
            n_initial=5,
            seed=42
        )
        assert result.status == Status.FEASIBLE

    def test_more_initial_points(self):
        def objective(x):
            return (x[0] - 0.5) ** 2

        # More initial exploration
        result = bayesian_opt(
            objective,
            bounds=[(0, 1)],
            max_iter=15,
            n_initial=10,
            seed=42
        )
        assert result.status == Status.FEASIBLE


class TestEdgeCases:
    def test_flat_function(self):
        # Constant function
        def objective(x):
            return 1.0

        result = bayesian_opt(objective, bounds=[(0, 1)], max_iter=10, seed=42)
        assert result.status == Status.FEASIBLE
        assert abs(result.objective - 1.0) < 1e-6

    def test_linear_function(self):
        # Linear: minimum at boundary
        def objective(x):
            return x[0]

        result = bayesian_opt(objective, bounds=[(0, 1)], max_iter=15, seed=42)
        assert result.status == Status.FEASIBLE
        # Should find x close to 0
        assert result.solution[0] < 0.3

    def test_single_iteration(self):
        def objective(x):
            return x[0] ** 2

        result = bayesian_opt(objective, bounds=[(0, 1)], max_iter=1, seed=42)
        assert result.status == Status.FEASIBLE


class TestStress:
    def test_many_iterations(self):
        def objective(x):
            return (x[0] - 0.5) ** 2

        result = bayesian_opt(objective, bounds=[(0, 1)], max_iter=50, seed=42)
        assert result.status == Status.FEASIBLE
        assert abs(result.solution[0] - 0.5) < 0.15

    def test_evaluations_tracked(self):
        def objective(x):
            return x[0] ** 2

        result = bayesian_opt(objective, bounds=[(0, 1)], max_iter=20, seed=42)
        # Should have tracked evaluations
        assert result.evaluations >= 20

"""Tests for the Bayesian optimization solver."""

from solvor.bayesian import bayesian_opt
from solvor.types import Status


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


class TestBayesianBehavior:
    def test_beats_random_search(self):
        # Bayesian optimization should find better solutions than pure random
        # on a problem where the surrogate model helps
        import random

        def objective(x):
            # Needle-in-haystack: optimum at (0.3, 0.7)
            return (x[0] - 0.3) ** 2 + (x[1] - 0.7) ** 2

        # Bayesian optimization
        bayes_result = bayesian_opt(
            objective,
            bounds=[(0, 1), (0, 1)],
            max_iter=30,
            n_initial=5,
            seed=42
        )

        # Pure random search with same budget
        random.seed(42)
        random_best = float('inf')
        for _ in range(30):
            x = [random.uniform(0, 1), random.uniform(0, 1)]
            random_best = min(random_best, objective(x))

        # Bayesian should find better or equal solution
        # (with GP guidance vs pure luck)
        assert bayes_result.objective <= random_best * 1.5  # Allow some margin

    def test_exploits_surrogate_model(self):
        # Test that Bayesian opt concentrates samples near the optimum
        # by tracking where it evaluates
        evaluated_points = []

        def tracking_objective(x):
            evaluated_points.append(x[:])
            return (x[0] - 0.5) ** 2

        bayesian_opt(
            tracking_objective,
            bounds=[(0, 1)],
            max_iter=25,
            n_initial=5,
            seed=42
        )

        # After initial random points, should concentrate near x=0.5
        later_points = evaluated_points[10:]  # Skip initial exploration
        near_optimum = sum(1 for p in later_points if abs(p[0] - 0.5) < 0.3)

        # Most later samples should be near the optimum
        assert near_optimum > len(later_points) * 0.5


class TestSeedReproducibility:
    def test_same_seed_same_result(self):
        def objective(x):
            return (x[0] - 0.5) ** 2 + (x[1] - 0.3) ** 2

        result1 = bayesian_opt(objective, bounds=[(0, 1), (0, 1)], max_iter=20, seed=123)
        result2 = bayesian_opt(objective, bounds=[(0, 1), (0, 1)], max_iter=20, seed=123)

        assert result1.solution == result2.solution
        assert result1.objective == result2.objective
        assert result1.evaluations == result2.evaluations

    def test_different_seed_different_result(self):
        def objective(x):
            return (x[0] - 0.5) ** 2

        result1 = bayesian_opt(objective, bounds=[(0, 1)], max_iter=15, seed=42)
        result2 = bayesian_opt(objective, bounds=[(0, 1)], max_iter=15, seed=99)

        # Different seeds should (almost certainly) give different solutions
        assert result1.solution != result2.solution

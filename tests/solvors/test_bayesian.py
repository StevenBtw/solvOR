"""Tests for the Bayesian optimization solver."""

from solvor.bayesian import bayesian_opt
from solvor.types import Progress, Status


class TestBasicBayesian:
    def test_1d_optimization(self):
        def objective(x):
            return (x[0] - 0.5) ** 2

        result = bayesian_opt(objective, bounds=[(0, 1)], max_iter=20, seed=42)
        assert result.ok or result.status == Status.MAX_ITER
        assert abs(result.solution[0] - 0.5) < 0.3

    def test_2d_optimization(self):
        def objective(x):
            return (x[0] - 0.3) ** 2 + (x[1] - 0.7) ** 2

        result = bayesian_opt(objective, bounds=[(0, 1), (0, 1)], max_iter=30, seed=42)
        assert result.ok or result.status == Status.MAX_ITER
        assert result.objective < 0.1  # Should find near-optimal solution

    def test_maximize(self):
        def objective(x):
            return -((x[0] - 0.5) ** 2)

        result = bayesian_opt(objective, bounds=[(0, 1)], max_iter=20, minimize=False, seed=42)
        assert result.ok or result.status == Status.MAX_ITER
        assert abs(result.solution[0] - 0.5) < 0.3


class TestBoundHandling:
    def test_wide_bounds(self):
        def objective(x):
            return (x[0] - 50) ** 2

        result = bayesian_opt(objective, bounds=[(0, 100)], max_iter=25, seed=42)
        assert result.ok or result.status == Status.MAX_ITER
        assert abs(result.solution[0] - 50) < 20

    def test_narrow_bounds(self):
        def objective(x):
            return (x[0] - 0.5) ** 2

        result = bayesian_opt(objective, bounds=[(0.4, 0.6)], max_iter=15, seed=42)
        assert result.ok or result.status == Status.MAX_ITER
        assert 0.4 <= result.solution[0] <= 0.6

    def test_asymmetric_bounds(self):
        def objective(x):
            return (x[0] - 0.1) ** 2

        result = bayesian_opt(objective, bounds=[(0, 1)], max_iter=20, seed=42)
        assert result.ok or result.status == Status.MAX_ITER


class TestMultiDimensional:
    def test_3d_optimization(self):
        def objective(x):
            return (x[0] - 0.3) ** 2 + (x[1] - 0.5) ** 2 + (x[2] - 0.7) ** 2

        result = bayesian_opt(objective, bounds=[(0, 1), (0, 1), (0, 1)], max_iter=40, seed=42)
        assert result.ok or result.status == Status.MAX_ITER
        assert result.objective < 0.15  # Should find reasonably good solution


class TestMultiModal:
    def test_simple_multimodal(self):
        import math

        def objective(x):
            return math.sin(5 * x[0]) + (x[0] - 0.5) ** 2

        result = bayesian_opt(objective, bounds=[(0, 1)], max_iter=25, seed=42)
        assert result.ok or result.status == Status.MAX_ITER
        assert result.objective < 1.0


class TestEdgeCases:
    def test_flat_function(self):
        def objective(x):
            return 1.0

        result = bayesian_opt(objective, bounds=[(0, 1)], max_iter=10, seed=42)
        assert result.ok or result.status == Status.MAX_ITER
        assert abs(result.objective - 1.0) < 1e-6

    def test_linear_function(self):
        def objective(x):
            return x[0]

        result = bayesian_opt(objective, bounds=[(0, 1)], max_iter=15, seed=42)
        assert result.ok or result.status == Status.MAX_ITER
        assert result.solution[0] < 0.3


class TestStress:
    def test_many_iterations(self):
        def objective(x):
            return (x[0] - 0.5) ** 2

        result = bayesian_opt(objective, bounds=[(0, 1)], max_iter=50, seed=42)
        assert result.ok or result.status == Status.MAX_ITER
        assert abs(result.solution[0] - 0.5) < 0.15

    def test_evaluations_tracked(self):
        def objective(x):
            return x[0] ** 2

        result = bayesian_opt(objective, bounds=[(0, 1)], max_iter=20, seed=42)
        assert result.evaluations >= 20


class TestBayesianBehavior:
    def test_beats_random_search(self):
        import random

        def objective(x):
            return (x[0] - 0.3) ** 2 + (x[1] - 0.7) ** 2

        bayes_result = bayesian_opt(objective, bounds=[(0, 1), (0, 1)], max_iter=30, n_initial=5, seed=42)

        random.seed(42)
        random_best = float("inf")
        for _ in range(30):
            x = [random.uniform(0, 1), random.uniform(0, 1)]
            random_best = min(random_best, objective(x))

        assert bayes_result.objective <= random_best * 1.5

    def test_exploits_surrogate_model(self):
        evaluated_points = []

        def tracking_objective(x):
            evaluated_points.append(x[:])
            return (x[0] - 0.5) ** 2

        bayesian_opt(tracking_objective, bounds=[(0, 1)], max_iter=25, n_initial=5, seed=42)

        later_points = evaluated_points[10:]
        near_optimum = sum(1 for p in later_points if abs(p[0] - 0.5) < 0.3)

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

        assert result1.solution != result2.solution


class TestAcquisitionFunctions:
    def test_expected_improvement(self):
        def objective(x):
            return (x[0] - 0.5) ** 2

        result = bayesian_opt(objective, bounds=[(0, 1)], max_iter=20, acquisition="ei", seed=42)
        assert result.ok or result.status == Status.MAX_ITER
        assert abs(result.solution[0] - 0.5) < 0.3

    def test_ucb(self):
        def objective(x):
            return (x[0] - 0.5) ** 2

        result = bayesian_opt(objective, bounds=[(0, 1)], max_iter=20, acquisition="ucb", seed=42)
        assert result.ok or result.status == Status.MAX_ITER
        assert abs(result.solution[0] - 0.5) < 0.3

    def test_ucb_kappa(self):
        def objective(x):
            return (x[0] - 0.5) ** 2

        # Higher kappa = more exploration
        result = bayesian_opt(objective, bounds=[(0, 1)], max_iter=20, acquisition="ucb", kappa=5.0, seed=42)
        assert result.ok or result.status == Status.MAX_ITER

    def test_invalid_acquisition(self):
        def objective(x):
            return x[0] ** 2

        try:
            bayesian_opt(objective, bounds=[(0, 1)], acquisition="invalid")
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert "ei" in str(e) and "ucb" in str(e)


class TestProgressCallback:
    def test_callback_called(self):
        calls = []

        def callback(p: Progress) -> None:
            calls.append(p)

        def objective(x):
            return x[0] ** 2

        bayesian_opt(objective, bounds=[(0, 1)], max_iter=20, on_progress=callback, progress_interval=5, seed=42)

        assert len(calls) > 0
        for p in calls:
            assert p.iteration > 0
            assert p.evaluations > 0

    def test_early_stopping(self):
        def stop_early(p: Progress) -> bool:
            return p.iteration >= 10

        def objective(x):
            return x[0] ** 2

        result = bayesian_opt(
            objective, bounds=[(0, 1)], max_iter=50, on_progress=stop_early, progress_interval=1, seed=42
        )

        # Should have stopped early
        assert result.iterations <= 15
        assert result.status == Status.FEASIBLE

    def test_no_callback_no_crash(self):
        def objective(x):
            return x[0] ** 2

        result = bayesian_opt(objective, bounds=[(0, 1)], max_iter=10, seed=42)
        assert result.ok or result.status == Status.MAX_ITER

    def test_progress_interval_zero_disabled(self):
        calls = []

        def callback(p: Progress) -> None:
            calls.append(p)

        def objective(x):
            return x[0] ** 2

        bayesian_opt(objective, bounds=[(0, 1)], max_iter=20, on_progress=callback, progress_interval=0, seed=42)

        # With interval=0, callback should not be called
        assert len(calls) == 0

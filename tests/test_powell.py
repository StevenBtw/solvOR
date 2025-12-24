"""Tests for Powell's conjugate direction method."""

from solvor.powell import powell
from solvor.types import Status


class TestBasicOptimization:
    def test_quadratic(self):
        """Simple quadratic, minimum at origin."""

        def f(x):
            return x[0] ** 2 + x[1] ** 2

        result = powell(f, [5.0, 5.0])
        assert result.objective < 1e-6
        assert abs(result.solution[0]) < 1e-3
        assert abs(result.solution[1]) < 1e-3

    def test_rosenbrock(self):
        """Rosenbrock function, minimum at (1, 1)."""

        def f(x):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

        result = powell(f, [0.0, 0.0], max_iter=500)
        assert result.objective < 1e-3
        assert abs(result.solution[0] - 1.0) < 0.1
        assert abs(result.solution[1] - 1.0) < 0.1

    def test_maximize(self):
        """Test maximize mode."""

        def f(x):
            return -(x[0] ** 2 + x[1] ** 2)

        result = powell(f, [5.0, 5.0], minimize=False)
        # Maximum is at origin
        assert result.objective > -1e-6

    def test_1d(self):
        """Single dimension optimization."""

        def f(x):
            return (x[0] - 3) ** 2

        result = powell(f, [0.0])
        assert abs(result.solution[0] - 3.0) < 0.01


class TestBounds:
    def test_bounded_optimization(self):
        """Optimization with bounds moves toward boundary."""

        def f(x):
            return (x[0] - 10) ** 2 + (x[1] - 10) ** 2

        # Optimum at (10, 10) but bounds limit to [0, 5]
        result = powell(f, [0.0, 0.0], bounds=[(0, 5), (0, 5)], max_iter=50)
        # Should move toward (5, 5) - check improvement over start
        start_obj = f([0.0, 0.0])  # 200
        assert result.objective < start_obj  # Improved
        assert result.solution[0] >= -0.5  # Approximately respects lower bound
        assert result.solution[1] >= -0.5

    def test_bounded_1d(self):
        """1D bounded optimization moves toward boundary."""

        def f(x):
            return (x[0] - 10) ** 2

        result = powell(f, [0.0], bounds=[(-2, 3)], max_iter=20)
        # Should move toward x=3 (closer to 10 within bounds)
        start_obj = f([0.0])  # 100
        assert result.objective < start_obj  # Improved
        assert result.solution[0] > 0  # Moved in right direction


class TestConvergence:
    def test_tolerance(self):
        """Stops when change in f < tol."""

        def f(x):
            return x[0] ** 2

        result = powell(f, [10.0], tol=1e-10, max_iter=200)
        assert result.objective < 1e-8

    def test_max_iter(self):
        """Respects max_iter limit."""

        def f(x):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

        result = powell(f, [0.0, 0.0], max_iter=3, tol=1e-20)
        assert result.iterations <= 3
        assert result.status == Status.MAX_ITER


class TestProgressCallback:
    def test_callback_called(self):
        calls = []

        def callback(progress):
            calls.append(progress)

        def f(x):
            return x[0] ** 2 + x[1] ** 2

        powell(f, [5.0, 5.0], on_progress=callback, progress_interval=1)
        assert len(calls) > 0

    def test_early_stop(self):
        def callback(progress):
            return True  # Stop immediately

        def f(x):
            return x[0] ** 2

        result = powell(f, [10.0], on_progress=callback, progress_interval=1)
        assert result.status == Status.FEASIBLE


class TestClassicFunctions:
    def test_booth(self):
        """Booth function, minimum at (1, 3)."""

        def f(x):
            return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2

        result = powell(f, [0.0, 0.0], max_iter=100)
        assert result.objective < 1e-4
        assert abs(result.solution[0] - 1.0) < 0.01
        assert abs(result.solution[1] - 3.0) < 0.01

    def test_beale(self):
        """Beale function, minimum at (3, 0.5)."""

        def f(x):
            return (
                (1.5 - x[0] + x[0] * x[1]) ** 2
                + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2
                + (2.625 - x[0] + x[0] * x[1] ** 3) ** 2
            )

        result = powell(f, [0.0, 0.0], max_iter=200)
        assert result.objective < 1e-3

    def test_himmelblau(self):
        """Himmelblau function, has multiple minima at f(x) = 0."""

        def f(x):
            return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

        result = powell(f, [0.0, 0.0], max_iter=200)
        assert result.objective < 1e-4


class TestHigherDimensions:
    def test_sphere_3d(self):
        """Sphere function in 3D."""

        def f(x):
            return sum(xi**2 for xi in x)

        result = powell(f, [1.0, 2.0, 3.0], max_iter=100)
        assert result.objective < 1e-6
        assert all(abs(xi) < 1e-3 for xi in result.solution)

    def test_sphere_5d(self):
        """Sphere function in 5D."""

        def f(x):
            return sum(xi**2 for xi in x)

        result = powell(f, [1.0] * 5, max_iter=200)
        assert result.objective < 1e-4


class TestEdgeCases:
    def test_already_at_minimum(self):
        """Start at minimum."""

        def f(x):
            return x[0] ** 2

        result = powell(f, [0.0])
        assert result.objective < 1e-8

    def test_evaluations_counted(self):
        """Counts function evaluations."""

        def f(x):
            return x[0] ** 2 + x[1] ** 2

        result = powell(f, [5.0, 5.0], max_iter=50)
        assert result.evaluations > 0
        assert result.evaluations >= result.iterations

    def test_negative_start(self):
        """Works with negative starting point."""

        def f(x):
            return (x[0] + 5) ** 2 + (x[1] + 3) ** 2

        result = powell(f, [0.0, 0.0])
        assert abs(result.solution[0] + 5.0) < 0.1
        assert abs(result.solution[1] + 3.0) < 0.1


class TestComparisonWithNelderMead:
    def test_faster_on_smooth(self):
        """Powell is often faster than Nelder-Mead on smooth functions."""

        def f(x):
            return (x[0] - 1) ** 2 + (x[1] - 2) ** 2 + (x[2] - 3) ** 2

        result = powell(f, [0.0, 0.0, 0.0], max_iter=50)
        assert result.objective < 1e-6
        # Should converge quickly due to quadratic nature
        assert result.iterations < 20

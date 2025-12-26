"""Tests for Nelder-Mead simplex optimizer."""

from solvor.nelder_mead import nelder_mead
from solvor.types import Status


class TestBasicOptimization:
    def test_rosenbrock_2d(self):
        """Rosenbrock function, minimum at (1, 1)."""

        def rosenbrock(x):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

        result = nelder_mead(rosenbrock, [0.0, 0.0], max_iter=2000, tol=1e-8)
        assert result.objective < 1e-4
        assert abs(result.solution[0] - 1.0) < 0.1
        assert abs(result.solution[1] - 1.0) < 0.1

    def test_quadratic(self):
        """Simple quadratic, minimum at origin."""

        def quadratic(x):
            return x[0] ** 2 + x[1] ** 2

        result = nelder_mead(quadratic, [5.0, 5.0])
        assert result.status == Status.OPTIMAL
        assert result.objective < 1e-6
        assert abs(result.solution[0]) < 1e-3
        assert abs(result.solution[1]) < 1e-3

    def test_maximize(self):
        """Test maximize=False (maximize)."""

        def negative_quadratic(x):
            return -(x[0] ** 2 + x[1] ** 2)

        result = nelder_mead(negative_quadratic, [5.0, 5.0], minimize=False)
        # Maximum is at origin where f(x) = 0
        assert result.objective > -1e-6

    def test_1d(self):
        """Single dimension optimization."""

        def f(x):
            return (x[0] - 3) ** 2

        result = nelder_mead(f, [0.0], initial_step=1.0)
        assert abs(result.solution[0] - 3.0) < 0.01


class TestAdaptive:
    def test_adaptive_rosenbrock(self):
        """Adaptive parameters for Rosenbrock."""

        def rosenbrock(x):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

        result = nelder_mead(rosenbrock, [0.0, 0.0], adaptive=True, max_iter=2000)
        assert result.objective < 1e-3

    def test_adaptive_higher_dim(self):
        """Adaptive helps in higher dimensions."""

        def sphere(x):
            return sum(xi**2 for xi in x)

        result = nelder_mead(sphere, [1.0] * 5, adaptive=True, max_iter=3000)
        assert result.objective < 1e-4


class TestConvergence:
    def test_tolerance(self):
        """Stops when spread < tol."""

        def f(x):
            return x[0] ** 2

        result = nelder_mead(f, [10.0], tol=1e-10, max_iter=500, initial_step=1.0)
        assert result.objective < 1e-6

    def test_max_iter(self):
        """Respects max_iter limit."""

        def f(x):
            return x[0] ** 2

        result = nelder_mead(f, [10.0], max_iter=5, tol=1e-20)
        assert result.iterations <= 5
        assert result.status == Status.MAX_ITER


class TestProgressCallback:
    def test_callback_called(self):
        calls = []

        def callback(progress):
            calls.append(progress)

        def f(x):
            return x[0] ** 2 + x[1] ** 2

        nelder_mead(f, [5.0, 5.0], on_progress=callback, progress_interval=10)
        assert len(calls) > 0
        assert all(p.iteration % 10 == 0 for p in calls)

    def test_early_stop(self):
        def callback(progress):
            return True  # Stop immediately

        def f(x):
            return x[0] ** 2

        result = nelder_mead(f, [10.0], on_progress=callback, progress_interval=1)
        assert result.status == Status.FEASIBLE
        assert result.iterations <= 2


class TestEdgeCases:
    def test_already_at_minimum(self):
        """Start at minimum."""

        def f(x):
            return x[0] ** 2

        result = nelder_mead(f, [0.0])
        assert result.objective < 1e-6

    def test_initial_step(self):
        """Custom initial step size helps reach distant optima."""

        def f(x):
            return (x[0] - 10) ** 2 + (x[1] - 10) ** 2

        # Small step won't reach far optimum easily
        result = nelder_mead(f, [0.0, 0.0], initial_step=1.0, max_iter=500)
        assert abs(result.solution[0] - 10) < 0.5
        assert abs(result.solution[1] - 10) < 0.5

    def test_evaluations_counted(self):
        """Counts function evaluations."""

        def f(x):
            return x[0] ** 2

        result = nelder_mead(f, [5.0], max_iter=100)
        assert result.evaluations > 0
        assert result.evaluations >= result.iterations


class TestClassicFunctions:
    def test_beale(self):
        """Beale function, minimum at (3, 0.5)."""

        def beale(x):
            return (
                (1.5 - x[0] + x[0] * x[1]) ** 2
                + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2
                + (2.625 - x[0] + x[0] * x[1] ** 3) ** 2
            )

        result = nelder_mead(beale, [0.0, 0.0], max_iter=2000)
        assert result.objective < 1e-3

    def test_booth(self):
        """Booth function, minimum at (1, 3)."""

        def booth(x):
            return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2

        result = nelder_mead(booth, [0.0, 0.0])
        assert result.objective < 1e-6
        assert abs(result.solution[0] - 1.0) < 1e-3
        assert abs(result.solution[1] - 3.0) < 1e-3

    def test_himmelblau(self):
        """Himmelblau function, has multiple minima at f(x) = 0."""

        def himmelblau(x):
            return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

        result = nelder_mead(himmelblau, [0.0, 0.0], max_iter=1000)
        assert result.objective < 1e-4

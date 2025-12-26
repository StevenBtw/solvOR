"""Tests for BFGS and L-BFGS quasi-Newton optimizers."""

from solvor.bfgs import bfgs, lbfgs
from solvor.types import Status


class TestBFGSBasic:
    def test_quadratic(self):
        """Simple quadratic, minimum at origin."""

        def f(x):
            return x[0] ** 2 + x[1] ** 2

        def grad(x):
            return [2 * x[0], 2 * x[1]]

        result = bfgs(grad, [5.0, 5.0], objective_fn=f)
        assert result.objective < 1e-6
        assert abs(result.solution[0]) < 1e-3
        assert abs(result.solution[1]) < 1e-3

    def test_rosenbrock(self):
        """Rosenbrock function, minimum at (1, 1)."""

        def f(x):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

        def grad(x):
            g0 = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2)
            g1 = 200 * (x[1] - x[0] ** 2)
            return [g0, g1]

        result = bfgs(grad, [0.0, 0.0], objective_fn=f, max_iter=500)
        assert result.objective < 1e-4
        assert abs(result.solution[0] - 1.0) < 0.1
        assert abs(result.solution[1] - 1.0) < 0.1

    def test_maximize(self):
        """Test maximize mode - finds maximum of negative quadratic."""

        def f(x):
            return -(x[0] ** 2 + x[1] ** 2)

        def grad(x):
            return [-2 * x[0], -2 * x[1]]

        # Without line search, BFGS can still improve
        result = bfgs(grad, [5.0, 5.0], minimize=False, max_iter=100)
        # Should move toward origin (f increases as we approach 0)
        # Starting f = -50, maximum at origin is 0
        assert abs(result.solution[0]) < 4.9 or abs(result.solution[1]) < 4.9


class TestBFGSWithoutObjective:
    def test_works_without_line_search(self):
        """BFGS works without objective function (no line search)."""

        def grad(x):
            return [2 * x[0], 2 * x[1]]

        result = bfgs(grad, [5.0, 5.0], max_iter=500)
        # Should still converge, though maybe slower
        assert abs(result.solution[0]) < 0.5
        assert abs(result.solution[1]) < 0.5


class TestLBFGS:
    def test_quadratic(self):
        """L-BFGS on simple quadratic."""

        def f(x):
            return x[0] ** 2 + x[1] ** 2

        def grad(x):
            return [2 * x[0], 2 * x[1]]

        result = lbfgs(grad, [5.0, 5.0], objective_fn=f)
        assert result.objective < 1e-6
        assert abs(result.solution[0]) < 1e-3

    def test_rosenbrock(self):
        """L-BFGS on Rosenbrock."""

        def f(x):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

        def grad(x):
            g0 = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2)
            g1 = 200 * (x[1] - x[0] ** 2)
            return [g0, g1]

        result = lbfgs(grad, [0.0, 0.0], objective_fn=f, max_iter=500)
        assert result.objective < 1e-3

    def test_high_dimensional(self):
        """L-BFGS on high-dimensional sphere."""
        n = 50

        def f(x):
            return sum(xi**2 for xi in x)

        def grad(x):
            return [2 * xi for xi in x]

        result = lbfgs(grad, [1.0] * n, objective_fn=f, m=20, max_iter=200)
        assert result.objective < 1e-4

    def test_memory_parameter(self):
        """Different memory sizes work."""

        def f(x):
            return x[0] ** 2 + x[1] ** 2

        def grad(x):
            return [2 * x[0], 2 * x[1]]

        for m in [3, 10, 50]:
            result = lbfgs(grad, [5.0, 5.0], objective_fn=f, m=m)
            assert result.objective < 1e-4


class TestConvergence:
    def test_tolerance(self):
        """Stops when gradient norm < tol."""

        def grad(x):
            return [2 * x[0], 2 * x[1]]

        result = bfgs(grad, [5.0, 5.0], tol=1e-8, max_iter=1000)
        final_grad_norm = (result.solution[0] ** 2 + result.solution[1] ** 2) ** 0.5
        assert final_grad_norm * 2 < 1e-6  # grad_norm = 2 * |x|

    def test_max_iter(self):
        """Respects max_iter limit."""

        def f(x):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

        def grad(x):
            g0 = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2)
            g1 = 200 * (x[1] - x[0] ** 2)
            return [g0, g1]

        result = bfgs(grad, [0.0, 0.0], objective_fn=f, max_iter=5, tol=1e-20)
        assert result.iterations <= 5
        assert result.status == Status.MAX_ITER


class TestProgressCallback:
    def test_callback_called(self):
        calls = []

        def callback(progress):
            calls.append(progress)

        def f(x):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

        def grad(x):
            g0 = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2)
            g1 = 200 * (x[1] - x[0] ** 2)
            return [g0, g1]

        # Use Rosenbrock which takes longer to converge
        bfgs(grad, [0.0, 0.0], objective_fn=f, on_progress=callback, progress_interval=5, max_iter=100)
        assert len(calls) > 0

    def test_early_stop(self):
        def callback(progress):
            return True  # Stop immediately

        def grad(x):
            return [2 * x[0], 2 * x[1]]

        result = bfgs(grad, [5.0, 5.0], on_progress=callback, progress_interval=1)
        assert result.status == Status.FEASIBLE
        assert result.iterations <= 2


class TestClassicFunctions:
    def test_booth(self):
        """Booth function, minimum at (1, 3)."""

        def f(x):
            return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2

        def grad(x):
            g0 = 2 * (x[0] + 2 * x[1] - 7) + 4 * (2 * x[0] + x[1] - 5)
            g1 = 4 * (x[0] + 2 * x[1] - 7) + 2 * (2 * x[0] + x[1] - 5)
            return [g0, g1]

        result = bfgs(grad, [0.0, 0.0], objective_fn=f)
        assert result.objective < 1e-6
        assert abs(result.solution[0] - 1.0) < 1e-3
        assert abs(result.solution[1] - 3.0) < 1e-3

    def test_sphere_high_dim(self):
        """Sphere function in higher dimensions."""
        n = 10

        def f(x):
            return sum(xi**2 for xi in x)

        def grad(x):
            return [2 * xi for xi in x]

        result = bfgs(grad, [1.0] * n, objective_fn=f)
        assert result.objective < 1e-6
        assert all(abs(xi) < 1e-3 for xi in result.solution)


class TestEdgeCases:
    def test_1d(self):
        """Single dimension."""

        def f(x):
            return (x[0] - 3) ** 2

        def grad(x):
            return [2 * (x[0] - 3)]

        result = bfgs(grad, [0.0], objective_fn=f)
        assert abs(result.solution[0] - 3.0) < 1e-4

    def test_already_at_minimum(self):
        """Start at minimum."""

        def grad(x):
            return [2 * x[0], 2 * x[1]]

        result = bfgs(grad, [0.0, 0.0])
        assert result.iterations == 0  # Should converge immediately

    def test_evaluations_counted(self):
        """Counts function evaluations."""

        def f(x):
            return x[0] ** 2

        def grad(x):
            return [2 * x[0]]

        result = bfgs(grad, [5.0], objective_fn=f, max_iter=100)
        assert result.evaluations > 0
        assert result.evaluations >= result.iterations

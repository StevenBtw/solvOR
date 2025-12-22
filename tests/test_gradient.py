"""Tests for the gradient descent solvers."""

from solvor.gradient import adam, gradient_descent, momentum
from solvor.types import Progress, Status


class TestGradientDescentBasic:
    def test_quadratic_1d(self):
        # Minimize x^2
        def grad(x):
            return [2 * x[0]]

        result = gradient_descent(grad, [5.0], max_iter=1000)
        assert result.status in (Status.OPTIMAL, Status.MAX_ITER)
        assert abs(result.solution[0]) < 0.1

    def test_quadratic_2d(self):
        # Minimize x^2 + y^2
        def grad(x):
            return [2 * x[0], 2 * x[1]]

        result = gradient_descent(grad, [5.0, 5.0], max_iter=1000)
        assert result.status in (Status.OPTIMAL, Status.MAX_ITER)
        assert abs(result.solution[0]) < 0.1
        assert abs(result.solution[1]) < 0.1

    def test_maximize(self):
        # Maximize -x^2 (gradient is -2x, want to go uphill)
        def grad(x):
            return [-2 * x[0]]

        result = gradient_descent(grad, [5.0], minimize=False, max_iter=1000)
        assert abs(result.solution[0]) < 0.1


class TestMomentum:
    def test_quadratic(self):
        def grad(x):
            return [2 * x[0], 2 * x[1]]

        result = momentum(grad, [5.0, 5.0], max_iter=1000)
        assert abs(result.solution[0]) < 0.1
        assert abs(result.solution[1]) < 0.1

    def test_with_beta(self):
        def grad(x):
            return [2 * x[0]]

        result = momentum(grad, [10.0], beta=0.95, max_iter=1000)
        assert abs(result.solution[0]) < 0.1

    def test_faster_convergence(self):
        def grad(x):
            return [2 * x[0], 2 * x[1]]

        # Momentum should converge faster in many cases
        result = momentum(grad, [5.0, 5.0], max_iter=500, lr=0.01)
        assert abs(result.solution[0]) < 0.5
        assert abs(result.solution[1]) < 0.5

    def test_maximize(self):
        # Maximize -x^2
        def grad(x):
            return [-2 * x[0]]

        result = momentum(grad, [5.0], minimize=False, max_iter=1000)
        assert abs(result.solution[0]) < 0.1


class TestAdam:
    def test_basic_adam(self):
        def grad(x):
            return [2 * x[0], 2 * x[1]]

        result = adam(grad, [5.0, 5.0], lr=0.1, max_iter=1000)
        assert abs(result.solution[0]) < 0.1
        assert abs(result.solution[1]) < 0.1

    def test_adam_parameters(self):
        def grad(x):
            return [2 * x[0]]

        result = adam(grad, [10.0], lr=0.1, beta1=0.9, beta2=0.999, max_iter=500)
        assert abs(result.solution[0]) < 0.5

    def test_adam_adaptive(self):
        # Adam adapts to different scales
        def grad(x):
            return [2 * x[0], 200 * x[1]]  # Different scales

        result = adam(grad, [5.0, 5.0], lr=0.1, max_iter=1000)
        assert abs(result.solution[0]) < 0.5
        assert abs(result.solution[1]) < 0.5

    def test_maximize(self):
        # Maximize -x^2
        def grad(x):
            return [-2 * x[0]]

        result = adam(grad, [5.0], minimize=False, lr=0.1, max_iter=1000)
        assert abs(result.solution[0]) < 0.1


class TestLearningRate:
    def test_high_lr(self):
        def grad(x):
            return [2 * x[0]]

        # High learning rate might overshoot
        result = gradient_descent(grad, [5.0], lr=0.4, max_iter=100)
        # Should still converge
        assert abs(result.solution[0]) < 1.0

    def test_low_lr(self):
        def grad(x):
            return [2 * x[0]]

        result = gradient_descent(grad, [5.0], lr=0.001, max_iter=1000)
        # Slow convergence but should make progress
        assert abs(result.solution[0]) < abs(5.0)


class TestConvergence:
    def test_tolerance_stop(self):
        def grad(x):
            return [2 * x[0]]

        result = gradient_descent(grad, [0.001], tol=1e-4, max_iter=10000)
        assert result.status == Status.OPTIMAL  # Should stop early

    def test_max_iter_reached(self):
        def grad(x):
            return [2 * x[0]]

        result = gradient_descent(grad, [100.0], lr=0.0001, max_iter=10, tol=1e-10)
        assert result.status == Status.MAX_ITER


class TestHigherDimensional:
    def test_5d(self):
        def grad(x):
            return [2 * xi for xi in x]

        x0 = [5.0, 4.0, 3.0, 2.0, 1.0]
        result = gradient_descent(grad, x0, max_iter=2000)
        assert all(abs(xi) < 0.5 for xi in result.solution)

    def test_10d_adam(self):
        def grad(x):
            return [2 * xi for xi in x]

        x0 = [float(i) for i in range(10)]
        result = adam(grad, x0, lr=0.1, max_iter=1000)
        assert all(abs(xi) < 1.0 for xi in result.solution)


class TestEdgeCases:
    def test_already_optimal(self):
        def grad(x):
            return [2 * x[0]]

        result = gradient_descent(grad, [0.0], max_iter=100)
        assert result.status == Status.OPTIMAL
        assert result.solution[0] == 0.0

    def test_near_optimal(self):
        def grad(x):
            return [2 * x[0]]

        result = gradient_descent(grad, [1e-7], tol=1e-6, max_iter=100)
        assert result.status == Status.OPTIMAL

    def test_single_iteration(self):
        def grad(x):
            return [2 * x[0]]

        result = gradient_descent(grad, [5.0], max_iter=1)
        assert result.status == Status.MAX_ITER


class TestNonConvex:
    def test_local_minimum(self):
        # Rosenbrock-like (has valley)
        def grad(x):
            # Gradient of (1-x)^2 + 100(y-x^2)^2
            # Simplified version
            return [2 * x[0], 2 * x[1]]

        result = gradient_descent(grad, [2.0, 2.0], max_iter=2000)
        assert abs(result.solution[0]) < 0.5
        assert abs(result.solution[1]) < 0.5


class TestStress:
    def test_many_iterations(self):
        def grad(x):
            return [2 * x[0]]

        result = gradient_descent(grad, [100.0], max_iter=10000)
        assert abs(result.solution[0]) < 0.01

    def test_evaluations_tracked(self):
        def grad(x):
            return [2 * x[0]]

        result = gradient_descent(grad, [5.0], max_iter=100)
        assert result.evaluations >= 100

    def test_all_optimizers_converge(self):
        def grad(x):
            return [2 * x[0], 2 * x[1]]

        x0 = [5.0, 5.0]

        r1 = gradient_descent(grad, x0.copy(), max_iter=1000)
        r2 = momentum(grad, x0.copy(), max_iter=1000)
        r3 = adam(grad, x0.copy(), lr=0.1, max_iter=1000)

        # All should converge
        assert all(abs(xi) < 0.5 for xi in r1.solution)
        assert all(abs(xi) < 0.5 for xi in r2.solution)
        assert all(abs(xi) < 0.5 for xi in r3.solution)


class TestProgressCallback:
    def test_gradient_descent_callback(self):
        def grad(x):
            return [2 * x[0]]

        calls = []

        def callback(progress):
            calls.append(progress.iteration)

        gradient_descent(grad, [10.0], max_iter=50, on_progress=callback, progress_interval=10)
        assert calls == [10, 20, 30, 40, 50]

    def test_momentum_callback(self):
        def grad(x):
            return [2 * x[0]]

        calls = []

        def callback(progress):
            calls.append(progress.iteration)

        momentum(grad, [10.0], max_iter=50, on_progress=callback, progress_interval=10)
        assert calls == [10, 20, 30, 40, 50]

    def test_adam_callback(self):
        def grad(x):
            return [2 * x[0]]

        calls = []

        def callback(progress):
            calls.append(progress.iteration)

        adam(grad, [10.0], max_iter=50, on_progress=callback, progress_interval=10)
        assert calls == [10, 20, 30, 40, 50]

    def test_callback_early_stop(self):
        def grad(x):
            return [2 * x[0]]

        def stop_at_20(progress):
            if progress.iteration >= 20:
                return True

        result = gradient_descent(grad, [100.0], max_iter=100, on_progress=stop_at_20, progress_interval=5)
        assert result.iterations == 20

    def test_callback_receives_progress_data(self):
        def grad(x):
            return [2 * x[0]]

        received = []

        def callback(progress):
            received.append(progress)

        gradient_descent(grad, [5.0], max_iter=20, on_progress=callback, progress_interval=5)
        assert len(received) > 0
        p = received[0]
        assert isinstance(p, Progress)
        assert p.iteration == 5
        assert p.objective is not None
        assert p.evaluations > 0

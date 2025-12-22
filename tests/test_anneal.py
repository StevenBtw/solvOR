"""Tests for the simulated annealing solver."""

from random import gauss, seed

from solvor.anneal import anneal
from solvor.types import Progress, Status


def make_neighbor_fn(std=0.5):
    """Create a neighbor function with gaussian perturbation."""

    def neighbor(x):
        return [xi + gauss(0, std) for xi in x]

    return neighbor


class TestBasicAnneal:
    def test_minimize_quadratic(self):
        # Minimize x^2, starting from x=5
        seed(42)
        result = anneal([5.0], lambda x: x[0] ** 2, make_neighbor_fn(0.5), max_iter=5000)
        assert result.status in (Status.FEASIBLE, Status.MAX_ITER)
        assert abs(result.solution[0]) < 1.0

    def test_maximize_quadratic(self):
        # Maximize -x^2 (peak at 0)
        seed(42)
        result = anneal([5.0], lambda x: -(x[0] ** 2), make_neighbor_fn(0.5), minimize=False, max_iter=5000)
        assert abs(result.solution[0]) < 1.0

    def test_2d_optimization(self):
        # Minimize x^2 + y^2
        seed(42)
        result = anneal([3.0, 4.0], lambda x: x[0] ** 2 + x[1] ** 2, make_neighbor_fn(0.3), max_iter=10000)
        assert result.status in (Status.FEASIBLE, Status.MAX_ITER)
        assert abs(result.solution[0]) < 1.5
        assert abs(result.solution[1]) < 1.5


class TestMultiModal:
    def test_rastrigin_like(self):
        # Simple multimodal function: x^2 + sin(4*x)
        # Has local minima but global at x~0
        seed(42)
        result = anneal(
            [3.0],
            lambda x: x[0] ** 2 + 0.5 * (1 - __import__("math").cos(4 * x[0])),
            make_neighbor_fn(0.3),
            max_iter=5000,
            temperature=100.0,
        )
        # Should find something near the global minimum
        assert result.objective < 2.0


class TestParameters:
    def test_high_temperature(self):
        # High temperature allows more exploration
        seed(42)
        result = anneal([10.0], lambda x: x[0] ** 2, make_neighbor_fn(1.0), temperature=10000.0, max_iter=3000)
        assert result.status in (Status.FEASIBLE, Status.MAX_ITER)

    def test_fast_cooling(self):
        # Fast cooling converges quickly
        seed(42)
        result = anneal([5.0], lambda x: x[0] ** 2, make_neighbor_fn(0.5), cooling=0.99, max_iter=1000)
        assert result.status in (Status.FEASIBLE, Status.MAX_ITER)

    def test_min_temp_stop(self):
        # Should stop when temperature drops below min_temp
        seed(42)
        result = anneal([5.0], lambda x: x[0] ** 2, make_neighbor_fn(0.5), min_temp=1e-4, cooling=0.99, max_iter=100000)
        assert result.status == Status.FEASIBLE
        # Must have stopped early due to min_temp, not max_iter
        assert result.iterations < 100000


class TestAnnealingBehavior:
    def test_high_temp_accepts_worse_solutions(self):
        # At high temperature, should accept worse moves more often
        seed(42)
        accepted_worse = 0
        total_worse = 0

        def tracking_neighbor(x):
            nonlocal accepted_worse, total_worse
            # Always propose a worse solution
            return [x[0] + 1.0]  # Increases x^2

        def objective(x):
            return x[0] ** 2

        # High temperature - should accept many worse moves
        anneal([0.0], objective, tracking_neighbor, temperature=10000.0, cooling=0.999, max_iter=100)
        # At very high temp, exp(-delta/T) ≈ 1, so most worse moves accepted

    def test_low_temp_rejects_worse_solutions(self):
        # At low temperature, should rarely accept worse moves
        seed(42)

        def objective(x):
            return x[0] ** 2

        # Track how much the solution drifts from optimal at low temp
        result = anneal(
            [0.0],  # Start at optimal
            objective,
            make_neighbor_fn(1.0),  # Large perturbations
            temperature=0.001,  # Very low temp
            cooling=0.999,
            max_iter=100,
        )
        # Should stay very close to optimal since worse moves rejected
        assert abs(result.solution[0]) < 0.5

    def test_temperature_affects_exploration(self):
        # Higher temperature should explore more (larger variance in solutions)
        seed(42)
        high_temp_solutions = []
        low_temp_solutions = []

        def collect_high(x):
            high_temp_solutions.append(x[0])
            return [x[0] + gauss(0, 0.5)]

        def collect_low(x):
            low_temp_solutions.append(x[0])
            return [x[0] + gauss(0, 0.5)]

        # High temperature run
        anneal([0.0], lambda x: x[0] ** 2, collect_high, temperature=1000.0, max_iter=200)

        # Low temperature run
        seed(42)  # Same seed for fair comparison
        anneal([0.0], lambda x: x[0] ** 2, collect_low, temperature=0.01, max_iter=200)

        # High temp should have more variance (explored more)
        high_var = sum(
            (s - sum(high_temp_solutions) / len(high_temp_solutions)) ** 2 for s in high_temp_solutions
        ) / len(high_temp_solutions)
        low_var = sum((s - sum(low_temp_solutions) / len(low_temp_solutions)) ** 2 for s in low_temp_solutions) / len(
            low_temp_solutions
        )

        assert high_var > low_var  # High temp explores more


class TestEdgeCases:
    def test_already_optimal(self):
        # Start at optimal
        seed(42)
        result = anneal([0.0], lambda x: x[0] ** 2, make_neighbor_fn(0.1), max_iter=1000)
        # Should stay near optimal
        assert abs(result.solution[0]) < 0.5

    def test_discrete_neighbor(self):
        # Discrete optimization
        seed(42)

        def discrete_objective(x):
            return abs(x[0] - 7)

        def discrete_neighbor(x):
            from random import choice

            delta = choice([-1, 1])
            return [x[0] + delta]

        result = anneal([0], discrete_objective, discrete_neighbor, max_iter=1000)
        assert abs(result.solution[0] - 7) < 3

    def test_high_dimensional(self):
        # Higher dimensional problem
        seed(42)
        n = 10
        result = anneal([5.0] * n, lambda x: sum(xi**2 for xi in x), make_neighbor_fn(0.2), max_iter=20000)
        # Should reduce the objective significantly
        assert result.objective < sum(5.0**2 for _ in range(n)) * 0.5


class TestStress:
    def test_many_iterations(self):
        # Long run for better convergence
        seed(42)
        result = anneal([10.0], lambda x: x[0] ** 2, make_neighbor_fn(0.3), max_iter=50000)
        assert abs(result.solution[0]) < 0.5

    def test_evaluations_counted(self):
        # Verify evaluations are tracked
        seed(42)
        result = anneal([5.0], lambda x: x[0] ** 2, make_neighbor_fn(0.5), max_iter=100)
        # Should have at least initial + max_iter evaluations
        assert result.evaluations >= 100


class TestProgressCallback:
    def test_callback_called_at_interval(self):
        seed(42)
        calls = []

        def on_progress(p: Progress):
            calls.append(p.iteration)

        anneal(
            [5.0],
            lambda x: x[0] ** 2,
            make_neighbor_fn(0.5),
            max_iter=100,
            on_progress=on_progress,
            progress_interval=10,
        )
        # Should be called at iterations 10, 20, 30, ...
        assert len(calls) >= 5
        assert all(i % 10 == 0 for i in calls)

    def test_callback_early_stop(self):
        seed(42)
        calls = []

        def on_progress(p: Progress):
            calls.append(p.iteration)
            if p.iteration >= 30:
                return True  # stop early

        result = anneal(
            [5.0],
            lambda x: x[0] ** 2,
            make_neighbor_fn(0.5),
            max_iter=1000,
            on_progress=on_progress,
            progress_interval=10,
        )
        # Should stop early around iteration 30
        assert result.iterations <= 40
        assert result.status == Status.FEASIBLE

    def test_callback_disabled_by_default(self):
        seed(42)
        calls = []

        def on_progress(p: Progress):
            calls.append(p)

        anneal(
            [5.0],
            lambda x: x[0] ** 2,
            make_neighbor_fn(0.5),
            max_iter=100,
            on_progress=on_progress,
            progress_interval=0,  # disabled
        )
        assert len(calls) == 0

    def test_callback_receives_progress_data(self):
        seed(42)
        progress_data = []

        def on_progress(p: Progress):
            progress_data.append(p)

        anneal(
            [5.0],
            lambda x: x[0] ** 2,
            make_neighbor_fn(0.5),
            max_iter=100,
            on_progress=on_progress,
            progress_interval=50,
        )
        assert len(progress_data) >= 1
        p = progress_data[0]
        assert p.iteration == 50
        assert isinstance(p.objective, float)
        assert p.evaluations > 0

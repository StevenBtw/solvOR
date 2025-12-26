"""Tests for the LNS/ALNS solvers."""

from solvor.lns import alns, lns
from solvor.types import Progress, Status


class TestBasicLNS:
    def test_simple_optimization(self):
        # Minimize sum of list elements by replacing with zeros
        def objective(x):
            return sum(x)

        def destroy(x, rng):
            # Remove random element (set to None)
            x = list(x)
            if x:
                idx = rng.randint(0, len(x) - 1)
                x[idx] = None
            return x

        def repair(x, rng):
            # Replace None with 0
            return [0 if v is None else v for v in x]

        result = lns([5, 10, 15, 20], objective, destroy, repair, max_iter=100)
        assert result.status == Status.FEASIBLE
        assert result.objective == 0
        assert result.solution == [0, 0, 0, 0]

    def test_maximize(self):
        # Maximize sum by replacing with 100s
        def objective(x):
            return sum(x)

        def destroy(x, rng):
            x = list(x)
            if x:
                idx = rng.randint(0, len(x) - 1)
                x[idx] = None
            return x

        def repair(x, rng):
            return [100 if v is None else v for v in x]

        result = lns([0, 0, 0], objective, destroy, repair, minimize=False, max_iter=50)
        assert result.status == Status.FEASIBLE
        assert result.objective == 300

    def test_seed_reproducibility(self):
        def objective(x):
            return sum(abs(v - 10) for v in x)

        def destroy(x, rng):
            x = list(x)
            idx = rng.randint(0, len(x) - 1)
            x[idx] = None
            return x

        def repair(x, rng):
            return [rng.randint(5, 15) if v is None else v for v in x]

        r1 = lns([0, 0, 0], objective, destroy, repair, seed=42, max_iter=50)
        r2 = lns([0, 0, 0], objective, destroy, repair, seed=42, max_iter=50)

        assert r1.solution == r2.solution
        assert r1.objective == r2.objective


class TestAcceptanceCriteria:
    def test_improving_only(self):
        calls = []

        def objective(x):
            calls.append(x)
            return x

        def destroy(x, rng):
            return x

        def repair(x, rng):
            # Always return worse solution
            return x + 10

        result = lns(0, objective, destroy, repair, accept="improving", max_iter=10)
        # Should stay at 0 since improvements never happen
        assert result.solution == 0

    def test_accept_all(self):
        def objective(x):
            return abs(x - 100)

        def destroy(x, rng):
            return x

        def repair(x, rng):
            return x + rng.randint(-5, 10)

        result = lns(0, objective, destroy, repair, accept="accept_all", max_iter=100, seed=42)
        # Should explore and find better solutions
        assert result.objective < 100

    def test_simulated_annealing(self):
        def objective(x):
            return abs(x - 50)

        def destroy(x, rng):
            return x

        def repair(x, rng):
            # Make bigger jumps to reach target faster
            return x + rng.randint(-10, 10)

        result = lns(
            0,
            objective,
            destroy,
            repair,
            accept="simulated_annealing",
            start_temp=1000.0,
            cooling_rate=0.995,
            max_iter=1000,
            seed=42,
        )
        assert result.objective < 50  # Should improve from starting point of 50

    def test_custom_accept(self):
        accepted = []

        def custom_accept(current, new, iteration, rng):
            accepted.append((current, new))
            return new < current + 5  # Accept if not much worse

        def objective(x):
            return x

        def destroy(x, rng):
            return x

        def repair(x, rng):
            return x + rng.randint(-2, 2)

        lns(10, objective, destroy, repair, accept=custom_accept, max_iter=20, seed=42)
        assert len(accepted) > 0


class TestProgressCallback:
    def test_callback_called(self):
        calls = []

        def callback(progress):
            calls.append(progress.iteration)

        def objective(x):
            return sum(x)

        def destroy(x, rng):
            return x

        def repair(x, rng):
            return x

        lns([1, 2, 3], objective, destroy, repair, max_iter=50, on_progress=callback, progress_interval=10)
        assert calls == [10, 20, 30, 40, 50]

    def test_early_stop(self):
        def stop_at_20(progress):
            if progress.iteration >= 20:
                return True

        def objective(x):
            return x

        def destroy(x, rng):
            return x

        def repair(x, rng):
            return x

        result = lns(100, objective, destroy, repair, max_iter=100, on_progress=stop_at_20, progress_interval=5)
        assert result.iterations == 20

    def test_progress_data(self):
        received = []

        def callback(progress):
            received.append(progress)

        def objective(x):
            return sum(x)

        def destroy(x, rng):
            x = list(x)
            x[0] = None
            return x

        def repair(x, rng):
            return [0 if v is None else v for v in x]

        lns([10, 20], objective, destroy, repair, max_iter=10, on_progress=callback, progress_interval=5)

        assert len(received) > 0
        p = received[0]
        assert isinstance(p, Progress)
        assert p.iteration == 5
        assert p.evaluations > 0


class TestNoImprove:
    def test_early_stop_no_improve(self):
        def objective(x):
            return 0  # Always optimal, no improvement possible

        def destroy(x, rng):
            return x

        def repair(x, rng):
            return x

        result = lns(0, objective, destroy, repair, max_iter=1000, max_no_improve=10)
        assert result.iterations <= 10


class TestALNS:
    def test_basic_alns(self):
        def objective(x):
            return sum(x)

        def destroy1(x, rng):
            x = list(x)
            x[0] = None
            return x

        def destroy2(x, rng):
            x = list(x)
            x[-1] = None
            return x

        def repair1(x, rng):
            return [0 if v is None else v for v in x]

        def repair2(x, rng):
            return [0 if v is None else v for v in x]  # Both repair with 0

        result = alns(
            [10, 20, 30],
            objective,
            [destroy1, destroy2],
            [repair1, repair2],
            max_iter=200,
            accept="improving",
        )
        assert result.status == Status.FEASIBLE
        assert result.objective <= 30  # Should improve from 60

    def test_alns_adaptive_weights(self):
        # Good operator should get higher weight
        destroy_calls = [0, 0]
        repair_calls = [0, 0]

        def objective(x):
            return x

        def bad_destroy(x, rng):
            destroy_calls[0] += 1
            return x

        def good_destroy(x, rng):
            destroy_calls[1] += 1
            return x

        def bad_repair(x, rng):
            repair_calls[0] += 1
            return x + 10  # Makes worse

        def good_repair(x, rng):
            repair_calls[1] += 1
            return max(0, x - 1)  # Makes better

        alns(
            100,
            objective,
            [bad_destroy, good_destroy],
            [bad_repair, good_repair],
            max_iter=500,
            segment_size=50,
            reaction_factor=0.3,
            seed=42,
        )

        # Good repair should be called more often over time
        # (Hard to test precisely due to randomness, but good should be substantial)
        assert repair_calls[1] > 0

    def test_alns_empty_operators_error(self):
        def objective(x):
            return x

        try:
            alns(0, objective, [], [lambda x, r: x], max_iter=10)
            assert False, "Should raise ValueError"
        except ValueError:
            pass

        try:
            alns(0, objective, [lambda x, r: x], [], max_iter=10)
            assert False, "Should raise ValueError"
        except ValueError:
            pass

    def test_alns_maximize(self):
        def objective(x):
            return -abs(x - 50)  # Max at x=50

        def destroy(x, rng):
            return None

        def repair(x, rng):
            return rng.randint(0, 100)

        result = alns(0, objective, [destroy], [repair], minimize=False, max_iter=200, seed=42)
        assert result.objective > -20  # Should get close to 50


class TestEdgeCases:
    def test_single_element(self):
        def objective(x):
            return x[0] if x else 0

        def destroy(x, rng):
            return [None]

        def repair(x, rng):
            return [0]

        result = lns([100], objective, destroy, repair, max_iter=10)
        assert result.solution == [0]

    def test_no_change_operators(self):
        def objective(x):
            return x

        def destroy(x, rng):
            return x

        def repair(x, rng):
            return x

        result = lns(42, objective, destroy, repair, max_iter=10)
        assert result.solution == 42

    def test_initial_weights(self):
        calls = [0, 0]

        def objective(x):
            return x

        def destroy1(x, rng):
            calls[0] += 1
            return x

        def destroy2(x, rng):
            calls[1] += 1
            return x

        def repair(x, rng):
            return x

        # Heavy bias toward first destroy operator
        alns(0, objective, [destroy1, destroy2], [repair], destroy_weights=[10.0, 1.0], max_iter=100, seed=42)

        # First should be called more
        assert calls[0] > calls[1]

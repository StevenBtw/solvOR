"""Tests for helper utilities."""

from solvor.types import Progress
from solvor.utils import (
    Evaluator,
    assignment_cost,
    default_progress,
    is_feasible,
    pairwise_swap_neighbors,
    random_permutation,
    report_progress,
    timed_progress,
)


class TestAssignmentCost:
    def test_simple(self):
        matrix = [[1, 2], [3, 4]]
        assert assignment_cost(matrix, [0, 1]) == 5
        assert assignment_cost(matrix, [1, 0]) == 5

    def test_with_unassigned(self):
        matrix = [[1, 2], [3, 4]]
        assert assignment_cost(matrix, [0, -1]) == 1

    def test_empty(self):
        assert assignment_cost([], []) == 0

    def test_negative_index_ignored(self):
        matrix = [[1, 2], [3, 4]]
        assert assignment_cost(matrix, [-2, 0]) == 3  # -2 ignored, only [1][0]=3 counted

    def test_out_of_bounds(self):
        matrix = [[1, 2]]
        assert assignment_cost(matrix, [0, 5]) == 1  # row 1 doesn't exist, row 0 col 5 doesn't exist


class TestIsFeasible:
    def test_feasible(self):
        A = [[1, 1], [2, 1]]
        b = [4, 5]
        x = [1, 1]
        assert is_feasible(A, b, x) is True

    def test_infeasible(self):
        A = [[1, 1]]
        b = [2]
        x = [2, 2]
        assert is_feasible(A, b, x) is False

    def test_boundary(self):
        A = [[1, 0]]
        b = [5]
        x = [5, 0]
        assert is_feasible(A, b, x) is True

    def test_dimension_mismatch(self):
        A = [[1, 2, 3]]  # expects 3 variables
        b = [10]
        x = [1, 2]  # only 2 variables provided
        assert is_feasible(A, b, x) is True  # 1*1 + 2*2 = 5 <= 10


class TestRandomPermutation:
    def test_length(self):
        perm = random_permutation(10)
        assert len(perm) == 10

    def test_contains_all(self):
        perm = random_permutation(10)
        assert set(perm) == set(range(10))

    def test_empty(self):
        assert random_permutation(0) == []

    def test_single(self):
        assert random_permutation(1) == [0]


class TestPairwiseSwapNeighbors:
    def test_count(self):
        perm = [0, 1, 2]
        neighbors = list(pairwise_swap_neighbors(perm))
        assert len(neighbors) == 3

    def test_swaps(self):
        perm = [0, 1, 2]
        neighbors = list(pairwise_swap_neighbors(perm))
        assert [1, 0, 2] in neighbors
        assert [2, 1, 0] in neighbors
        assert [0, 2, 1] in neighbors

    def test_original_unchanged(self):
        perm = [0, 1, 2]
        list(pairwise_swap_neighbors(perm))
        assert perm == [0, 1, 2]


class TestTimedProgress:
    def test_receives_elapsed_time(self):
        """Callback receives elapsed time as second argument."""
        elapsed_times = []

        def callback(progress, elapsed):
            elapsed_times.append(elapsed)

        wrapped = timed_progress(callback)
        wrapped(Progress(iteration=1, objective=1.0))
        wrapped(Progress(iteration=2, objective=0.5))

        assert len(elapsed_times) == 2
        assert elapsed_times[0] >= 0
        assert elapsed_times[1] >= elapsed_times[0]

    def test_returns_callback_value(self):
        """Wrapper returns value from inner callback."""

        def stop_callback(progress, elapsed):
            return True

        def continue_callback(progress, elapsed):
            return None

        wrapped_stop = timed_progress(stop_callback)
        wrapped_continue = timed_progress(continue_callback)

        assert wrapped_stop(Progress(iteration=1, objective=1.0)) is True
        assert wrapped_continue(Progress(iteration=1, objective=1.0)) is None

    def test_time_based_stopping(self):
        """Can use elapsed time to stop optimization."""
        import time

        def time_limit_callback(progress, elapsed):
            return elapsed > 0.001  # Stop after 1ms

        wrapped = timed_progress(time_limit_callback)
        # First call might be quick enough
        time.sleep(0.002)  # Wait a bit
        result = wrapped(Progress(iteration=1, objective=1.0))
        assert result is True


class TestDefaultProgress:
    def test_creates_callback(self):
        """default_progress returns a callable."""
        cb = default_progress()
        assert callable(cb)

    def test_callback_returns_none(self, capsys):
        """Callback returns None when no time limit."""
        cb = default_progress(interval=1)
        result = cb(Progress(iteration=1, objective=1.0))
        assert result is None

    def test_prints_at_interval(self, capsys):
        """Prints progress at specified interval."""
        cb = default_progress("TEST", interval=2)
        cb(Progress(iteration=1, objective=5.0))
        cb(Progress(iteration=2, objective=4.0))
        cb(Progress(iteration=3, objective=3.0))
        cb(Progress(iteration=4, objective=2.0))

        captured = capsys.readouterr()
        # Should print at iterations 2 and 4 (multiples of interval)
        assert "iter=2" in captured.out
        assert "iter=4" in captured.out
        assert "iter=1" not in captured.out
        assert "iter=3" not in captured.out

    def test_includes_name_prefix(self, capsys):
        """Output includes solver name prefix."""
        cb = default_progress("PSO", interval=1)
        cb(Progress(iteration=1, objective=1.0))
        captured = capsys.readouterr()
        assert "PSO " in captured.out

    def test_time_limit_stops(self):
        """Returns True when time limit exceeded."""
        import time

        cb = default_progress(time_limit=0.001)
        time.sleep(0.002)
        result = cb(Progress(iteration=100, objective=1.0))
        assert result is True

    def test_shows_best_value(self, capsys):
        """Shows best value when provided."""
        cb = default_progress(interval=1)
        cb(Progress(iteration=1, objective=5.0, best=2.0))
        captured = capsys.readouterr()
        assert "best=2" in captured.out

    def test_uses_objective_as_best_when_none(self, capsys):
        """Uses objective as best when best is None."""
        cb = default_progress(interval=1)
        cb(Progress(iteration=1, objective=3.5))
        captured = capsys.readouterr()
        assert "best=3.5" in captured.out


class TestEvaluator:
    def test_minimize_mode(self):
        """Evaluator in minimize mode returns objective as-is."""
        evaluate = Evaluator(lambda x: x * 2, minimize=True)
        assert evaluate(5) == 10
        assert evaluate.evals == 1

    def test_maximize_mode(self):
        """Evaluator in maximize mode negates objective."""
        evaluate = Evaluator(lambda x: x * 2, minimize=False)
        assert evaluate(5) == -10
        assert evaluate.evals == 1

    def test_to_user_minimize(self):
        """to_user returns original value in minimize mode."""
        evaluate = Evaluator(lambda x: x, minimize=True)
        internal = evaluate(7)
        assert evaluate.to_user(internal) == 7

    def test_to_user_maximize(self):
        """to_user converts back to positive in maximize mode."""
        evaluate = Evaluator(lambda x: x, minimize=False)
        internal = evaluate(7)  # -7 internally
        assert evaluate.to_user(internal) == 7

    def test_tracks_evaluations(self):
        """Evaluator counts function evaluations."""
        evaluate = Evaluator(lambda x: x)
        evaluate(1)
        evaluate(2)
        evaluate(3)
        assert evaluate.evals == 3


class TestReportProgress:
    def test_no_callback(self):
        """Returns False when no callback provided."""
        result = report_progress(None, 10, 100, 5.0, 3.0, 50)
        assert result is False

    def test_interval_zero(self):
        """Returns False when interval is 0 (disabled)."""
        called = []
        result = report_progress(lambda p: called.append(p), 0, 100, 5.0, 3.0, 50)
        assert result is False
        assert len(called) == 0

    def test_not_at_interval(self):
        """Returns False when not at interval."""
        called = []
        result = report_progress(lambda p: called.append(p), 10, 15, 5.0, 3.0, 50)
        assert result is False
        assert len(called) == 0

    def test_at_interval_calls_callback(self):
        """Calls callback when at interval."""
        called = []
        result = report_progress(lambda p: called.append(p), 10, 20, 5.0, 3.0, 50)
        assert result is False
        assert len(called) == 1
        assert called[0].iteration == 20
        assert called[0].objective == 5.0
        assert called[0].best == 3.0
        assert called[0].evaluations == 50

    def test_callback_returns_true_stops(self):
        """Returns True when callback returns True."""
        result = report_progress(lambda p: True, 10, 10, 5.0, 3.0, 50)
        assert result is True

    def test_callback_returns_none_continues(self):
        """Returns False when callback returns None."""
        result = report_progress(lambda p: None, 10, 10, 5.0, 3.0, 50)
        assert result is False

    def test_best_same_as_current(self):
        """best is None when current equals best."""
        called = []
        report_progress(lambda p: called.append(p), 1, 1, 5.0, 5.0, 10)
        assert called[0].best is None

    def test_best_different_from_current(self):
        """best is set when different from current."""
        called = []
        report_progress(lambda p: called.append(p), 1, 1, 5.0, 3.0, 10)
        assert called[0].best == 3.0


class TestDebug:
    def test_debug_without_env(self, capsys):
        """debug() prints nothing when DEBUG not set."""
        from solvor.utils import debug

        debug("test message")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_debug_with_env(self, capsys, monkeypatch):
        """debug() prints when DEBUG=1."""
        monkeypatch.setenv("DEBUG", "1")
        import importlib

        import solvor.utils.helpers

        importlib.reload(solvor.utils.helpers)

        solvor.utils.helpers.debug("test message")
        captured = capsys.readouterr()
        assert "test message" in captured.out

        # Cleanup
        monkeypatch.delenv("DEBUG", raising=False)
        importlib.reload(solvor.utils.helpers)

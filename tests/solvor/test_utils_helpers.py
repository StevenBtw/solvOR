"""Tests for helper utilities."""

from solvor.types import Progress
from solvor.utils import (
    assignment_cost,
    default_progress,
    is_feasible,
    pairwise_swap_neighbors,
    random_permutation,
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

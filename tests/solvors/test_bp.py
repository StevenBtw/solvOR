"""Tests for Branch-and-Price solver."""

import pytest

from solvor.bp import solve_bp
from solvor.types import Status


class TestCuttingStock:
    def test_simple(self):
        """Simple cutting stock problem."""
        result = solve_bp(
            demands=[4, 2],
            roll_width=10,
            piece_sizes=[6, 4],
        )
        assert result.ok
        assert result.objective >= 2
        # Verify integer solution
        for count in result.solution.values():
            assert count == int(count)

    def test_single_piece(self):
        """Single piece type - trivial case."""
        result = solve_bp(
            demands=[10],
            roll_width=100,
            piece_sizes=[25],
        )
        assert result.ok
        assert result.objective == 3  # ceil(10/4) = 3 rolls
        for count in result.solution.values():
            assert count == int(count)

    def test_exact_fit(self):
        """Pieces that exactly fill rolls."""
        result = solve_bp(
            demands=[4],
            roll_width=10,
            piece_sizes=[10],
        )
        assert result.ok
        assert result.objective == 4
        for count in result.solution.values():
            assert count == int(count)

    def test_gilmore_gomory(self):
        """Classic Gilmore-Gomory example."""
        result = solve_bp(
            demands=[97, 610, 395, 211],
            roll_width=100,
            piece_sizes=[45, 36, 31, 14],
        )
        assert result.ok
        # Verify all counts are integers
        for count in result.solution.values():
            assert count == int(count)
        # Should be close to optimal (453-454)
        assert result.objective <= 460

    def test_empty(self):
        """Empty problem."""
        result = solve_bp(demands=[], roll_width=100, piece_sizes=[])
        assert result.ok
        assert result.objective == 0
        assert result.solution == {}

    def test_zero_demands(self):
        """All zero demands."""
        result = solve_bp(
            demands=[0, 0, 0],
            roll_width=100,
            piece_sizes=[45, 36, 31],
        )
        assert result.ok
        assert result.objective == 0


class TestVsCG:
    def test_bp_at_least_as_good_as_cg(self):
        """B&P should be at least as good as CG rounding."""
        from solvor.cg import solve_cg

        demands = [10, 20, 15]
        kwargs = dict(roll_width=100, piece_sizes=[45, 36, 31])

        cg_result = solve_cg(demands, **kwargs)
        bp_result = solve_bp(demands, **kwargs)

        assert bp_result.ok
        assert bp_result.objective <= cg_result.objective


class TestBranching:
    def test_branching_occurs(self):
        """Verify branching happens for fractional LP."""
        result = solve_bp(
            demands=[3, 3],
            roll_width=10,
            piece_sizes=[6, 4],
        )
        assert result.ok
        for count in result.solution.values():
            assert count == int(count)

    def test_deep_branching(self):
        """Problem requiring deeper branching."""
        result = solve_bp(
            demands=[5, 7, 3],
            roll_width=20,
            piece_sizes=[8, 6, 5],
        )
        assert result.ok
        for count in result.solution.values():
            assert count == int(count)


class TestCustomPricing:
    def test_simple_custom(self):
        """Basic custom pricing - identity columns only."""
        def pricing(duals):
            return (None, 0.0)

        result = solve_bp(
            demands=[1, 1, 1],
            pricing_fn=pricing,
            initial_columns=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        )
        assert result.ok
        assert result.objective == 3

    def test_custom_with_better_column(self):
        """Custom pricing that finds improving column."""
        calls = [0]

        def pricing(duals):
            calls[0] += 1
            if calls[0] == 1:
                return ((2, 1, 0), -0.5)
            return (None, 0.0)

        result = solve_bp(
            demands=[2, 1, 1],
            pricing_fn=pricing,
            initial_columns=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        )
        assert result.ok
        assert result.objective <= 3


class TestValidation:
    def test_negative_demand(self):
        """Negative demand raises error."""
        with pytest.raises(ValueError, match="cannot be negative"):
            solve_bp(demands=[-1], roll_width=10, piece_sizes=[5])

    def test_piece_too_large(self):
        """Piece larger than roll raises error."""
        with pytest.raises(ValueError, match="exceeds"):
            solve_bp(demands=[1], roll_width=10, piece_sizes=[15])

    def test_negative_piece_size(self):
        """Negative piece size raises error."""
        with pytest.raises(ValueError, match="positive"):
            solve_bp(demands=[1], roll_width=10, piece_sizes=[-5])

    def test_length_mismatch(self):
        """Mismatched lengths raises error."""
        with pytest.raises(ValueError, match="mismatch"):
            solve_bp(demands=[1, 2], roll_width=10, piece_sizes=[5])

    def test_missing_mode(self):
        """Missing both modes raises error."""
        with pytest.raises(ValueError, match="Provide"):
            solve_bp(demands=[1])

    def test_both_modes(self):
        """Providing both modes raises error."""
        with pytest.raises(ValueError, match="not both"):
            solve_bp(
                demands=[1],
                roll_width=10,
                piece_sizes=[5],
                pricing_fn=lambda d: (None, 0),
                initial_columns=[(1,)],
            )

    def test_custom_missing_columns(self):
        """Custom mode without initial columns raises error."""
        with pytest.raises(ValueError, match="initial_columns"):
            solve_bp(demands=[1], pricing_fn=lambda d: (None, 0))


class TestEdgeCases:
    def test_large_demand(self):
        """Large demand values."""
        result = solve_bp(
            demands=[1000],
            roll_width=100,
            piece_sizes=[10],
        )
        assert result.ok
        assert result.objective == 100

    def test_max_nodes_limit(self):
        """Respects node limit."""
        result = solve_bp(
            demands=[97, 610, 395, 211],
            roll_width=100,
            piece_sizes=[45, 36, 31, 14],
            max_nodes=5,
        )
        # Should return best found even if not optimal
        assert result.solution is not None or result.status == Status.INFEASIBLE

    def test_already_integer_lp(self):
        """LP relaxation is already integer."""
        result = solve_bp(
            demands=[4],
            roll_width=10,
            piece_sizes=[10],
        )
        assert result.ok
        assert result.objective == 4
        assert result.status == Status.OPTIMAL


class TestProgress:
    def test_progress_callback(self):
        """Progress callback is called."""
        calls = []

        def callback(progress):
            calls.append(progress)
            return False

        solve_bp(
            demands=[10, 20, 15],
            roll_width=100,
            piece_sizes=[45, 36, 31],
            on_progress=callback,
            progress_interval=1,
        )
        assert len(calls) > 0

    def test_early_stop(self):
        """Early stop via progress callback."""
        def callback(progress):
            return True  # Stop immediately

        result = solve_bp(
            demands=[97, 610, 395, 211],
            roll_width=100,
            piece_sizes=[45, 36, 31, 14],
            on_progress=callback,
            progress_interval=1,
        )
        # Should stop early but still have a result from rounding
        assert result.solution is not None or result.iterations <= 2


class TestCustomEdgeCases:
    def test_column_length_mismatch(self):
        """Column with wrong length raises error."""
        with pytest.raises(ValueError, match="wrong length"):
            solve_bp(
                demands=[1, 2],
                pricing_fn=lambda d: (None, 0),
                initial_columns=[(1,)],  # Wrong length
            )

    def test_custom_finds_improving_column(self):
        """Custom pricing finds columns with negative reduced cost."""
        iteration = [0]

        def pricing(duals):
            iteration[0] += 1
            if iteration[0] <= 3:
                # Return column with negative reduced cost
                return ((1, 1), -0.1)
            return (None, 0.0)

        result = solve_bp(
            demands=[2, 2],
            pricing_fn=pricing,
            initial_columns=[(1, 0), (0, 1)],
        )
        assert result.ok


class TestInternalFunctions:
    def test_greedy_knapsack_fallback(self):
        """Test greedy knapsack is used as fallback."""
        from solvor.bp import _greedy_knapsack

        # Simple greedy test
        sizes = [3.0, 4.0, 5.0]
        capacity = 10.0
        values = [3.0, 4.0, 5.0]
        max_copies = [2, 2, 2]

        pattern, val = _greedy_knapsack(sizes, capacity, values, max_copies)
        assert sum(pattern[i] * sizes[i] for i in range(len(sizes))) <= capacity
        assert val > 0

    def test_greedy_with_zero_values(self):
        """Greedy handles zero/negative values."""
        from solvor.bp import _greedy_knapsack

        sizes = [3.0, 4.0]
        capacity = 10.0
        values = [0.0, -1.0]
        max_copies = [2, 2]

        pattern, val = _greedy_knapsack(sizes, capacity, values, max_copies)
        assert pattern == (0, 0)
        assert val == 0.0

    def test_knapsack_empty(self):
        """Knapsack with no items."""
        from solvor.bp import _knapsack_pricing

        pattern, val = _knapsack_pricing([], 100.0, [], 1e-9)
        assert pattern == ()
        assert val == 0.0

    def test_knapsack_zero_values(self):
        """Knapsack with zero dual values."""
        from solvor.bp import _knapsack_pricing

        sizes = [10.0, 20.0]
        capacity = 100.0
        values = [0.0, 0.0]

        pattern, val = _knapsack_pricing(sizes, capacity, values, 1e-9)
        assert val == 0.0

    def test_bounded_master_empty(self):
        """Master LP with no columns."""
        from solvor.bp import _solve_bounded_master_lp

        x_vals, duals, obj = _solve_bounded_master_lp([], [1, 2], {}, 1e-9)
        assert x_vals == []
        assert obj == float("inf")

    def test_most_fractional_all_integer(self):
        """Most fractional returns None for integer solution."""
        from solvor.bp import _most_fractional

        x_vals = [1.0, 2.0, 3.0, 0.0]
        idx, val = _most_fractional(x_vals, 1e-9)
        assert idx is None
        assert val is None

    def test_build_solution(self):
        """Build solution from x values."""
        from solvor.bp import _build_solution

        columns = [(1, 0), (0, 1), (1, 1)]
        x_vals = [2.0, 0.0, 1.0]
        solution = _build_solution(x_vals, columns, 1e-9)
        assert solution == {(1, 0): 2, (1, 1): 1}

    def test_round_solution_infeasible(self):
        """Round solution returns None if infeasible."""
        from solvor.bp import _round_solution

        columns = [(1, 0), (0, 1)]
        x_vals = [0.5, 0.5]
        demands = [10, 10]  # Can't meet with rounded values
        result = _round_solution(x_vals, columns, demands, 1e-9)
        assert result is None

"""Tests for Column Generation solver."""

import pytest

from solvor.cg import solve_cg
from solvor.types import Status


class TestCuttingStock:
    def test_simple(self):
        """Simple cutting stock problem."""
        result = solve_cg(
            demands=[4, 2],
            roll_width=10,
            piece_sizes=[6, 4],
        )
        assert result.ok
        # Need to produce at least 4 of size 6 and 2 of size 4
        for pattern, count in result.solution.items():
            assert count > 0
        # Verify demands met
        produced = [0, 0]
        for pattern, count in result.solution.items():
            produced[0] += pattern[0] * count
            produced[1] += pattern[1] * count
        assert produced[0] >= 4
        assert produced[1] >= 2

    def test_classic_gilmore_gomory(self):
        """Classic Gilmore-Gomory example from 1961 paper."""
        result = solve_cg(
            demands=[97, 610, 395, 211],
            roll_width=100,
            piece_sizes=[45, 36, 31, 14],
        )
        assert result.ok
        # Optimal is 453 rolls
        assert result.objective <= 460  # Allow small rounding gap

    def test_single_piece(self):
        """Single piece type."""
        result = solve_cg(
            demands=[10],
            roll_width=100,
            piece_sizes=[25],
        )
        assert result.ok
        assert result.objective >= 3  # ceil(10/4) = 3 rolls needed

    def test_exact_fit(self):
        """Pieces that fit exactly."""
        result = solve_cg(
            demands=[5],
            roll_width=10,
            piece_sizes=[10],
        )
        assert result.ok
        assert result.objective == 5  # One piece per roll

    def test_multiple_pieces_per_roll(self):
        """Multiple pieces fit per roll."""
        result = solve_cg(
            demands=[10],
            roll_width=100,
            piece_sizes=[10],
        )
        assert result.ok
        assert result.objective == 1  # All 10 pieces fit in one roll

    def test_empty(self):
        """Empty demands."""
        result = solve_cg(demands=[], roll_width=100, piece_sizes=[])
        assert result.ok
        assert result.objective == 0
        assert result.solution == {}

    def test_zero_demands(self):
        """All zero demands."""
        result = solve_cg(
            demands=[0, 0, 0],
            roll_width=100,
            piece_sizes=[10, 20, 30],
        )
        assert result.ok
        assert result.objective == 0


class TestCustomPricing:
    def test_simple_custom(self):
        """Simple custom pricing function."""
        # Simple set covering: each column covers one constraint
        call_count = [0]

        def pricing(duals):
            call_count[0] += 1
            # After first iteration, no improving column
            return (None, 0.0)

        result = solve_cg(
            demands=[1, 1, 1],
            pricing_fn=pricing,
            initial_columns=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        )
        assert result.ok
        assert result.objective == 3

    def test_custom_with_better_column(self):
        """Custom pricing finds an improving column."""
        found_better = [False]

        def pricing(duals):
            if not found_better[0]:
                found_better[0] = True
                # Return column covering all constraints with negative reduced cost
                return ((1, 1, 1), -0.5)
            return (None, 0.0)

        result = solve_cg(
            demands=[1, 1, 1],
            pricing_fn=pricing,
            initial_columns=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        )
        assert result.ok
        # Should use the combined column
        assert result.objective <= 3


class TestValidation:
    def test_negative_demand(self):
        """Negative demand raises error."""
        with pytest.raises(ValueError, match="cannot be negative"):
            solve_cg(demands=[-1], roll_width=10, piece_sizes=[5])

    def test_piece_too_large(self):
        """Piece larger than roll raises error."""
        with pytest.raises(ValueError, match="exceeds"):
            solve_cg(demands=[1], roll_width=10, piece_sizes=[15])

    def test_negative_piece_size(self):
        """Negative piece size raises error."""
        with pytest.raises(ValueError, match="positive"):
            solve_cg(demands=[1], roll_width=10, piece_sizes=[-5])

    def test_length_mismatch(self):
        """Mismatched lengths raise error."""
        with pytest.raises(ValueError, match="mismatch"):
            solve_cg(demands=[1, 2], roll_width=10, piece_sizes=[5])

    def test_missing_mode(self):
        """Neither cutting stock nor custom mode raises error."""
        with pytest.raises(ValueError, match="Provide"):
            solve_cg(demands=[1, 2, 3])

    def test_both_modes(self):
        """Both modes specified raises error."""
        with pytest.raises(ValueError, match="not both"):
            solve_cg(
                demands=[1],
                roll_width=10,
                piece_sizes=[5],
                pricing_fn=lambda x: (None, 0),
            )

    def test_custom_missing_columns(self):
        """Custom mode without initial columns raises error."""
        with pytest.raises(ValueError, match="initial_columns"):
            solve_cg(
                demands=[1],
                pricing_fn=lambda x: (None, 0),
            )


class TestEdgeCases:
    def test_large_demand(self):
        """Large demand for single piece type."""
        result = solve_cg(
            demands=[100],
            roll_width=10,
            piece_sizes=[3],
        )
        assert result.ok
        # 3 pieces per roll, need 100, so ceil(100/3) = 34 rolls
        assert result.objective >= 34

    def test_fractional_sizes(self):
        """Fractional piece sizes."""
        result = solve_cg(
            demands=[10, 10],
            roll_width=10.0,
            piece_sizes=[2.5, 3.5],
        )
        assert result.ok
        # Verify demands met
        produced = [0, 0]
        for pattern, count in result.solution.items():
            produced[0] += pattern[0] * count
            produced[1] += pattern[1] * count
        assert produced[0] >= 10
        assert produced[1] >= 10

    def test_many_piece_types(self):
        """Many different piece types."""
        n = 10
        result = solve_cg(
            demands=[5] * n,
            roll_width=100,
            piece_sizes=[10 + i for i in range(n)],
        )
        assert result.ok


class TestCustomEdgeCases:
    def test_column_length_mismatch(self):
        """Column length doesn't match demands."""
        with pytest.raises(ValueError, match="wrong length"):
            solve_cg(
                demands=[1, 2, 3],
                pricing_fn=lambda x: (None, 0),
                initial_columns=[(1, 0)],  # Wrong length
            )

    def test_custom_with_duplicate_column(self):
        """Pricing returns a column that already exists."""
        call_count = [0]

        def pricing(duals):
            call_count[0] += 1
            if call_count[0] == 1:
                # Return existing column
                return ((1, 0, 0), -0.5)
            return (None, 0.0)

        result = solve_cg(
            demands=[1, 1, 1],
            pricing_fn=pricing,
            initial_columns=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        )
        assert result.ok

    def test_custom_many_iterations(self):
        """Custom pricing with multiple improving columns."""
        iteration = [0]

        def pricing(duals):
            iteration[0] += 1
            if iteration[0] <= 3:
                # Generate different columns
                col = [0, 0, 0]
                col[iteration[0] - 1] = 2
                return (tuple(col), -0.1)
            return (None, 0.0)

        result = solve_cg(
            demands=[2, 2, 2],
            pricing_fn=pricing,
            initial_columns=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        )
        assert result.ok


class TestProgress:
    def test_progress_callback(self):
        """Progress callback is called."""
        iterations = []

        def on_progress(p):
            iterations.append(p.iteration)
            return False

        result = solve_cg(
            demands=[10, 20, 15],
            roll_width=100,
            piece_sizes=[45, 36, 31],
            on_progress=on_progress,
            progress_interval=1,
        )
        assert result.ok
        assert len(iterations) > 0

    def test_early_stop(self):
        """Progress callback can stop early."""
        def on_progress(p):
            return True  # Stop immediately

        result = solve_cg(
            demands=[100, 200, 150],
            roll_width=100,
            piece_sizes=[45, 36, 31],
            on_progress=on_progress,
            progress_interval=1,
        )
        assert result.iterations == 0

    def test_custom_progress_callback(self):
        """Progress callback works in custom mode."""
        iterations = []

        def on_progress(p):
            iterations.append(p.iteration)
            return False

        iter_count = [0]

        def pricing(duals):
            iter_count[0] += 1
            if iter_count[0] <= 2:
                col = [0, 0, 0]
                col[iter_count[0] - 1] = 2
                return (tuple(col), -0.1)
            return (None, 0.0)

        result = solve_cg(
            demands=[2, 2, 2],
            pricing_fn=pricing,
            initial_columns=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
            on_progress=on_progress,
            progress_interval=1,
        )
        assert result.ok
        assert len(iterations) > 0

    def test_custom_early_stop(self):
        """Progress callback can stop custom mode early."""
        def on_progress(p):
            return True  # Stop immediately

        def pricing(duals):
            return ((2, 2, 2), -0.5)  # Would keep generating

        result = solve_cg(
            demands=[2, 2, 2],
            pricing_fn=pricing,
            initial_columns=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
            on_progress=on_progress,
            progress_interval=1,
        )
        assert result.iterations == 0


class TestInternalFunctions:
    def test_knapsack_empty(self):
        """Knapsack with empty inputs."""
        from solvor.cg import _knapsack_pricing

        pattern, value = _knapsack_pricing([], 100, [], 1e-9)
        assert pattern == ()
        assert value == 0.0

    def test_knapsack_zero_values(self):
        """Knapsack with all zero values."""
        from solvor.cg import _knapsack_pricing

        pattern, value = _knapsack_pricing([10, 20], 100, [0.0, 0.0], 1e-9)
        assert pattern == (0, 0)
        assert value == 0.0

    def test_greedy_knapsack(self):
        """Test greedy knapsack fallback directly."""
        from solvor.cg import _greedy_knapsack

        # Test with valid inputs
        pattern, value = _greedy_knapsack(
            sizes=[10, 20, 30],
            capacity=100,
            values=[1.0, 2.0, 3.0],
            max_copies=[10, 5, 3],
        )
        assert sum(pattern[i] * [10, 20, 30][i] for i in range(3)) <= 100
        assert value > 0

    def test_greedy_with_zero_values(self):
        """Greedy with zero or negative values."""
        from solvor.cg import _greedy_knapsack

        pattern, value = _greedy_knapsack(
            sizes=[10, 20],
            capacity=100,
            values=[0.0, -1.0],
            max_copies=[10, 5],
        )
        assert pattern == (0, 0)
        assert value == 0.0

    def test_master_lp_empty_columns(self):
        """Master LP with no columns."""
        from solvor.cg import _solve_master_lp

        x_vals, duals, obj = _solve_master_lp([], [1, 2, 3], 1e-9)
        assert x_vals == []
        assert len(duals) == 3
        assert obj == float("inf")

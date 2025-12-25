"""Tests for 0/1 Knapsack solver."""

import pytest

from solvor.knapsack import solve_knapsack
from solvor.types import Status


class TestBasicKnapsack:
    def test_simple(self):
        """Simple 3-item knapsack."""
        values = [60, 100, 120]
        weights = [10, 20, 30]
        capacity = 50

        result = solve_knapsack(values, weights, capacity)
        assert result.status == Status.OPTIMAL
        assert result.objective == 220  # items 1 and 2
        assert set(result.solution) == {1, 2}

    def test_all_fit(self):
        """All items fit in knapsack."""
        values = [10, 20, 30]
        weights = [1, 2, 3]
        capacity = 10

        result = solve_knapsack(values, weights, capacity)
        assert result.objective == 60
        assert set(result.solution) == {0, 1, 2}

    def test_single_item(self):
        """Single item that fits."""
        values = [100]
        weights = [5]
        capacity = 10

        result = solve_knapsack(values, weights, capacity)
        assert result.objective == 100
        assert result.solution == (0,)

    def test_single_item_too_heavy(self):
        """Single item that doesn't fit."""
        values = [100]
        weights = [15]
        capacity = 10

        result = solve_knapsack(values, weights, capacity)
        assert result.objective == 0
        assert result.solution == ()

    def test_empty(self):
        """Empty item list."""
        result = solve_knapsack([], [], 100)
        assert result.objective == 0
        assert result.solution == ()
        assert result.status == Status.OPTIMAL


class TestMinimize:
    def test_minimize_takes_nothing(self):
        """Minimize with no requirement takes nothing (optimal = 0)."""
        values = [100, 10, 50]
        weights = [5, 5, 5]
        capacity = 10

        result = solve_knapsack(values, weights, capacity, minimize=True)
        # Without a requirement to take items, minimize takes nothing
        assert result.solution == ()
        assert result.objective == 0

    def test_minimize_with_negative_values(self):
        """Minimize takes items with most negative values."""
        values = [-100, -10, -50]  # Negative values
        weights = [5, 5, 5]
        capacity = 10

        result = solve_knapsack(values, weights, capacity, minimize=True)
        # Should take items 0 and 2 (most negative values that fit)
        assert 0 in result.solution  # -100 is most negative
        assert 2 in result.solution  # -50 is second most negative
        assert result.objective == -150  # -100 + -50

    def test_minimize_nothing_fits(self):
        """Minimize when nothing fits."""
        values = [100, 200, 300]
        weights = [10, 10, 10]
        capacity = 5  # Nothing fits

        result = solve_knapsack(values, weights, capacity, minimize=True)
        assert result.solution == ()
        assert result.objective == 0


class TestEdgeCases:
    def test_zero_capacity(self):
        """Zero capacity means nothing can be taken."""
        values = [10, 20, 30]
        weights = [1, 2, 3]
        capacity = 0

        result = solve_knapsack(values, weights, capacity)
        assert result.objective == 0
        assert result.solution == ()

    def test_zero_weight_item(self):
        """Item with zero weight is always taken."""
        values = [10, 20, 30]
        weights = [0, 5, 10]
        capacity = 10

        result = solve_knapsack(values, weights, capacity)
        assert 0 in result.solution  # zero weight, should take it
        assert result.objective >= 10

    def test_exact_fit(self):
        """Items exactly fill capacity."""
        values = [50, 60]
        weights = [25, 25]
        capacity = 50

        result = solve_knapsack(values, weights, capacity)
        assert result.objective == 110
        assert set(result.solution) == {0, 1}

    def test_fractional_weights(self):
        """Handles fractional weights via scaling."""
        values = [100, 200]
        weights = [1.5, 2.5]
        capacity = 3.0

        result = solve_knapsack(values, weights, capacity)
        # Should take item 0 (weight 1.5) and not item 1 (would exceed)
        # or take item 1 only
        assert result.objective >= 100
        total_weight = sum(weights[i] for i in result.solution)
        assert total_weight <= capacity + 1e-9


class TestClassicProblems:
    def test_textbook_example(self):
        """Classic textbook knapsack problem."""
        values = [1, 6, 18, 22, 28]
        weights = [1, 2, 5, 6, 7]
        capacity = 11

        result = solve_knapsack(values, weights, capacity)
        assert result.status == Status.OPTIMAL
        # Optimal: items 3 and 4 (values 22+28=50, weights 6+7=13 > 11)
        # Actually: items 2 and 3 (values 18+22=40, weights 5+6=11)
        # Or: items 1 and 4 (values 6+28=34, weights 2+7=9)
        # Or: items 0, 1, 2 (values 1+6+18=25, weights 1+2+5=8)
        # Best is items 2+3 = 40
        assert result.objective == 40

    def test_high_value_low_weight(self):
        """Prefer high value/weight ratio items."""
        values = [10, 40, 50, 70]
        weights = [1, 3, 4, 5]
        capacity = 8

        result = solve_knapsack(values, weights, capacity)
        # Optimal: items 1 and 3 (40+70=110, weights 3+5=8)
        assert result.objective == 110


class TestValidation:
    def test_mismatched_lengths(self):
        """Values and weights must have same length."""
        with pytest.raises(ValueError, match="Length mismatch"):
            solve_knapsack([1, 2, 3], [1, 2], 10)

    def test_negative_capacity(self):
        """Negative capacity raises error."""
        with pytest.raises(ValueError, match="cannot be negative"):
            solve_knapsack([1, 2], [1, 2], -5)


class TestLargeInstances:
    def test_many_items(self):
        """Handles larger instances."""
        n = 100
        values = list(range(1, n + 1))
        weights = [1] * n
        capacity = 50

        result = solve_knapsack(values, weights, capacity)
        # Should take the 50 highest value items (51-100)
        assert result.objective == sum(range(51, 101))
        assert len(result.solution) == 50

    def test_large_capacity(self):
        """Handles large capacity values."""
        values = [100, 200, 300]
        weights = [10, 20, 30]
        capacity = 1000

        result = solve_knapsack(values, weights, capacity)
        # All items fit
        assert result.objective == 600
        assert set(result.solution) == {0, 1, 2}


class TestScaling:
    """Test capacity scaling logic in _to_int_capacity."""

    def test_integer_weights_no_scaling(self):
        """Integer weights should not require scaling."""
        values = [10, 20, 30]
        weights = [5, 10, 15]
        capacity = 20

        result = solve_knapsack(values, weights, capacity)
        assert result.ok
        # Check weight constraint
        total_weight = sum(weights[i] for i in result.solution)
        assert total_weight <= capacity

    def test_fractional_weights_scaling(self):
        """Fractional weights require scaling."""
        values = [100, 200, 150]
        weights = [1.5, 2.5, 2.0]
        capacity = 4.0

        result = solve_knapsack(values, weights, capacity)
        assert result.ok
        total_weight = sum(weights[i] for i in result.solution)
        assert total_weight <= capacity + 1e-9

    def test_very_small_fractional_weights(self):
        """Very small fractional weights."""
        values = [10, 20, 30]
        weights = [0.1, 0.2, 0.3]
        capacity = 0.5

        result = solve_knapsack(values, weights, capacity)
        assert result.ok
        total_weight = sum(weights[i] for i in result.solution)
        assert total_weight <= capacity + 1e-9

    def test_mixed_integer_and_float_weights(self):
        """Mix of integer and floating point weights."""
        values = [50, 60, 70]
        weights = [1, 1.5, 2]
        capacity = 3

        result = solve_knapsack(values, weights, capacity)
        assert result.ok
        total_weight = sum(weights[i] for i in result.solution)
        assert total_weight <= capacity + 1e-9


class TestGreedyFallback:
    """Test the greedy fallback function."""

    def test_fallback_maximize(self):
        """Test greedy fallback for maximization."""
        from solvor.knapsack import _greedy_fallback

        values = [60, 100, 120]
        weights = [10, 20, 30]
        capacity = 50

        result = _greedy_fallback(values, weights, capacity, minimize=False)
        assert result.ok or result.status == Status.FEASIBLE
        total_weight = sum(weights[i] for i in result.solution)
        assert total_weight <= capacity

    def test_fallback_minimize(self):
        """Test greedy fallback for minimization."""
        from solvor.knapsack import _greedy_fallback

        values = [60, 100, 120]
        weights = [10, 20, 30]
        capacity = 50

        result = _greedy_fallback(values, weights, capacity, minimize=True)
        total_weight = sum(weights[i] for i in result.solution)
        assert total_weight <= capacity

    def test_fallback_empty(self):
        """Test greedy fallback with empty items."""
        from solvor.knapsack import _greedy_fallback

        result = _greedy_fallback([], [], 100, minimize=False)
        assert result.solution == ()
        assert result.objective == 0

    def test_fallback_nothing_fits(self):
        """Test greedy fallback when no items fit."""
        from solvor.knapsack import _greedy_fallback

        values = [100, 200]
        weights = [50, 60]
        capacity = 10

        result = _greedy_fallback(values, weights, capacity, minimize=False)
        assert result.solution == ()


class TestIntCapacityConversion:
    """Test the _to_int_capacity helper function."""

    def test_integer_capacity(self):
        """Integer capacity returns itself with scale 1."""
        from solvor.knapsack import _to_int_capacity

        int_cap, scale = _to_int_capacity(100, [10, 20, 30])
        assert int_cap == 100
        assert scale == 1.0

    def test_zero_capacity(self):
        """Zero capacity returns 0."""
        from solvor.knapsack import _to_int_capacity

        int_cap, scale = _to_int_capacity(0, [1, 2, 3])
        assert int_cap == 0

    def test_float_capacity(self):
        """Float capacity gets scaled."""
        from solvor.knapsack import _to_int_capacity

        int_cap, scale = _to_int_capacity(10.5, [1.5, 2.5])
        assert int_cap > 0
        assert scale >= 1.0

    def test_large_capacity_limits_scale(self):
        """Very large capacity limits the scale factor."""
        from solvor.knapsack import _to_int_capacity

        int_cap, scale = _to_int_capacity(50000.5, [100.5, 200.5])
        # Scale should be limited to avoid huge DP tables
        assert int_cap <= 100001  # max_capacity limit


class TestZeroValueItems:
    """Test handling of items with zero values."""

    def test_zero_value_item_not_preferred(self):
        """Zero value item shouldn't be taken if capacity needed."""
        values = [0, 100]
        weights = [50, 50]
        capacity = 50

        result = solve_knapsack(values, weights, capacity)
        assert result.objective == 100
        assert 1 in result.solution

    def test_all_zero_values(self):
        """All items have zero value."""
        values = [0, 0, 0]
        weights = [10, 20, 30]
        capacity = 50

        result = solve_knapsack(values, weights, capacity)
        assert result.objective == 0

    def test_zero_value_zero_weight(self):
        """Item with zero value and zero weight."""
        values = [0, 100]
        weights = [0, 10]
        capacity = 10

        result = solve_knapsack(values, weights, capacity)
        # Zero weight item might be taken, but zero value doesn't add
        assert result.objective >= 100


class TestAllItemsTooHeavy:
    """Test when all items exceed capacity."""

    def test_all_items_too_heavy(self):
        """No items can fit."""
        values = [100, 200, 300]
        weights = [50, 60, 70]
        capacity = 10

        result = solve_knapsack(values, weights, capacity)
        assert result.objective == 0
        assert result.solution == ()
        assert result.status == Status.OPTIMAL


class TestEqualValueWeight:
    """Test items with equal value/weight ratios."""

    def test_same_ratio_items(self):
        """All items have same value/weight ratio."""
        values = [10, 20, 30]
        weights = [1, 2, 3]  # All ratio = 10
        capacity = 4

        result = solve_knapsack(values, weights, capacity)
        assert result.ok
        # Should take items that fit best
        total_weight = sum(weights[i] for i in result.solution)
        assert total_weight <= capacity

    def test_identical_items(self):
        """All items identical."""
        values = [10, 10, 10]
        weights = [5, 5, 5]
        capacity = 10

        result = solve_knapsack(values, weights, capacity)
        assert result.objective == 20  # Can fit 2 items
        assert len(result.solution) == 2


class TestBoundaryConditions:
    """Test boundary conditions in DP."""

    def test_exact_capacity_match(self):
        """Weight exactly equals capacity."""
        values = [100]
        weights = [50]
        capacity = 50

        result = solve_knapsack(values, weights, capacity)
        assert result.objective == 100
        assert result.solution == (0,)

    def test_one_over_capacity(self):
        """Weight is one unit over capacity."""
        values = [100]
        weights = [51]
        capacity = 50

        result = solve_knapsack(values, weights, capacity)
        assert result.objective == 0
        assert result.solution == ()

    def test_capacity_one(self):
        """Capacity of 1."""
        values = [10, 20]
        weights = [1, 2]
        capacity = 1

        result = solve_knapsack(values, weights, capacity)
        assert result.objective == 10
        assert result.solution == (0,)


class TestNegativeValuesMaximize:
    """Test handling of negative values in maximize mode."""

    def test_negative_values_maximize(self):
        """Negative values should not be taken when maximizing."""
        values = [-10, 20, -30]
        weights = [1, 1, 1]
        capacity = 2

        result = solve_knapsack(values, weights, capacity, minimize=False)
        # Should only take item 1 (value 20)
        assert result.objective == 20
        assert 1 in result.solution
        assert 0 not in result.solution
        assert 2 not in result.solution

    def test_all_negative_maximize(self):
        """All negative values when maximizing - take nothing."""
        values = [-10, -20, -30]
        weights = [1, 2, 3]
        capacity = 5

        result = solve_knapsack(values, weights, capacity, minimize=False)
        # Taking nothing gives objective 0, better than any negative
        assert result.objective == 0
        assert result.solution == ()

    def test_mixed_positive_negative_maximize(self):
        """Mix of positive and negative values."""
        values = [100, -50, 75, -25]
        weights = [10, 5, 8, 3]
        capacity = 20

        result = solve_knapsack(values, weights, capacity, minimize=False)
        # Should prefer positive value items
        total_weight = sum(weights[i] for i in result.solution)
        assert total_weight <= capacity
        assert result.objective >= 0


class TestMinimizeAdvanced:
    """Advanced minimize mode tests."""

    def test_minimize_prefers_most_negative(self):
        """Minimize should prefer most negative values."""
        values = [-100, -10, -50]
        weights = [5, 5, 5]
        capacity = 10

        result = solve_knapsack(values, weights, capacity, minimize=True)
        # Should take items 0 and 2 (most negative: -100 and -50)
        assert 0 in result.solution
        assert 2 in result.solution
        assert result.objective == -150

    def test_minimize_zero_and_negative(self):
        """Minimize with zero and negative values."""
        values = [0, -10, -20]
        weights = [1, 1, 1]
        capacity = 2

        result = solve_knapsack(values, weights, capacity, minimize=True)
        # Should take items 1 and 2 (most negative)
        assert result.objective == -30


class TestDPTableBehavior:
    """Test DP table edge cases."""

    def test_first_item_only_fits(self):
        """Only the first item fits."""
        values = [50, 100, 150]
        weights = [5, 15, 25]
        capacity = 10

        result = solve_knapsack(values, weights, capacity)
        assert result.objective == 50
        assert result.solution == (0,)

    def test_last_item_only_fits(self):
        """Only the last item fits."""
        values = [50, 100, 150]
        weights = [15, 25, 5]
        capacity = 10

        result = solve_knapsack(values, weights, capacity)
        assert result.objective == 150
        assert result.solution == (2,)

    def test_middle_items_optimal(self):
        """Middle items are optimal."""
        values = [10, 100, 20]
        weights = [20, 5, 30]
        capacity = 10

        result = solve_knapsack(values, weights, capacity)
        assert result.objective == 100
        assert result.solution == (1,)

"""Tests for Bin Packing solver."""

import pytest

from solvor.bin_pack import lower_bound, solve_bin_pack
from solvor.types import Status


class TestBasicPacking:
    def test_simple(self):
        """Simple packing that fits in one bin."""
        sizes = [3, 3, 3]
        capacity = 10
        result = solve_bin_pack(sizes, capacity)
        assert result.objective == 1
        assert all(a == 0 for a in result.solution)  # All in bin 0

    def test_two_bins(self):
        """Items requiring two bins."""
        sizes = [6, 6, 4]  # 6+4=10 fits, 6 needs another bin
        capacity = 10
        result = solve_bin_pack(sizes, capacity)
        assert result.objective == 2

    def test_exact_fit(self):
        """Items exactly fill bins."""
        sizes = [5, 5, 5, 5]
        capacity = 10
        result = solve_bin_pack(sizes, capacity)
        assert result.objective == 2

    def test_single_item(self):
        """Single item."""
        sizes = [5]
        capacity = 10
        result = solve_bin_pack(sizes, capacity)
        assert result.objective == 1
        assert result.solution == (0,)

    def test_empty(self):
        """Empty item list."""
        result = solve_bin_pack([], 10)
        assert result.objective == 0
        assert result.solution == ()
        assert result.status == Status.OPTIMAL


class TestAlgorithms:
    def test_first_fit(self):
        """First-fit algorithm."""
        sizes = [7, 5, 3, 4, 6]
        capacity = 10
        result = solve_bin_pack(sizes, capacity, algorithm="first-fit")
        assert result.objective >= 3  # At least 3 bins needed

    def test_best_fit(self):
        """Best-fit algorithm."""
        sizes = [7, 5, 3, 4, 6]
        capacity = 10
        result = solve_bin_pack(sizes, capacity, algorithm="best-fit")
        assert result.objective >= 3

    def test_first_fit_decreasing(self):
        """First-fit-decreasing algorithm."""
        sizes = [7, 5, 3, 4, 6]
        capacity = 10
        result = solve_bin_pack(sizes, capacity, algorithm="first-fit-decreasing")
        assert result.objective >= 3

    def test_best_fit_decreasing(self):
        """Best-fit-decreasing is default."""
        sizes = [7, 5, 3, 4, 6]
        capacity = 10
        result = solve_bin_pack(sizes, capacity, algorithm="best-fit-decreasing")
        assert result.objective >= 3

    def test_decreasing_often_better(self):
        """Decreasing variants often use fewer bins."""
        sizes = [1, 1, 1, 6, 6, 6]
        capacity = 7

        result_ff = solve_bin_pack(sizes, capacity, algorithm="first-fit")
        result_ffd = solve_bin_pack(sizes, capacity, algorithm="first-fit-decreasing")

        # FFD should do at least as well as FF (never worse)
        assert result_ffd.objective <= result_ff.objective


class TestEdgeCases:
    def test_zero_size_items(self):
        """Zero-size items are handled."""
        sizes = [0, 5, 0, 3]
        capacity = 10
        result = solve_bin_pack(sizes, capacity)
        assert result.objective == 1

    def test_all_same_size(self):
        """All items same size."""
        sizes = [3, 3, 3, 3, 3]
        capacity = 10
        result = solve_bin_pack(sizes, capacity)
        # 5 items of size 3, capacity 10: 3 items per bin, need 2 bins
        assert result.objective == 2

    def test_one_item_per_bin(self):
        """Each item needs its own bin."""
        sizes = [6, 7, 8, 9]
        capacity = 10
        result = solve_bin_pack(sizes, capacity)
        assert result.objective == 4

    def test_tight_packing(self):
        """Items that pack perfectly."""
        sizes = [2, 3, 5, 4, 6]
        capacity = 10
        result = solve_bin_pack(sizes, capacity)
        # Total = 20, could fit in 2 bins of 10
        assert result.objective <= 3


class TestValidation:
    def test_item_too_large(self):
        """Item larger than bin capacity raises error."""
        with pytest.raises(ValueError, match="exceeds bin capacity"):
            solve_bin_pack([5, 15, 3], 10)

    def test_negative_size(self):
        """Negative item size raises error."""
        with pytest.raises(ValueError, match="negative size"):
            solve_bin_pack([5, -2, 3], 10)

    def test_zero_capacity(self):
        """Zero capacity raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            solve_bin_pack([1, 2, 3], 0)

    def test_negative_capacity(self):
        """Negative capacity raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            solve_bin_pack([1, 2, 3], -10)

    def test_unknown_algorithm(self):
        """Unknown algorithm raises error."""
        with pytest.raises(ValueError, match="Unknown algorithm"):
            solve_bin_pack([1, 2, 3], 10, algorithm="unknown")


class TestLowerBound:
    def test_lower_bound_basic(self):
        """Lower bound calculation."""
        sizes = [3, 3, 3, 3]  # Total = 12
        capacity = 10
        lb = lower_bound(sizes, capacity)
        assert lb == 2  # ceil(12/10) = 2

    def test_lower_bound_exact(self):
        """Lower bound with exact fit."""
        sizes = [5, 5, 5, 5]  # Total = 20
        capacity = 10
        lb = lower_bound(sizes, capacity)
        assert lb == 2

    def test_lower_bound_ceiling(self):
        """Lower bound uses ceiling."""
        sizes = [1, 1, 1]  # Total = 3
        capacity = 10
        lb = lower_bound(sizes, capacity)
        assert lb == 1


class TestLargeInstances:
    def test_many_small_items(self):
        """Many small items."""
        sizes = [1] * 100
        capacity = 10
        result = solve_bin_pack(sizes, capacity)
        assert result.objective == 10  # 100 items / 10 per bin

    def test_varied_sizes(self):
        """Varied item sizes."""
        sizes = list(range(1, 21))  # 1 to 20
        capacity = 20
        result = solve_bin_pack(sizes, capacity)
        # Total = 210, capacity 20, lower bound = 11
        assert result.objective >= 11
        assert result.objective <= 15  # Heuristic should be reasonable


class TestAssignments:
    def test_assignment_validity(self):
        """Verify bin assignments are valid."""
        sizes = [4, 5, 3, 7, 2]
        capacity = 10
        result = solve_bin_pack(sizes, capacity)

        # Check each bin doesn't exceed capacity
        num_bins = int(result.objective)
        bin_totals = [0.0] * num_bins
        for i, bin_idx in enumerate(result.solution):
            bin_totals[bin_idx] += sizes[i]

        for total in bin_totals:
            assert total <= capacity

    def test_all_items_assigned(self):
        """All items are assigned to some bin."""
        sizes = [3, 4, 5, 6]
        capacity = 10
        result = solve_bin_pack(sizes, capacity)

        assert len(result.solution) == len(sizes)
        num_bins = int(result.objective)
        for bin_idx in result.solution:
            assert 0 <= bin_idx < num_bins

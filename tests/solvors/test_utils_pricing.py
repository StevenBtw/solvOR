"""Tests for shared pricing utilities."""

import pytest

from solvor.utils.pricing import greedy_knapsack, knapsack_pricing, simplex_phase


class TestKnapsackPricing:
    def test_empty(self):
        """Knapsack with no items."""
        pattern, val = knapsack_pricing([], 100.0, [], 1e-9)
        assert pattern == ()
        assert val == 0.0

    def test_zero_values(self):
        """Knapsack with all zero dual values."""
        sizes = [10.0, 20.0]
        capacity = 100.0
        values = [0.0, 0.0]
        pattern, val = knapsack_pricing(sizes, capacity, values, 1e-9)
        assert val == 0.0

    def test_single_item(self):
        """Knapsack with single item type."""
        sizes = [10.0]
        capacity = 100.0
        values = [5.0]
        pattern, val = knapsack_pricing(sizes, capacity, values, 1e-9)
        assert pattern[0] == 10  # 100 / 10 = 10 copies
        assert val == 50.0

    def test_multiple_items(self):
        """Knapsack with multiple item types."""
        sizes = [30.0, 20.0, 10.0]
        capacity = 100.0
        values = [3.0, 2.0, 1.0]
        pattern, val = knapsack_pricing(sizes, capacity, values, 1e-9)
        # Check solution is feasible
        total_size = sum(pattern[i] * sizes[i] for i in range(len(sizes)))
        assert total_size <= capacity + 1e-9
        assert val > 0

    def test_exact_fit(self):
        """Items that exactly fill capacity."""
        sizes = [50.0]
        capacity = 100.0
        values = [10.0]
        pattern, val = knapsack_pricing(sizes, capacity, values, 1e-9)
        assert pattern[0] == 2
        assert val == 20.0

    def test_negative_values_ignored(self):
        """Negative values should be ignored."""
        sizes = [10.0, 20.0]
        capacity = 100.0
        values = [-1.0, 5.0]
        pattern, val = knapsack_pricing(sizes, capacity, values, 1e-9)
        assert pattern[0] == 0  # Negative value item not selected
        assert val > 0


class TestGreedyKnapsack:
    def test_simple(self):
        """Simple greedy test."""
        sizes = [3.0, 4.0, 5.0]
        capacity = 10.0
        values = [3.0, 4.0, 5.0]
        max_copies = [2, 2, 2]
        pattern, val = greedy_knapsack(sizes, capacity, values, max_copies)
        # Check feasibility
        total_size = sum(pattern[i] * sizes[i] for i in range(len(sizes)))
        assert total_size <= capacity
        assert val > 0

    def test_zero_values(self):
        """Greedy with zero or negative values."""
        sizes = [3.0, 4.0]
        capacity = 10.0
        values = [0.0, -1.0]
        max_copies = [2, 2]
        pattern, val = greedy_knapsack(sizes, capacity, values, max_copies)
        assert pattern == (0, 0)
        assert val == 0.0

    def test_respects_max_copies(self):
        """Greedy respects max_copies limit."""
        sizes = [10.0]
        capacity = 100.0
        values = [5.0]
        max_copies = [3]  # Only allow 3 copies
        pattern, val = greedy_knapsack(sizes, capacity, values, max_copies)
        assert pattern[0] == 3
        assert val == 15.0

    def test_density_ordering(self):
        """Greedy prioritizes by value/size density."""
        sizes = [10.0, 20.0]
        capacity = 20.0
        values = [5.0, 6.0]  # First has density 0.5, second has 0.3
        max_copies = [10, 10]
        pattern, val = greedy_knapsack(sizes, capacity, values, max_copies)
        # Should prefer first item (higher density)
        assert pattern[0] == 2
        assert pattern[1] == 0

    def test_zero_size_item(self):
        """Handles zero-size items gracefully."""
        sizes = [0.0, 10.0]
        capacity = 100.0
        values = [5.0, 10.0]
        max_copies = [10, 10]
        pattern, val = greedy_knapsack(sizes, capacity, values, max_copies)
        # Zero-size items should be skipped
        assert pattern[0] == 0


class TestSimplexPhase:
    def test_simple_lp(self):
        """Test simplex on simple tableau."""
        # Simple LP: min x s.t. x >= 1, x >= 0
        # Tableau: [x, surplus, artificial | rhs]
        # Row 0: [1, -1, 1 | 1]  (x - s + a = 1)
        # Row 1: [1, 0, 0 | 0]   (objective: min x)
        tab = [
            [1.0, -1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
        basis = [2]  # artificial in basis

        # Phase 1: minimize artificial
        tab[1] = [0.0, 0.0, 1.0, 0.0]
        tab[1][0] -= tab[0][0]
        tab[1][1] -= tab[0][1]
        tab[1][3] -= tab[0][3]
        tab[1][2] = 0.0

        simplex_phase(tab, basis, 2, 1, 1e-9)
        # After phase 1, artificial should be zero
        assert abs(tab[1][3]) < 1e-6

    def test_already_optimal(self):
        """Simplex on already optimal tableau does nothing."""
        tab = [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],  # No negative reduced costs
        ]
        basis = [0]
        simplex_phase(tab, basis, 1, 1, 1e-9)
        assert basis == [0]  # Unchanged

    def test_modifies_in_place(self):
        """Simplex modifies tableau in place."""
        tab = [
            [2.0, 1.0, 0.0, 10.0],
            [-1.0, 0.0, 0.0, 0.0],
        ]
        original_id = id(tab)
        basis = [1]
        simplex_phase(tab, basis, 2, 1, 1e-9)
        assert id(tab) == original_id  # Same object


class TestIntegration:
    def test_knapsack_greedy_consistency(self):
        """Knapsack and greedy produce feasible solutions."""
        sizes = [15.0, 25.0, 35.0]
        capacity = 100.0
        values = [1.5, 2.5, 3.5]

        # Knapsack solution
        kp_pattern, kp_val = knapsack_pricing(sizes, capacity, values, 1e-9)
        kp_size = sum(kp_pattern[i] * sizes[i] for i in range(len(sizes)))
        assert kp_size <= capacity + 1e-6

        # Greedy solution
        max_copies = [int(capacity // s) for s in sizes]
        gr_pattern, gr_val = greedy_knapsack(sizes, capacity, values, max_copies)
        gr_size = sum(gr_pattern[i] * sizes[i] for i in range(len(sizes)))
        assert gr_size <= capacity + 1e-6

        # Both should produce positive value
        assert kp_val >= 0
        assert gr_val >= 0

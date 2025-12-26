"""Tests for validation utilities."""

import pytest

from solvor.utils import (
    check_bounds,
    check_edge_nodes,
    check_in_range,
    check_integers_valid,
    check_matrix_dims,
    check_non_negative,
    check_positive,
    check_sequence_lengths,
)


class TestMatrixDims:
    def test_valid_dims(self):
        """Valid LP dimensions pass."""
        c = [1, 2, 3]
        A = [[1, 0, 0], [0, 1, 0]]
        b = [5, 10]
        check_matrix_dims(c, A, b)  # Should not raise

    def test_empty_A(self):
        """Empty constraint matrix raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            check_matrix_dims([1, 2], [], [])

    def test_row_count_mismatch(self):
        """Mismatch between A rows and b length."""
        c = [1, 2]
        A = [[1, 0], [0, 1], [1, 1]]  # 3 rows
        b = [5, 10]  # 2 elements
        with pytest.raises(ValueError, match="constraints"):
            check_matrix_dims(c, A, b)

    def test_column_count_mismatch(self):
        """Mismatch between A columns and c length."""
        c = [1, 2, 3]  # 3 variables
        A = [[1, 0], [0, 1]]  # 2 columns
        b = [5, 10]
        with pytest.raises(ValueError, match="variables"):
            check_matrix_dims(c, A, b)


class TestSequenceLengths:
    def test_equal_lengths(self):
        """Equal length sequences pass."""
        n = check_sequence_lengths(
            ([1, 2, 3], "values"),
            ([4, 5, 6], "weights"),
        )
        assert n == 3

    def test_mismatched_lengths(self):
        """Mismatched lengths raise error."""
        with pytest.raises(ValueError, match="Length mismatch"):
            check_sequence_lengths(
                ([1, 2, 3], "values"),
                ([4, 5], "weights"),
            )

    def test_expected_length(self):
        """Can specify expected length."""
        check_sequence_lengths(([1, 2], "a"), expected=2)

    def test_expected_length_mismatch(self):
        """Wrong expected length raises error."""
        with pytest.raises(ValueError, match="expected 3"):
            check_sequence_lengths(([1, 2], "a"), expected=3)


class TestBounds:
    def test_valid_bounds(self):
        """Valid bounds pass."""
        n = check_bounds([(-5, 5), (0, 10), (-1, 1)])
        assert n == 3

    def test_invalid_bounds(self):
        """Lower > upper raises error."""
        with pytest.raises(ValueError, match="lower bound"):
            check_bounds([(-5, 5), (10, 0)])  # Second bound is invalid


class TestPositive:
    def test_positive(self):
        """Positive value passes."""
        check_positive(5, name="capacity")

    def test_zero(self):
        """Zero raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            check_positive(0, name="capacity")

    def test_negative(self):
        """Negative raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            check_positive(-5, name="capacity")


class TestNonNegative:
    def test_positive(self):
        """Positive passes."""
        check_non_negative(5, name="weight")

    def test_zero(self):
        """Zero passes."""
        check_non_negative(0, name="weight")

    def test_negative(self):
        """Negative raises error."""
        with pytest.raises(ValueError, match="cannot be negative"):
            check_non_negative(-1, name="weight")


class TestInRange:
    def test_in_range_inclusive(self):
        """Value in range passes (inclusive)."""
        check_in_range(5, 0, 10, name="value")
        check_in_range(0, 0, 10, name="value")  # At boundary
        check_in_range(10, 0, 10, name="value")  # At boundary

    def test_out_of_range(self):
        """Value out of range raises error."""
        with pytest.raises(ValueError, match="must be in"):
            check_in_range(15, 0, 10, name="value")

    def test_exclusive(self):
        """Exclusive range check."""
        check_in_range(5, 0, 10, name="value", inclusive=False)
        with pytest.raises(ValueError):
            check_in_range(0, 0, 10, name="value", inclusive=False)


class TestIntegersValid:
    def test_valid_indices(self):
        """Valid indices pass."""
        check_integers_valid([0, 2, 4], 5)

    def test_out_of_range(self):
        """Index out of range raises error."""
        with pytest.raises(ValueError, match="Invalid index"):
            check_integers_valid([0, 5], 5)  # 5 is out of range for n_vars=5

    def test_negative_index(self):
        """Negative index raises error."""
        with pytest.raises(ValueError, match="Invalid index"):
            check_integers_valid([-1, 0], 5)


class TestEdgeNodes:
    def test_valid_edges(self):
        """Valid edges pass."""
        edges = [(0, 1, 1.0), (1, 2, 2.0), (0, 2, 3.0)]
        check_edge_nodes(edges, 3)  # Should not raise

    def test_invalid_u(self):
        """Invalid u node raises error."""
        edges = [(0, 1, 1.0), (5, 2, 2.0)]  # 5 out of range
        with pytest.raises(ValueError, match="u=5"):
            check_edge_nodes(edges, 3)

    def test_invalid_v(self):
        """Invalid v node raises error."""
        edges = [(0, 1, 1.0), (1, 5, 2.0)]  # v=5 out of range
        with pytest.raises(ValueError, match="v=5"):
            check_edge_nodes(edges, 3)

    def test_negative_node(self):
        """Negative node raises error."""
        edges = [(0, 1, 1.0), (-1, 2, 2.0)]
        with pytest.raises(ValueError, match="u=-1"):
            check_edge_nodes(edges, 3)

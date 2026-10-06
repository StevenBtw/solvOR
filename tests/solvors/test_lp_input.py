"""Tests for LP/MILP input normalization."""

import math

import pytest

from solvor.utils.lp_input import normalize_lp


class TestRows:
    def test_dense_rows_become_sparse(self):
        prob = normalize_lp([1, 2, 3], [[1, 0, 2], [0, 0, 0]], [4, 5])
        assert prob.rows == [{0: 1.0, 2: 2.0}, {}]
        assert prob.b == [4.0, 5.0]

    def test_sparse_and_dense_rows_can_mix(self):
        prob = normalize_lp([1, 2, 3], [{2: 5, 0: 0}, [1, 1, 0]], [1, 2])
        assert prob.rows == [{2: 5.0}, {0: 1.0, 1: 1.0}]

    def test_columns_view(self):
        prob = normalize_lp([1, 1], [{0: 2}, {0: 1, 1: 3}], [1, 1])
        assert prob.columns() == [[(0, 2.0), (1, 1.0)], [(1, 3.0)]]

    def test_empty_A_is_allowed(self):
        prob = normalize_lp([1, 1], [], [])
        assert prob.rows == []
        assert prob.n == 2

    def test_row_count_must_match_b(self):
        with pytest.raises(ValueError, match="b has 1 constraints but A has 2 rows"):
            normalize_lp([1], [[1], [1]], [1])

    def test_dense_row_length_must_match_c(self):
        with pytest.raises(ValueError, match="A row 0 has 1 columns"):
            normalize_lp([1, 1], [[1]], [1])

    def test_sparse_column_out_of_range(self):
        with pytest.raises(ValueError, match="column 2 outside the valid range 0 to 1"):
            normalize_lp([1, 1], [{2: 1.0}], [1])

    def test_sparse_column_must_be_int(self):
        with pytest.raises(TypeError, match="non-integer column key"):
            normalize_lp([1, 1], [{"0": 1.0}], [1])

    def test_nan_rejected(self):
        with pytest.raises(ValueError, match="NaN"):
            normalize_lp([1], [[math.nan]], [1])
        with pytest.raises(ValueError, match="NaN"):
            normalize_lp([1], [[1]], [math.nan])
        with pytest.raises(ValueError, match="c contains NaN"):
            normalize_lp([math.nan], [[1]], [1])

    def test_infinite_costs_and_coefficients_rejected(self):
        """Solving an infinite cost or coefficient gave OPTIMAL with a NaN objective or a violated row."""
        with pytest.raises(ValueError, match="c contains inf"):
            normalize_lp([math.inf, 1], [[1, 1]], [1])
        with pytest.raises(ValueError, match="c contains -inf"):
            normalize_lp([1, -math.inf], [[1, 1]], [1])
        with pytest.raises(ValueError, match="A row 0 contains inf"):
            normalize_lp([1, 1], [{0: math.inf, 1: 1}], [1])
        with pytest.raises(ValueError, match="A row 1 contains -inf"):
            normalize_lp([1, 1], [[1, 1], [-math.inf, 0]], [1, 1])

    def test_large_coefficients_warn(self):
        with pytest.warns(UserWarning, match="Large coefficients"):
            normalize_lp([1], [{0: 1e12}], [1])


class TestSensesAndBounds:
    def test_defaults(self):
        prob = normalize_lp([1, 1], [[1, 1]], [1])
        assert prob.senses == ["<="]
        assert prob.lb == [0.0, 0.0]
        assert prob.ub == [math.inf, math.inf]

    def test_explicit(self):
        prob = normalize_lp([1, 1], [[1, 1], [1, 0]], [1, 2], senses=[">=", "="], lb=[-1, -math.inf], ub=[3, 4])
        assert prob.senses == [">=", "="]
        assert prob.lb == [-1.0, -math.inf]
        assert prob.ub == [3.0, 4.0]

    def test_bad_sense(self):
        with pytest.raises(ValueError, match=r"senses\[0\] must be one of"):
            normalize_lp([1], [[1]], [1], senses=["<"])

    def test_senses_length(self):
        with pytest.raises(ValueError, match="expected 1 elements in senses"):
            normalize_lp([1], [[1]], [1], senses=["<=", "<="])

    def test_bounds_length(self):
        with pytest.raises(ValueError, match="expected 2 elements in lb"):
            normalize_lp([1, 1], [[1, 1]], [1], lb=[0])
        with pytest.raises(ValueError, match="expected 2 elements in ub"):
            normalize_lp([1, 1], [[1, 1]], [1], ub=[1, 1, 1])

    def test_inverted_bounds_are_not_an_input_error(self):
        """lb > ub is a valid (infeasible) model; the solvers report INFEASIBLE."""
        prob = normalize_lp([1], [[1]], [1], lb=[2], ub=[1])
        assert prob.lb == [2.0]
        assert prob.ub == [1.0]


class TestImpossibleBounds:
    """A lower bound of +inf or an upper bound of -inf leaves no value for the variable."""

    def test_lower_bound_plus_infinity_is_rejected(self):
        with pytest.raises(ValueError, match="lb contains inf"):
            normalize_lp([1, 1], [[1, 1]], [4], lb=[0, math.inf])

    def test_upper_bound_minus_infinity_is_rejected(self):
        with pytest.raises(ValueError, match="ub contains -inf"):
            normalize_lp([1, 1], [[1, 1]], [4], ub=[-math.inf, 5])

    def test_solvers_reject_them_too(self):
        from solvor import MilpModel, solve_lp, solve_milp

        with pytest.raises(ValueError, match="lb contains inf"):
            solve_lp([1], [[1]], [4], lb=[math.inf])
        with pytest.raises(ValueError, match="ub contains -inf"):
            solve_milp([1], [[1]], [4], [0], ub=[-math.inf])
        with pytest.raises(ValueError, match="lb contains inf"):
            MilpModel(1, lb=[math.inf])

    @pytest.mark.parametrize("name", ["lb", "ub"])
    def test_nan_bounds_are_rejected(self, name):
        with pytest.raises(ValueError, match=f"{name} contains NaN"):
            normalize_lp([1, 1], [[1, 1]], [4], **{name: [0, math.nan]})

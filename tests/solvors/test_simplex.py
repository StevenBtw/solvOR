"""Tests for the simplex (linear programming) solver."""

import math

import pytest

from solvor.simplex import solve_lp
from solvor.types import Status


class TestBasicLP:
    def test_minimize_basic(self):
        # minimize x + 2y subject to x + y >= 2, x >= 0, y >= 0
        # Optimal: x=2, y=0, obj=2
        result = solve_lp(c=[1, 2], A=[[-1, -1]], b=[-2])
        assert result.status == Status.OPTIMAL
        assert abs(result.objective - 2.0) < 1e-6

    def test_maximize_basic(self):
        # maximize x + y subject to x + y <= 4, x <= 3
        result = solve_lp(c=[1, 1], A=[[1, 1], [1, 0]], b=[4, 3], minimize=False)
        assert result.status == Status.OPTIMAL
        assert abs(result.objective - 4.0) < 1e-6

    def test_single_variable(self):
        # minimize x subject to x >= 5
        result = solve_lp(c=[1], A=[[-1]], b=[-5])
        assert result.status == Status.OPTIMAL
        assert abs(result.objective - 5.0) < 1e-6
        assert abs(result.solution[0] - 5.0) < 1e-6


class TestInfeasibleLP:
    def test_infeasible_constraints(self):
        # x >= 1, x <= 0 - infeasible
        result = solve_lp(c=[1], A=[[-1], [1]], b=[-1, 0])
        assert result.status == Status.INFEASIBLE

    def test_infeasible_contradictory(self):
        # x + y >= 10 and x + y <= 5 - infeasible
        result = solve_lp(c=[1, 1], A=[[-1, -1], [1, 1]], b=[-10, 5])
        assert result.status == Status.INFEASIBLE


class TestUnboundedLP:
    def test_unbounded_basic(self):
        # minimize -x subject to y <= 1 (x unbounded below when minimizing -x)
        result = solve_lp(c=[-1, 0], A=[[0, 1]], b=[1])
        assert result.status == Status.UNBOUNDED

    def test_unbounded_maximize(self):
        # maximize x, no upper bound on x
        result = solve_lp(c=[1, 0], A=[[0, 1]], b=[1], minimize=False)
        assert result.status == Status.UNBOUNDED


class TestEdgeCases:
    def test_zero_coefficients(self):
        # minimize x subject to y <= 10, x >= 0 (y doesn't matter)
        result = solve_lp(c=[1, 0], A=[[0, 1]], b=[10])
        assert result.status == Status.OPTIMAL
        assert abs(result.objective) < 1e-6  # x=0 is optimal

    def test_multiple_constraints(self):
        # Multiple upper bound constraints
        # minimize -x - y (maximize x + y), x <= 3, y <= 4, x + y <= 5
        result = solve_lp(c=[-1, -1], A=[[1, 0], [0, 1], [1, 1]], b=[3, 4, 5])
        assert result.status == Status.OPTIMAL
        # Optimal at x + y = 5
        assert result.solution[0] + result.solution[1] <= 5 + 1e-6

    def test_tight_constraints(self):
        # All constraints should be tight at optimum
        # minimize x + y, x + y >= 5, x >= 2, y >= 2
        result = solve_lp(c=[1, 1], A=[[-1, -1], [-1, 0], [0, -1]], b=[-5, -2, -2])
        assert result.status == Status.OPTIMAL
        # Optimal is x=2.5, y=2.5 or x=3, y=2 etc, obj=5
        assert abs(result.objective - 5.0) < 1e-6


class TestStress:
    def test_many_variables(self):
        # minimize sum(x_i) subject to sum(x_i) >= 100
        n = 50
        c = [1.0] * n
        A = [[-1.0] * n]
        b = [-100.0]
        result = solve_lp(c=c, A=A, b=b)
        assert result.status == Status.OPTIMAL
        assert abs(result.objective - 100.0) < 1e-4
        assert abs(sum(result.solution) - 100.0) < 1e-4

    def test_many_constraints(self):
        # Multiple upper bound constraints on same variable
        # minimize -x (maximize x), x <= 1, x <= 2, x <= 5, x <= 3
        result = solve_lp(c=[-1], A=[[1], [1], [1], [1]], b=[1, 2, 5, 3])
        assert result.status == Status.OPTIMAL
        # Tightest is x <= 1
        assert result.solution[0] <= 1 + 1e-6

    def test_degenerate_case(self):
        # Multiple optimal solutions (degenerate)
        # minimize x, x + y = 10, x >= 0, y >= 0
        result = solve_lp(c=[1, 0], A=[[-1, -1], [1, 1]], b=[-10, 10])
        assert result.status == Status.OPTIMAL
        assert abs(result.solution[0] + result.solution[1] - 10.0) < 1e-6


class TestNumericalStability:
    def test_small_coefficients(self):
        # Very small coefficients
        result = solve_lp(c=[1e-8, 1e-8], A=[[-1, -1]], b=[-1])
        assert result.status == Status.OPTIMAL

    def test_large_coefficients(self):
        # Large coefficients
        result = solve_lp(c=[1e6, 1e6], A=[[-1, -1]], b=[-100])
        assert result.status == Status.OPTIMAL
        assert abs(result.objective - 100e6) < 1e2

    def test_mixed_scale(self):
        # Mix of different scale coefficients
        # minimize x + 100y, x + y >= 2
        result = solve_lp(c=[1, 100], A=[[-1, -1]], b=[-2])
        assert result.status == Status.OPTIMAL
        # Should minimize y (more expensive), so x=2, y=0
        assert abs(result.objective - 2.0) < 1e-2


class TestBoundsAndSenses:
    def test_upper_bounds_without_rows(self):
        result = solve_lp([-1, -2], [], [], ub=[3, 4])
        assert result.status == Status.OPTIMAL
        assert result.solution == (3.0, 4.0)
        assert result.objective == -11.0

    def test_negative_lower_bound(self):
        # minimize x subject to x >= -5 (as a bound) and x + y <= 10
        result = solve_lp([1, 0], [[1, 1]], [10], lb=[-5, 0])
        assert abs(result.solution[0] + 5) < 1e-9
        assert abs(result.objective + 5) < 1e-9

    def test_free_variable(self):
        # minimize -x + y with x - y = 2, x free, y in [0, 3]: x = y + 2, objective = -2
        result = solve_lp([-1, 1], [[1, -1]], [2], senses=["="], lb=[-math.inf, 0], ub=[math.inf, 3])
        assert result.status == Status.OPTIMAL
        assert abs(result.objective + 2) < 1e-9

    def test_upper_bound_only(self):
        # maximize x with x <= 4 as a bound and no lower bound
        result = solve_lp([1], [], [], minimize=False, lb=[-math.inf], ub=[4])
        assert result.solution == (4.0,)

    def test_fixed_variable(self):
        result = solve_lp([1, 1], [[1, 1]], [10], minimize=False, lb=[2, 0], ub=[2, math.inf])
        assert abs(result.solution[0] - 2) < 1e-9
        assert abs(result.objective - 10) < 1e-9

    def test_greater_equal_and_equal_rows(self):
        # minimize x + y with x + y >= 3, x - y = 1
        result = solve_lp([1, 1], [[1, 1], [1, -1]], [3, 1], senses=[">=", "="])
        assert result.status == Status.OPTIMAL
        assert abs(result.solution[0] - 2) < 1e-9
        assert abs(result.solution[1] - 1) < 1e-9

    def test_inverted_bounds_are_infeasible(self):
        result = solve_lp([1], [[1]], [5], lb=[3], ub=[2])
        assert result.status == Status.INFEASIBLE

    def test_unbounded_objective_is_infinite(self):
        assert solve_lp([-1, 0], [[0, 1]], [1]).objective == -math.inf
        assert solve_lp([1, 0], [[0, 1]], [1], minimize=False).objective == math.inf

    def test_sparse_rows_equal_dense_rows(self):
        dense = solve_lp([2, 3, 1], [[1, 1, 1], [0, 2, 1]], [10, 8], minimize=False)
        sparse = solve_lp([2, 3, 1], [{0: 1, 1: 1, 2: 1}, {1: 2, 2: 1}], [10, 8], minimize=False)
        assert dense.objective == sparse.objective


class TestBoundsMatchRows:
    """Native bounds and senses give the same optimum as the same model written with rows only."""

    def test_random_models(self):
        import random

        rng = random.Random(777)
        for _ in range(400):
            n, m = rng.randint(1, 6), rng.randint(1, 5)
            c = [float(rng.randint(-5, 5)) for _ in range(n)]
            rows = [{j: float(rng.randint(-3, 5)) for j in range(n) if rng.random() < 0.6} for _ in range(m)]
            b = [float(rng.randint(0, 15)) for _ in range(m)]
            senses = [rng.choice(("<=", "<=", ">=", "=")) for _ in range(m)]
            lb = [float(rng.randint(-4, 1)) for _ in range(n)]
            ub = [lo + float(rng.randint(0, 6)) for lo in lb]
            minimize = rng.random() < 0.5
            native = solve_lp(c, rows, b, minimize=minimize, senses=senses, lb=lb, ub=ub)

            # Same model with x = lb + y, y >= 0, and senses and upper bounds as "<=" rows
            rows2, b2 = [], []
            for row, bi, s in zip(rows, b, senses):
                shift = sum(a * lb[j] for j, a in row.items())
                if s in ("<=", "="):
                    rows2.append(dict(row))
                    b2.append(bi - shift)
                if s in (">=", "="):
                    rows2.append({j: -a for j, a in row.items()})
                    b2.append(shift - bi)
            for j in range(n):
                rows2.append({j: 1.0})
                b2.append(ub[j] - lb[j])
            as_rows = solve_lp(c, rows2, b2, minimize=minimize)

            assert native.status == as_rows.status
            if native.status == Status.OPTIMAL:
                offset = sum(cj * lj for cj, lj in zip(c, lb))
                assert abs(native.objective - (as_rows.objective + offset)) < 1e-6


class TestPhaseOneTolerance:
    """Infeasibility must not be hidden by a large value in an unrelated row."""

    @pytest.mark.parametrize("big", [1e7, 1e9, 1e12])
    def test_infeasible_with_a_large_unrelated_row(self, big):
        # x + y >= 3 with x, y <= 1 is infeasible; z + w <= big has nothing to do with it
        result = solve_lp([1, 1, 0, 0], [[1, 1, 0, 0], [0, 0, 1, 1]], [3, big], senses=[">=", "<="], ub=[1, 1, 10, 10])
        assert result.status == Status.INFEASIBLE

    def test_infeasible_with_an_infinite_redundant_row(self):
        result = solve_lp([1, 1], [[1, 1], [1, 0]], [3, math.inf], senses=[">=", "<="], ub=[1, 1])
        assert result.status == Status.INFEASIBLE

    def test_row_that_can_never_hold(self):
        assert solve_lp([1], [[1]], [-math.inf]).status == Status.INFEASIBLE

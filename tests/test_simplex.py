"""Tests for the simplex (linear programming) solver."""

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
        result = solve_lp(
            c=[-1, -1],
            A=[[1, 0], [0, 1], [1, 1]],
            b=[3, 4, 5]
        )
        assert result.status == Status.OPTIMAL
        # Optimal at x + y = 5
        assert result.solution[0] + result.solution[1] <= 5 + 1e-6

    def test_tight_constraints(self):
        # All constraints should be tight at optimum
        # minimize x + y, x + y >= 5, x >= 2, y >= 2
        result = solve_lp(
            c=[1, 1],
            A=[[-1, -1], [-1, 0], [0, -1]],
            b=[-5, -2, -2]
        )
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

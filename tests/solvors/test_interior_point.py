"""Tests for the interior point (linear programming) solver."""

from solvor.interior_point import solve_lp_interior
from solvor.types import Status


class TestBasicLP:
    def test_minimize_basic(self):
        # minimize x + 2y subject to x + y >= 2, x >= 0, y >= 0
        # Optimal: x=2, y=0, obj=2
        result = solve_lp_interior(c=[1, 2], A=[[-1, -1]], b=[-2])
        assert result.ok
        assert abs(result.objective - 2.0) < 0.1

    def test_maximize_basic(self):
        # maximize x + y subject to x + y <= 4, x <= 3
        result = solve_lp_interior(c=[1, 1], A=[[1, 1], [1, 0]], b=[4, 3], minimize=False)
        assert result.ok
        assert abs(result.objective - 4.0) < 0.1

    def test_single_variable(self):
        # minimize x subject to x >= 5
        result = solve_lp_interior(c=[1], A=[[-1]], b=[-5])
        assert result.ok
        assert abs(result.objective - 5.0) < 0.1

    def test_two_variable_corner(self):
        # minimize -x - y (maximize x + y), x <= 3, y <= 2
        result = solve_lp_interior(c=[-1, -1], A=[[1, 0], [0, 1]], b=[3, 2])
        assert result.ok
        # Optimal: x=3, y=2, obj=-5
        assert abs(result.objective - (-5.0)) < 0.1


class TestFeasibility:
    def test_bounded_feasible(self):
        # A well-posed problem should find optimal
        result = solve_lp_interior(c=[1, 1], A=[[1, 0], [0, 1], [1, 1]], b=[10, 10, 15])
        assert result.ok

    def test_equality_constraint(self):
        # x + y = 10 encoded as x + y <= 10 and x + y >= 10
        result = solve_lp_interior(c=[1, -1], A=[[1, 1], [-1, -1]], b=[10, -10])
        assert result.ok
        # x + y should be 10
        assert abs(result.solution[0] + result.solution[1] - 10.0) < 0.5


class TestSimpleProblems:
    def test_diet_problem_style(self):
        # minimize cost, meet minimum requirements
        # 2x + 3y >= 12, x + 2y >= 8, minimize x + y
        result = solve_lp_interior(c=[1, 1], A=[[-2, -3], [-1, -2]], b=[-12, -8])
        assert result.ok
        # Check constraints are approximately satisfied
        x, y = result.solution
        assert 2 * x + 3 * y >= 11.5
        assert x + 2 * y >= 7.5

    def test_production_planning(self):
        # maximize profit: 3x + 2y, subject to x + y <= 4, 2x + y <= 6
        result = solve_lp_interior(c=[3, 2], A=[[1, 1], [2, 1]], b=[4, 6], minimize=False)
        assert result.ok
        assert result.objective > 9  # Optimal is around 10


class TestNumericalBehavior:
    def test_symmetric_problem(self):
        # Symmetric in x and y
        result = solve_lp_interior(c=[1, 1], A=[[-1, 0], [0, -1], [1, 1]], b=[-1, -1, 10])
        assert result.ok
        # Minimum is x=1, y=1, obj=2
        assert abs(result.objective - 2.0) < 0.2

    def test_returns_solution_tuple(self):
        result = solve_lp_interior(c=[1, 2, 3], A=[[1, 1, 1]], b=[10])
        assert result.ok
        assert isinstance(result.solution, tuple)
        assert len(result.solution) == 3


class TestEdgeCases:
    def test_empty_problem(self):
        # No variables - edge case
        # The solver requires at least one constraint
        import pytest

        with pytest.raises(ValueError):
            solve_lp_interior(c=[], A=[], b=[])

    def test_no_constraints(self):
        # Only non-negativity constraints (implicit)
        # The solver requires at least one constraint
        import pytest

        with pytest.raises(ValueError):
            solve_lp_interior(c=[1, 1], A=[], b=[])


class TestCompareWithSimplex:
    """Interior point should give similar results to simplex on well-posed problems."""

    def test_matches_simplex_basic(self):
        from solvor.simplex import solve_lp

        c = [2, 3]
        A = [[-1, -1], [1, 0], [0, 1]]
        b = [-4, 5, 5]

        simplex_result = solve_lp(c=c, A=A, b=b)
        ip_result = solve_lp_interior(c=c, A=A, b=b)

        if simplex_result.ok and ip_result.ok:
            # Should be within 10% of each other
            assert abs(simplex_result.objective - ip_result.objective) < 0.5

    def test_matches_simplex_maximize(self):
        from solvor.simplex import solve_lp

        c = [1, 1]
        A = [[1, 2], [2, 1]]
        b = [8, 8]

        simplex_result = solve_lp(c=c, A=A, b=b, minimize=False)
        ip_result = solve_lp_interior(c=c, A=A, b=b, minimize=False)

        if simplex_result.ok and ip_result.ok:
            assert abs(simplex_result.objective - ip_result.objective) < 0.5

"""Tests for the MILP (mixed-integer linear programming) solver."""

from solvor.milp import solve_milp
from solvor.types import Status


class TestBasicMILP:
    def test_single_integer(self):
        # minimize x + y, x integer, x + y >= 2.5
        result = solve_milp(c=[1, 1], A=[[-1, -1]], b=[-2.5], integers=[0])
        assert result.status in (Status.OPTIMAL, Status.FEASIBLE)
        assert result.solution[0] == round(result.solution[0])  # x is integer

    def test_pure_integer(self):
        # minimize x + y, both integer, x + y >= 3
        result = solve_milp(c=[1, 1], A=[[-1, -1]], b=[-3], integers=[0, 1])
        assert result.status in (Status.OPTIMAL, Status.FEASIBLE)
        assert abs(result.objective - 3.0) < 1e-6

    def test_maximize(self):
        # maximize x + y, x integer, x + y <= 5.5, x <= 3
        result = solve_milp(c=[1, 1], A=[[1, 1], [1, 0]], b=[5.5, 3], integers=[0], minimize=False)
        assert result.status in (Status.OPTIMAL, Status.FEASIBLE)
        assert result.solution[0] == round(result.solution[0])


class TestIntegerFeasibility:
    def test_integer_gap(self):
        # x integer, 1.5 <= x <= 1.9 -> infeasible (no integer in range)
        result = solve_milp(c=[1], A=[[-1], [1]], b=[-1.5, 1.9], integers=[0])
        assert result.status == Status.INFEASIBLE

    def test_integer_bounds(self):
        # x integer, x <= 2.9 -> x <= 2 (maximize -x is same as minimize x with x<=2)
        # minimize x where x is integer, x <= 2.9
        result = solve_milp(c=[1], A=[[1]], b=[2.9], integers=[0])
        assert result.status in (Status.OPTIMAL, Status.FEASIBLE)
        # Solution should be integer
        assert abs(result.solution[0] - round(result.solution[0])) < 1e-6


class TestKnapsackLike:
    def test_binary_selection(self):
        # Simple 0-1 knapsack: maximize value, weight <= capacity
        # Items: value=[3,4,5], weight=[2,3,4], capacity=5
        # Best: items 0 and 1 (value=7, weight=5)
        result = solve_milp(
            c=[3, 4, 5],  # values (maximize)
            A=[[2, 3, 4], [1, 0, 0], [0, 1, 0], [0, 0, 1]],  # weight + upper bounds
            b=[5, 1, 1, 1],
            integers=[0, 1, 2],
            minimize=False,
        )
        assert result.status in (Status.OPTIMAL, Status.FEASIBLE)
        # Should select items to maximize value within weight
        assert result.objective >= 7 - 1e-6


class TestEdgeCases:
    def test_already_integer_relaxation(self):
        # LP relaxation already gives integer solution
        result = solve_milp(c=[1, 1], A=[[-1, -1]], b=[-4], integers=[0, 1])
        assert result.status in (Status.OPTIMAL, Status.FEASIBLE)
        # Both should be integers
        assert result.solution[0] == round(result.solution[0])
        assert result.solution[1] == round(result.solution[1])

    def test_no_integer_constraints(self):
        # All continuous (empty integers list) should work like LP
        result = solve_milp(c=[1, 1], A=[[-1, -1]], b=[-3], integers=[])
        assert result.status in (Status.OPTIMAL, Status.FEASIBLE)
        assert abs(result.objective - 3.0) < 1e-6

    def test_single_variable_integer(self):
        # Single integer variable: maximize x (minimize -x), x <= 5
        result = solve_milp(c=[-1], A=[[1]], b=[5], integers=[0])
        assert result.status in (Status.OPTIMAL, Status.FEASIBLE)
        # Solution must be integer
        assert abs(result.solution[0] - round(result.solution[0])) < 1e-6


class TestStress:
    def test_multiple_integers(self):
        # 5 variables, 3 integer
        n = 5
        c = [1.0] * n
        A = [[-1.0] * n]
        b = [-10.5]
        result = solve_milp(c=c, A=A, b=b, integers=[0, 2, 4])
        assert result.status in (Status.OPTIMAL, Status.FEASIBLE)
        # Integer variables should be integers
        for i in [0, 2, 4]:
            assert abs(result.solution[i] - round(result.solution[i])) < 1e-6

    def test_tight_integer_problem(self):
        # Simple integer problem: maximize x + y (minimize -x - y), x + y <= 10
        result = solve_milp(c=[-1, -1], A=[[1, 1]], b=[10], integers=[0, 1])
        assert result.status in (Status.OPTIMAL, Status.FEASIBLE)
        x, y = result.solution
        # All should be integers
        assert abs(x - round(x)) < 1e-6
        assert abs(y - round(y)) < 1e-6
        # Constraint satisfied
        assert x + y <= 10 + 1e-6

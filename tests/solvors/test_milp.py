"""Tests for the MILP (mixed-integer linear programming) solver."""

from solvor.milp import solve_milp
from solvor.types import Status


class TestBasicMILP:
    def test_single_integer(self):
        # minimize x + y, x integer, x + y >= 2.5
        result = solve_milp(c=[1, 1], A=[[-1, -1]], b=[-2.5], integers=[0])
        assert result.status == Status.OPTIMAL  # Small problem, should prove optimality
        assert result.solution[0] == round(result.solution[0])  # x is integer

    def test_pure_integer(self):
        # minimize x + y, both integer, x + y >= 3
        result = solve_milp(c=[1, 1], A=[[-1, -1]], b=[-3], integers=[0, 1])
        assert result.status == Status.OPTIMAL  # Small problem, should prove optimality
        assert abs(result.objective - 3.0) < 1e-6

    def test_maximize(self):
        # maximize x + y, x integer, x + y <= 5.5, x <= 3
        result = solve_milp(c=[1, 1], A=[[1, 1], [1, 0]], b=[5.5, 3], integers=[0], minimize=False)
        assert result.status == Status.OPTIMAL  # Small problem, should prove optimality
        assert result.solution[0] == round(result.solution[0])


class TestIntegerFeasibility:
    def test_unbounded_milp(self):
        # maximize x with constraint that doesn't bound it
        # minimize -x (maximize x), x + y >= 1 (doesn't bound x from above)
        result = solve_milp(c=[-1, 0], A=[[-1, -1]], b=[-1], integers=[0])
        assert result.status == Status.UNBOUNDED

    def test_integer_gap(self):
        # x integer, 1.5 <= x <= 1.9 -> infeasible (no integer in range)
        result = solve_milp(c=[1], A=[[-1], [1]], b=[-1.5, 1.9], integers=[0])
        assert result.status == Status.INFEASIBLE

    def test_integer_bounds(self):
        # x integer, x <= 2.9 -> x <= 2 (maximize -x is same as minimize x with x<=2)
        # minimize x where x is integer, x <= 2.9
        result = solve_milp(c=[1], A=[[1]], b=[2.9], integers=[0])
        assert result.status == Status.OPTIMAL  # Single variable, should prove optimality
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
        assert result.status == Status.OPTIMAL  # Small knapsack, should prove optimality
        # Should select items to maximize value within weight
        assert result.objective >= 7 - 1e-6


class TestEdgeCases:
    def test_already_integer_relaxation(self):
        # LP relaxation already gives integer solution
        result = solve_milp(c=[1, 1], A=[[-1, -1]], b=[-4], integers=[0, 1])
        assert result.status == Status.OPTIMAL  # LP relaxation is integer, immediate optimal
        # Both should be integers
        assert result.solution[0] == round(result.solution[0])
        assert result.solution[1] == round(result.solution[1])

    def test_no_integer_constraints(self):
        # All continuous (empty integers list) should work like LP
        result = solve_milp(c=[1, 1], A=[[-1, -1]], b=[-3], integers=[])
        assert result.status == Status.OPTIMAL  # Pure LP, should be optimal
        assert abs(result.objective - 3.0) < 1e-6

    def test_single_variable_integer(self):
        # Single integer variable: maximize x (minimize -x), x <= 5
        result = solve_milp(c=[-1], A=[[1]], b=[5], integers=[0])
        assert result.status == Status.OPTIMAL  # Single variable, should prove optimality
        # Solution must be integer
        assert abs(result.solution[0] - round(result.solution[0])) < 1e-6


class TestWarmStart:
    def test_warm_start_feasible(self):
        """Warm start with feasible solution speeds up search."""
        # minimize x + y, x + y >= 5, x,y integers
        c = [1, 1]
        A = [[-1, -1]]
        b = [-5]

        # Solve without warm start first
        result_cold = solve_milp(c, A, b, integers=[0, 1])
        assert result_cold.status == Status.OPTIMAL  # Small problem, should prove optimality

        # Warm start with the solution
        result_warm = solve_milp(c, A, b, integers=[0, 1], warm_start=result_cold.solution)
        assert result_warm.status == Status.OPTIMAL  # Small problem, should prove optimality
        assert abs(result_warm.objective - result_cold.objective) < 1e-6

    def test_warm_start_suboptimal(self):
        """Warm start with suboptimal feasible solution still improves."""
        # minimize x + y, x + y >= 3, x,y integers, x,y <= 5
        c = [1, 1]
        A = [[-1, -1], [1, 0], [0, 1]]
        b = [-3, 5, 5]

        # Provide suboptimal warm start (5, 5) -> objective 10, optimal is 3
        warm = [5.0, 5.0]
        result = solve_milp(c, A, b, integers=[0, 1], warm_start=warm)
        assert result.status == Status.OPTIMAL  # Small problem, should prove optimality
        # Should find optimal (objective = 3)
        assert abs(result.objective - 3.0) < 1e-6

    def test_warm_start_infeasible_ignored(self):
        """Infeasible warm start is ignored gracefully."""
        # minimize x, x >= 5, x integer
        c = [1]
        A = [[-1]]
        b = [-5]

        # Warm start with infeasible point
        result = solve_milp(c, A, b, integers=[0], warm_start=[2.0])
        assert result.status == Status.OPTIMAL  # Single variable, should prove optimality
        assert result.solution[0] >= 5 - 1e-6

    def test_warm_start_wrong_length_ignored(self):
        """Warm start with wrong length is ignored."""
        c = [1, 1]
        A = [[-1, -1]]
        b = [-3]

        # Wrong length warm start
        result = solve_milp(c, A, b, integers=[0, 1], warm_start=[1.0])
        assert result.status == Status.OPTIMAL  # Small problem, should prove optimality


class TestSolutionPool:
    def test_solution_limit_one(self):
        """Default solution_limit=1 returns single solution."""
        c = [1, 1]
        A = [[-1, -1], [1, 0], [0, 1]]
        b = [-2, 5, 5]
        result = solve_milp(c, A, b, integers=[0, 1])
        assert result.solutions is None or len(result.solutions) == 1

    def test_solution_limit_multiple(self):
        """solution_limit > 1 collects multiple solutions."""
        # minimize x + y, x + y >= 3, x,y in [0,3], integers
        c = [1, 1]
        A = [[-1, -1], [1, 0], [0, 1]]
        b = [-3, 3, 3]
        result = solve_milp(c, A, b, integers=[0, 1], solution_limit=5)
        # Should find at least one solution
        assert result.ok
        if result.solutions:
            # All solutions should be feasible
            for sol in result.solutions:
                assert sol[0] + sol[1] >= 3 - 1e-6
                assert all(abs(x - round(x)) < 1e-6 for x in sol)

    def test_solutions_are_different(self):
        """Multiple solutions are distinct."""
        c = [1, 1]
        A = [[-1, -1], [1, 0], [0, 1]]
        b = [-3, 5, 5]
        result = solve_milp(c, A, b, integers=[0, 1], solution_limit=10)
        if result.solutions and len(result.solutions) > 1:
            # Check solutions are different
            seen = set()
            for sol in result.solutions:
                key = tuple(round(x) for x in sol)
                assert key not in seen
                seen.add(key)


class TestStress:
    def test_multiple_integers(self):
        # 5 variables, 3 integer
        n = 5
        c = [1.0] * n
        A = [[-1.0] * n]
        b = [-10.5]
        result = solve_milp(c=c, A=A, b=b, integers=[0, 2, 4])
        assert result.status == Status.OPTIMAL  # Small problem, should prove optimality
        # Integer variables should be integers
        for i in [0, 2, 4]:
            assert abs(result.solution[i] - round(result.solution[i])) < 1e-6

    def test_tight_integer_problem(self):
        # Simple integer problem: maximize x + y (minimize -x - y), x + y <= 10
        result = solve_milp(c=[-1, -1], A=[[1, 1]], b=[10], integers=[0, 1])
        assert result.status == Status.OPTIMAL  # Small problem, should prove optimality
        x, y = result.solution
        # All should be integers
        assert abs(x - round(x)) < 1e-6
        assert abs(y - round(y)) < 1e-6
        # Constraint satisfied
        assert x + y <= 10 + 1e-6
        # Verify optimal objective
        assert abs(result.objective - (-10.0)) < 1e-6

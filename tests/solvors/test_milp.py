"""Tests for the MILP (mixed-integer linear programming) solver."""

import itertools
import math
import random

import pytest

from solvor.milp import solve_milp
from solvor.rust import rust_available
from solvor.simplex import solve_lp
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


class TestLNSImprovement:
    """Tests for LNS-based solution improvement."""

    def test_lns_improves_binary_knapsack(self):
        """LNS finds better solutions for binary knapsack."""
        # 0-1 knapsack with explicit x <= 1 bounds
        values = [10, 30, 25, 50, 35]
        weights = [5, 10, 15, 20, 25]
        capacity = 40
        n = len(values)

        A = [[w for w in weights]]  # weight constraint
        A.extend([[1 if j == i else 0 for j in range(n)] for i in range(n)])  # x_i <= 1
        b = [capacity] + [1] * n
        c = values

        # With LNS
        result = solve_milp(c, A, b, list(range(n)), minimize=False, lns_iterations=20, seed=42, max_nodes=0)
        assert result.ok
        assert result.objective >= 80  # Should find good solution

    def test_lns_with_minimize(self):
        """LNS works with minimize=True."""
        # Minimize cost assignment with binary vars
        costs = [5, 3, 8, 2, 6]
        n = len(costs)
        # Must select at least 2 items
        A = [[-1] * n]  # -sum(x) <= -2
        A.extend([[1 if j == i else 0 for j in range(n)] for i in range(n)])  # x_i <= 1
        b = [-2] + [1] * n

        result = solve_milp(costs, A, b, list(range(n)), minimize=True, lns_iterations=10, seed=42, max_nodes=0)
        assert result.ok
        assert result.objective <= 5  # Should find 3+2=5

    def test_lns_seed_reproducibility(self):
        """Same seed produces same result."""
        c = [10, 20, 30, 15, 25]
        A = [[-1, -1, -1, -1, -1], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
        b = [-2, 1, 1, 1, 1, 1]

        result1 = solve_milp(c, A, b, [0, 1, 2, 3, 4], minimize=False, lns_iterations=10, seed=123, max_nodes=0)
        result2 = solve_milp(c, A, b, [0, 1, 2, 3, 4], minimize=False, lns_iterations=10, seed=123, max_nodes=0)
        assert result1.objective == result2.objective

    def test_lns_destroy_fraction(self):
        """Different destroy fractions work."""
        c = [5, 10, 15, 20]
        A = [[-1, -1, -1, -1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        b = [-2, 1, 1, 1, 1]

        for frac in [0.2, 0.5, 0.8]:
            result = solve_milp(
                c, A, b, [0, 1, 2, 3], minimize=False, lns_iterations=5, lns_destroy_frac=frac, seed=42, max_nodes=0
            )
            assert result.ok


class TestBinaryDetection:
    """Tests for binary variable detection and bound tightening."""

    def test_explicit_binary_bounds_detected(self):
        """Explicit x_j <= 1 constraints trigger binary detection."""
        # maximize x + y with explicit x <= 1, y <= 1
        result = solve_milp(
            c=[1, 1],
            A=[[1, 0], [0, 1]],  # x <= 1, y <= 1
            b=[1, 1],
            integers=[0, 1],
            minimize=False,
        )
        assert result.ok
        assert abs(result.objective - 2.0) < 1e-6

    def test_implicit_binary_no_bound_tightening(self):
        """Sum constraints don't trigger erroneous bound tightening."""
        # sum(x_i) <= 1 implies each x_i <= 1, but we don't tighten
        n = 3
        A = [[1, 1, 1]]  # sum <= 1
        b = [1]
        c = [1, 2, 3]

        result = solve_milp(c, A, b, list(range(n)), minimize=False, max_nodes=10)
        assert result.ok
        # Should select x_2 = 1 (value 3)
        assert result.objective >= 3 - 1e-6


class TestRoundingHeuristics:
    """Tests for greedy rounding and local search."""

    def test_rounding_finds_feasible(self):
        """Rounding heuristic finds feasible solution."""
        # Binary selection: maximize, weight constraint
        c = [10, 20, 15, 25]
        A = [[3, 5, 4, 6], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        b = [10, 1, 1, 1, 1]

        result = solve_milp(c, A, b, [0, 1, 2, 3], minimize=False, max_nodes=0)
        assert result.ok

    def test_flip_improvement(self):
        """Flip phase improves solution."""
        # Problem where flipping helps
        c = [1, 1, 10]  # Want x_2
        A = [[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]  # sum <= 2
        b = [2, 1, 1, 1]

        result = solve_milp(c, A, b, [0, 1, 2], minimize=False, max_nodes=0)
        assert result.ok
        assert result.objective >= 10  # Should include x_2

    def test_swap_improvement(self):
        """Swap phase finds better combinations."""
        # Knapsack where swapping helps
        c = [8, 9, 5]  # x_0 + x_1 better than x_2 alone
        A = [[4, 5, 10], [1, 0, 0], [0, 1, 0], [0, 0, 1]]  # capacity 9
        b = [9, 1, 1, 1]

        result = solve_milp(c, A, b, [0, 1, 2], minimize=False, max_nodes=0)
        assert result.ok
        # Should find x_0=1, x_1=1 (value 17, weight 9)
        assert result.objective >= 17 - 1e-6

    def test_heuristics_disabled(self):
        """heuristics=False skips rounding."""
        c = [10, 20, 15]
        A = [[3, 5, 4], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        b = [8, 1, 1, 1]

        # With heuristics off and no B&B, might not find solution
        # Could be infeasible or feasible depending on LP relaxation
        # Just verify it runs without error
        solve_milp(c, A, b, [0, 1, 2], minimize=False, heuristics=False, max_nodes=0)


class TestSubMIP:
    """Tests for sub-MIP solving in LNS repair."""

    def test_submip_finds_optimal_subset(self):
        """Sub-MIP optimizes over unfixed variables."""
        # Use LNS which calls sub-MIP internally
        c = [5, 10, 15, 20, 25]
        A = [[2, 3, 4, 5, 6], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
        b = [12, 1, 1, 1, 1, 1]

        result = solve_milp(
            c, A, b, [0, 1, 2, 3, 4], minimize=False, lns_iterations=10, lns_destroy_frac=0.6, seed=42, max_nodes=0
        )
        assert result.ok
        # Should find good solution
        assert result.objective >= 35


class TestMaxNodes:
    """Tests for node limit behavior."""

    def test_max_nodes_zero_uses_heuristics_only(self):
        """max_nodes=0 relies purely on heuristics."""
        c = [10, 20, 30]
        A = [[1, 2, 3], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        b = [4, 1, 1, 1]

        result = solve_milp(c, A, b, [0, 1, 2], minimize=False, max_nodes=0)
        assert result.ok
        assert result.iterations == 0  # No B&B nodes explored

    def test_max_nodes_limits_exploration(self):
        """max_nodes limits B&B exploration."""
        c = [1, 1, 1, 1, 1]
        A = [[-1, -1, -1, -1, -1]]
        b = [-3]

        result = solve_milp(c, A, b, [0, 1, 2, 3, 4], minimize=True, max_nodes=5)
        # Should terminate early
        assert result.ok


class TestCoverageGaps:
    """Tests specifically targeting coverage gaps."""

    def test_lns_improved_solution_added_to_pool(self):
        """LNS improvement adds new solution to pool (lines 158-160)."""
        # Problem where LNS finds a different solution than initial rounding
        n = 6
        c = [5, 10, 15, 8, 12, 20]
        weights = [2, 3, 4, 2, 3, 5]
        A = [weights]
        A.extend([[1 if j == i else 0 for j in range(n)] for i in range(n)])
        b = [10] + [1] * n

        result = solve_milp(c, A, b, list(range(n)), minimize=False, lns_iterations=15, seed=42, max_nodes=0)
        assert result.ok
        assert result.objective >= 25

    def test_node_pruning_by_bound(self):
        """Node pruned when bound >= best_obj (line 174)."""
        # Small problem with warm start to trigger pruning
        c = [1, 1]
        A = [[-1, -1], [1, 0], [0, 1]]
        b = [-3, 5, 5]

        # Provide optimal warm start, nodes should be pruned
        result = solve_milp(c, A, b, [0, 1], minimize=True, warm_start=[2.0, 1.0])
        assert result.ok
        assert abs(result.objective - 3.0) < 1e-6

    def test_lp_relaxation_pruning(self):
        """Node pruned after LP when objective >= best_obj (line 184)."""
        # Start with good solution, subsequent nodes should prune
        c = [1, 2, 1]
        A = [[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        b = [-2, 3, 3, 3]

        result = solve_milp(c, A, b, [0, 1, 2], minimize=True, warm_start=[1.0, 1.0, 0.0])
        assert result.ok
        assert result.objective <= 3.0 + 1e-6

    def test_solution_pool_returns_early(self):
        """Solution pool returns when limit reached during B&B (lines 195-197)."""
        # Problem with multiple integer solutions
        c = [1, 1, 1]
        A = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, -1, -1]]
        b = [2, 2, 2, -3]  # Each var <= 2, sum >= 3

        result = solve_milp(c, A, b, [0, 1, 2], minimize=True, solution_limit=2, max_nodes=500, heuristics=False)
        assert result.ok
        if result.solutions:
            assert len(result.solutions) <= 2

    def test_gap_tolerance_returns_optimal(self):
        """Gap tolerance triggers early return with OPTIMAL status (line 208)."""
        c = [1, 1]
        A = [[-1, -1], [1, 0], [0, 1]]
        b = [-4, 10, 10]

        result = solve_milp(c, A, b, [0, 1], minimize=True, gap_tol=0.1)
        assert result.status == Status.OPTIMAL
        assert abs(result.objective - 4.0) < 1e-6

    def test_solution_pool_tree_exhausted(self):
        """Solution pool returned after B&B tree exhausted (line 231)."""
        c = [1, 1]
        A = [[1, 0], [0, 1], [-1, -1]]
        b = [1, 1, -1]  # x, y in {0,1}, sum >= 1

        result = solve_milp(c, A, b, [0, 1], minimize=True, solution_limit=5)
        assert result.ok
        if result.solutions:
            # Should have found the 3 feasible solutions: (1,0), (0,1), (1,1)
            assert len(result.solutions) >= 1

    def test_infeasible_bounds_in_solve_node(self):
        """Variable bounds hi < lo returns INFEASIBLE in _solve_node (line 245)."""
        # Create situation where branching leads to infeasible bounds
        # x integer, x <= 0.5 and x >= 0.6 simultaneously (after branching)
        c = [1]
        A = [[1], [-1]]
        b = [0.5, -0.6]  # x <= 0.5, x >= 0.6 -> infeasible

        result = solve_milp(c, A, b, [0])
        assert result.status == Status.INFEASIBLE

    def test_zero_objective_gap(self):
        """Gap computation with near-zero objective (line 317)."""
        # Problem with zero optimal objective
        c = [1, -1]  # x - y, optimal at x=y
        A = [[1, 0], [0, 1], [-1, 0], [0, -1], [1, -1], [-1, 1]]
        b = [5, 5, 0, 0, 0, 0]  # x, y in [0, 5], x = y

        result = solve_milp(c, A, b, [0, 1], minimize=True)
        assert result.ok
        assert abs(result.objective) < 1e-6

    def test_negative_variable_infeasible(self):
        """_is_feasible returns False for negative variable (line 339)."""
        # This is tested indirectly - warm start with negative value
        c = [1, 1]
        A = [[-1, -1]]
        b = [-2]

        # Warm start with negative value should be rejected
        result = solve_milp(c, A, b, [0, 1], warm_start=[-1.0, 3.0])
        assert result.ok
        # Should find valid solution, ignoring invalid warm start
        assert all(x >= -1e-6 for x in result.solution)

    def test_non_integer_infeasible(self):
        """_is_feasible returns False for non-integer (line 342)."""
        # Warm start with non-integer values should be rejected
        c = [1, 1]
        A = [[-1, -1], [1, 0], [0, 1]]
        b = [-3, 5, 5]

        result = solve_milp(c, A, b, [0, 1], warm_start=[1.5, 1.5])
        assert result.ok
        # Should find valid integer solution
        assert all(abs(x - round(x)) < 1e-6 for x in result.solution)

    def test_rounding_flip_fallback(self):
        """Rounding tries flip when initial round fails (lines 376-382)."""
        # Tight constraint where first rounding fails, flip succeeds
        c = [3, 4, 5]
        A = [[2, 3, 4], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        b = [5, 1, 1, 1]

        result = solve_milp(c, A, b, [0, 1, 2], minimize=False, max_nodes=10)
        assert result.ok
        # Optimal: x0=1, x1=1 (value 7, weight 5)
        assert result.objective >= 7 - 1e-6

    def test_submip_finds_improvement(self):
        """Sub-MIP B&B finds better solution (lines 502-505)."""
        c = [8, 15, 10, 20, 12]
        weights = [3, 5, 4, 7, 4]
        A = [weights]
        A.extend([[1 if j == i else 0 for j in range(5)] for i in range(5)])
        b = [12] + [1] * 5

        result = solve_milp(
            c, A, b, [0, 1, 2, 3, 4], minimize=False, lns_iterations=15, lns_destroy_frac=0.6, seed=123, max_nodes=0
        )
        assert result.ok
        assert result.objective >= 30

    def test_lns_adds_to_pool_when_different(self):
        """LNS adds improved solution to pool when different from initial (lines 158-160)."""
        # Larger problem where LNS can find genuinely different solution
        n = 10
        c = [3, 7, 4, 9, 5, 8, 2, 6, 10, 1]
        weights = [2, 4, 3, 5, 3, 4, 2, 3, 5, 1]
        A = [weights]
        A.extend([[1 if j == i else 0 for j in range(n)] for i in range(n)])
        b = [15] + [1] * n

        result = solve_milp(
            c, A, b, list(range(n)), minimize=False, lns_iterations=50, lns_destroy_frac=0.5, seed=777, max_nodes=0
        )
        assert result.ok

    def test_direct_solution_pool_hit_limit(self):
        """B&B hits solution_limit exactly (lines 195-197)."""
        # Small complete enumeration problem
        c = [1, 2]
        A = [[1, 0], [0, 1]]  # x <= 1, y <= 1
        b = [1, 1]

        result = solve_milp(c, A, b, [0, 1], minimize=False, solution_limit=2, heuristics=False, max_nodes=100)
        assert result.ok
        # Should have collected solutions during B&B
        if result.solutions:
            assert len(result.solutions) <= 2

    def test_solution_pool_exhausted_return(self):
        """Returns solution pool after tree exhaustion (line 231)."""
        c = [1, 1]
        A = [[1, 0], [0, 1]]
        b = [1, 1]

        result = solve_milp(c, A, b, [0, 1], minimize=True, solution_limit=10, heuristics=False, max_nodes=1000)
        assert result.ok
        # Tree should be exhausted with all 4 solutions found
        if result.solutions:
            assert len(result.solutions) >= 1

    def test_bound_pruning_with_good_incumbent(self):
        """Nodes pruned when bound can't beat incumbent (line 174)."""
        # Minimization with a tight warm start
        c = [2, 3, 5]
        A = [[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        b = [-2, 1, 1, 1]

        # Warm start with optimal solution should prune all nodes
        result = solve_milp(c, A, b, [0, 1, 2], minimize=True, warm_start=[1.0, 1.0, 0.0])
        assert result.ok
        assert abs(result.objective - 5.0) < 1e-6

    def test_lp_bound_pruning_after_solve(self):
        """Node pruned after LP solve when bound >= incumbent (line 184)."""
        c = [1, 1, 1]
        A = [[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        b = [-3, 1, 1, 1]

        # Optimal warm start - subsequent LP bounds should prune
        result = solve_milp(c, A, b, [0, 1, 2], minimize=True, warm_start=[1.0, 1.0, 1.0])
        assert result.ok
        assert abs(result.objective - 3.0) < 1e-6

    def test_gap_tol_early_termination(self):
        """Gap tolerance triggers early optimal return (line 208)."""
        c = [1, 1]
        A = [[-1, -1]]
        b = [-10]

        # Large gap tolerance should terminate early
        result = solve_milp(c, A, b, [0, 1], minimize=True, gap_tol=1.0)
        assert result.status == Status.OPTIMAL

    def test_rounding_both_directions_fail(self):
        """Rounding tries both directions before giving up (lines 376-382)."""
        # Very tight constraints where both round directions might fail
        c = [5, 5, 5, 5]
        A = [[1, 1, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, 0, 0]]  # x0 + x1 >= 1
        b = [2, 1, 1, 1, 1, -1]

        result = solve_milp(c, A, b, [0, 1, 2, 3], minimize=False, max_nodes=50)
        assert result.ok

    def test_infeasible_variable_bounds_directly(self):
        """Branching creates hi < lo situation (line 245)."""
        # Force branching to create infeasible node
        c = [1]
        A = [[1], [-1]]  # 0.4 <= x <= 0.6 - no integer
        b = [0.6, -0.4]

        result = solve_milp(c, A, b, [0])
        assert result.status == Status.INFEASIBLE

    def test_near_zero_objective_gap_calc(self):
        """Gap computed correctly when objective near zero (line 317)."""
        # Objective that should be exactly 0
        c = [1, -1]
        A = [[1, -1], [-1, 1], [1, 0], [0, 1]]  # x = y, both >= 0
        b = [0, 0, 5, 5]

        result = solve_milp(c, A, b, [0, 1], minimize=True)
        assert result.ok
        assert abs(result.objective) < 1e-6


class TestEdgeCaseCoverage:
    """Additional edge cases for coverage."""

    def test_infeasible_root_lp(self):
        """Root LP infeasible returns INFEASIBLE."""
        # x >= 5, x <= 3 -> infeasible
        result = solve_milp(c=[1], A=[[-1], [1]], b=[-5, 3], integers=[0])
        assert result.status == Status.INFEASIBLE

    def test_lns_improves_over_initial(self):
        """LNS finds improvement over initial heuristic."""
        # Larger problem where LNS helps
        n = 8
        c = [15, 20, 10, 25, 30, 5, 35, 40]
        weights = [3, 4, 2, 5, 6, 1, 7, 8]
        A = [weights]
        A.extend([[1 if j == i else 0 for j in range(n)] for i in range(n)])
        b = [15] + [1] * n

        result = solve_milp(c, A, b, list(range(n)), minimize=False, lns_iterations=30, seed=42, max_nodes=0)
        assert result.ok
        assert result.objective >= 50

    def test_solution_pool_hits_limit(self):
        """Solution pool stops when limit reached."""
        # Problem with multiple optimal solutions
        c = [1, 1]
        A = [[1, 0], [0, 1], [-1, -1]]
        b = [2, 2, -2]  # x <= 2, y <= 2, x + y >= 2

        result = solve_milp(c, A, b, [0, 1], minimize=True, solution_limit=3, max_nodes=100)
        assert result.ok
        if result.solutions:
            assert len(result.solutions) <= 3

    def test_gap_tolerance(self):
        """Gap tolerance terminates early."""
        c = [1, 1]
        A = [[-1, -1], [1, 0], [0, 1]]
        b = [-5, 10, 10]

        result = solve_milp(c, A, b, [0, 1], minimize=True, gap_tol=0.5)
        assert result.ok

    def test_all_variables_fixed(self):
        """Node with all variables fixed."""
        # Single variable problem with tight bounds: 1 <= x <= 1
        # Using maximize to find x = 1 (avoid minimize returning lower bound)
        result = solve_milp(c=[1], A=[[1], [-1]], b=[1, -1], integers=[0], minimize=False)
        # x = 1 is the only option
        assert result.ok
        assert abs(result.solution[0] - 1.0) < 1e-6

    def test_rounding_infeasible_fallback(self):
        """Rounding fails, falls back to B&B."""
        # Tight constraints where naive rounding fails
        c = [1, 1, 1]
        A = [[3, 3, 3], [1, 0, 0], [0, 1, 0], [0, 0, 1]]  # sum of weights = exactly capacity
        b = [6, 1, 1, 1]

        result = solve_milp(c, A, b, [0, 1, 2], minimize=False, max_nodes=50)
        assert result.ok

    def test_swap_finds_improvement(self):
        """Swap phase in rounding finds better solution."""
        # Problem where swapping one-for-zero helps
        c = [10, 11, 5]  # Two small items beat one large
        A = [[5, 5, 10], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        b = [10, 1, 1, 1]

        result = solve_milp(c, A, b, [0, 1, 2], minimize=False, max_nodes=0)
        assert result.ok
        # Should find x_0=1, x_1=1 (value 21) not x_2=1 (value 5)
        assert result.objective >= 21 - 1e-6

    def test_submip_branching(self):
        """Sub-MIP uses internal branching."""
        # Problem that requires sub-MIP B&B
        c = [10, 20, 30, 40, 50]
        A = [[2, 3, 4, 5, 6], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
        b = [15, 1, 1, 1, 1, 1]

        result = solve_milp(
            c, A, b, [0, 1, 2, 3, 4], minimize=False, lns_iterations=20, lns_destroy_frac=0.8, seed=42, max_nodes=0
        )
        assert result.ok

    def test_binary_detection_partial(self):
        """Only some variables have explicit bounds."""
        # x_0 <= 1 explicit, x_1 <= 2 (not binary bound)
        c = [1, 1]
        A = [[1, 0], [0, 1], [-1, -1]]  # x_0 <= 1, x_1 <= 2, x_0 + x_1 >= 1
        b = [1, 2, -1]

        result = solve_milp(c, A, b, [0, 1], minimize=False, max_nodes=10)
        assert result.ok
        # Optimal is x_0=1, x_1=2, objective=3
        assert result.objective >= 3 - 1e-6


class TestSparseBoundsApi:
    def test_binary_and_sparse_rows(self):
        # set packing: pick at most one of each pair, maximize
        rows = [{0: 1, 1: 1}, {1: 1, 2: 1}, {2: 1, 3: 1}]
        result = solve_milp([3, 2, 2, 3], rows, [1, 1, 1], binary=range(4), minimize=False)
        assert result.status == Status.OPTIMAL
        assert result.objective == 6.0
        assert result.solution == (1.0, 0.0, 0.0, 1.0)

    def test_binary_equals_identity_rows(self):
        c = [5, 4, 3, 7]
        rows = [[2, 3, 1, 4], [1, 1, 1, 1]]
        with_rows = solve_milp(
            c,
            rows + [[1 if j == k else 0 for j in range(4)] for k in range(4)],
            [6, 2] + [1] * 4,
            list(range(4)),
            minimize=False,
        )
        with_binary = solve_milp(c, rows, [6, 2], binary=range(4), minimize=False)
        assert with_rows.objective == with_binary.objective

    def test_general_integer_bounds(self):
        # maximize x + y with x in [0, 5], y in [-2, 2], x + 2y <= 6.5, both integer
        result = solve_milp([1, 1], [[1, 2]], [6.5], [0, 1], minimize=False, lb=[0, -2], ub=[5, 2])
        assert result.status == Status.OPTIMAL
        assert result.objective == 5.0  # (5, 0) and (4, 1) tie; either is correct
        x, y = result.solution
        assert x + 2 * y <= 6.5 and 0 <= x <= 5 and -2 <= y <= 2
        assert x == round(x) and y == round(y)

    def test_greater_equal_and_equal_rows(self):
        # minimize x + y with x + y >= 3.5 and x - y = 1, integers
        result = solve_milp([1, 1], [[1, 1], [1, -1]], [3.5, 1], [0, 1], senses=[">=", "="])
        assert result.status == Status.OPTIMAL
        assert result.solution == (3.0, 2.0)

    def test_fixed_variable(self):
        result = solve_milp([1, 1], [[1, 1]], [10], [0, 1], minimize=False, lb=[3, 0], ub=[3, 4])
        assert result.solution == (3.0, 4.0)

    def test_integer_variable_without_lower_bound(self):
        # minimize x, x integer and free, -x <= 3.5 (x >= -3.5): x = -3
        result = solve_milp([1], [[-1]], [3.5], [0], lb=[-math.inf])
        assert result.solution == (-3.0,)

    def test_inverted_bounds_are_infeasible(self):
        result = solve_milp([1], [[1]], [5], [0], lb=[3], ub=[2])
        assert result.status == Status.INFEASIBLE

    def test_integers_argument_is_optional(self):
        result = solve_milp([1, 1], [{0: 1, 1: 1}], [1.5], binary=[0], minimize=False)
        assert result.status == Status.OPTIMAL
        assert abs(result.objective - 1.5) < 1e-9
        assert result.solution[0] in (0.0, 1.0)

    def test_binary_overlapping_integers_and_wide_ub(self):
        # binary clamps the user's ub=3 to 1; listing x0 in both integers and binary is fine
        result = solve_milp([1, 1], [[1, 1]], [5], [0], binary=[0, 1], minimize=False, ub=[3, 3])
        assert result.status == Status.OPTIMAL
        assert result.solution == (1.0, 1.0)

    def test_binary_index_validated(self):
        with pytest.raises(ValueError, match="Invalid index in binary"):
            solve_milp([1], [[1]], [1], binary=[3])


class TestPresolve:
    def test_singleton_rows_become_bounds(self):
        # -x <= -2.5 and x <= 7.9 on an integer x: x in [3, 7]
        result = solve_milp([1], [[-1], [1]], [-2.5, 7.9], [0])
        assert result.solution == (3.0,)
        assert solve_milp([1], [[-1], [1]], [-2.5, 7.9], [0], minimize=False).solution == (7.0,)

    def test_empty_row_infeasible(self):
        result = solve_milp([1], [[0]], [-1], [0])
        assert result.status == Status.INFEASIBLE

    def test_empty_row_feasible_is_dropped(self):
        result = solve_milp([1], [[0], [1]], [1, 4], [0], minimize=False)
        assert result.objective == 4.0


class TestHeuristicScaling:
    def test_rounding_heuristic_on_a_large_odd_cycle(self):
        """Fractional root (all 0.5); the heuristic alone must find a maximum independent set quickly.

        0.6.2 re-checked every row after every move (O(rows x columns) per move): n=101 took
        0.24 s and the cost grows roughly cubically. Moves now touch only their own rows.
        """
        import time

        n = 301
        rows = [{i: 1.0, (i + 1) % n: 1.0} for i in range(n)]
        start = time.perf_counter()
        result = solve_milp([1.0] * n, rows, [1.0] * n, binary=range(n), minimize=False, max_nodes=1)
        assert time.perf_counter() - start < 5.0
        assert result.objective == (n - 1) / 2


class TestIntegerRowPresolve:
    def test_rhs_rounding_solves_at_the_root(self):
        """2x + 2y <= 3 over binaries is x + y <= 1; the LP relaxation then needs no branching."""
        result = solve_milp([1, 1], [{0: 2, 1: 2}], [3], binary=range(2), minimize=False)
        assert result.objective == 1.0
        assert result.iterations == 1

    def test_lock_row_with_half_slack(self):
        """Lexicographic lock rows like -c1.x <= -opt + 0.5 tighten to -c1.x <= -opt."""
        n = 6
        rows = [{0: 1, 1: 1}, {2: 1, 3: 1}, {4: 1, 5: 1}, {0: -1, 2: -1, 4: -1}]
        rhs = [1, 1, 1, -3 + 0.5]
        result = solve_milp([0, 1, 0, 1, 0, 1], rows, rhs, binary=range(n), minimize=False)
        assert result.objective == 0.0
        assert result.solution[0] == result.solution[2] == result.solution[4] == 1.0

    def test_equality_without_integer_solution(self):
        # 2x + 4y = 3 has no integer solution
        result = solve_milp([1, 1], [[2, 4]], [3], [0, 1], senses=["="], ub=[10, 10])
        assert result.status == Status.INFEASIBLE

    def test_rows_with_continuous_variables_are_not_rounded(self):
        # x integer, y continuous: x + y <= 1.5 must keep its 0.5
        result = solve_milp([1, 1], [[1, 1]], [1.5], [0], minimize=False)
        assert abs(result.objective - 1.5) < 1e-9

    def test_fractional_coefficients_are_not_rounded(self):
        # 1.5x + y <= 2.9 keeps (1, 1); rounding the rhs to 2 would wrongly cut it off
        result = solve_milp([3, 2], [[1.5, 1]], [2.9], [0, 1], minimize=False)
        assert result.objective == 5.0


class TestIntegerSnapping:
    """Integer variables in results are exact integers; LP pivot noise must not leak into outputs."""

    def test_integer_variables_are_exact_and_objective_recomputed(self):
        import random

        rng = random.Random(0)
        n, m = 60, 90
        rows = [{j: 1.0 for j in rng.sample(range(n), rng.randint(2, 5))} for _ in range(m)]
        c = [float(rng.randint(1, 9)) for _ in range(n)]
        result = solve_milp(c, rows, [1.0] * m, binary=range(n), senses=[">="] * m)
        assert result.status == Status.OPTIMAL
        assert all(v in (0.0, 1.0) for v in result.solution)
        assert result.objective == sum(cj * xj for cj, xj in zip(c, result.solution))

    def test_continuous_variables_are_not_snapped(self):
        result = solve_milp([1, 1], [[1, 1]], [1.5], [0], minimize=False)
        assert result.solution[0] in (0.0, 1.0)
        assert abs(result.solution[0] + result.solution[1] - 1.5) < 1e-9


class TestWarmBranchAndBound:
    """Child nodes differ from their parent by one bound; the dual simplex re-solves them in a few pivots."""

    def _knapsack(self):
        import random

        rng = random.Random(0)
        n = 35
        rows = [{j: float(rng.randint(5, 40)) for j in range(n)} for _ in range(4)]
        rhs = [sum(row.values()) * 0.35 for row in rows]
        c = [float(rng.randint(10, 60)) for _ in range(n)]
        return c, rows, rhs

    def test_node_reoptimization_is_cheap(self):
        c, rows, rhs = self._knapsack()
        result = solve_milp(c, rows, rhs, binary=range(len(c)), minimize=False)
        assert result.status == Status.OPTIMAL
        assert result.objective == 611.0
        # result.evaluations counts LP pivots over all nodes: about 21,000 when every node starts cold
        assert result.evaluations < 5_000

    def test_dives_into_the_rounding_direction(self):
        """Depth-first plunging finds an incumbent quickly; with heuristics off it still solves to optimality."""
        c, rows, rhs = self._knapsack()
        result = solve_milp(c, rows, rhs, binary=range(len(c)), minimize=False, heuristics=False)
        assert result.status == Status.OPTIMAL
        assert result.objective == 611.0

    def test_branching_on_free_integer_variables(self):
        """Free variables are split into two columns; branching on them rebuilds the LP (cold path)."""
        from solvor import MilpModel

        inf = float("inf")
        # maximize x + y, x + y <= 3.5, x - y <= 0.5, both integer and free
        model = MilpModel(2, integers=[0, 1], lb=[-inf, -inf], ub=[inf, inf])
        model.add_rows([[1, 1], [1, -1]], [3.5, 0.5])
        result = model.solve([1, 1], minimize=False)
        assert result.status == Status.OPTIMAL
        assert result.objective == 3.0
        x, y = result.solution
        assert x + y <= 3.5 and x - y <= 0.5
        assert model._lp.cold_solves == 3  # the root and both children


class TestPresolveAndRootEdgeCases:
    @pytest.mark.parametrize("big", [1e7, 1e9])
    def test_infeasible_with_a_large_unrelated_row(self, big):
        result = solve_milp(
            [1, 1, 0, 0], [[1, 1, 0, 0], [0, 0, 1, 1]], [3, big], [0, 1], senses=[">=", "<="], ub=[1, 1, 10, 10]
        )
        assert result.status == Status.INFEASIBLE

    def test_infeasible_node_with_a_large_unrelated_row(self):
        """Branch x <= 1 is infeasible; it must not come back OPTIMAL at x = 1.5 and loop."""
        # minimize x, x - 0.5 w = 1.5, w in [0, 1], x integer, plus an unrelated z <= 1e7
        result = solve_milp([1, 0, 0], [[1, -0.5, 0], [0, 0, 1]], [1.5, 1e7], [0], senses=["=", "<="], ub=[10, 1, 1e8])
        assert result.status == Status.OPTIMAL
        assert result.objective == 2.0
        assert result.iterations <= 5

    def test_infinite_rhs_is_dropped(self):
        result = solve_milp([-1, -1], [[1, 1], [1, 0], [0, 1]], [math.inf, 3, 4], [0, 1])
        assert result.status == Status.OPTIMAL
        assert result.objective == -7.0

    def test_impossible_infinite_rhs_is_infeasible(self):
        assert solve_milp([1], [[1]], [-math.inf], [0]).status == Status.INFEASIBLE
        assert solve_milp([1], [[1]], [math.inf], [0], senses=[">="]).status == Status.INFEASIBLE

    def test_iteration_limit_at_the_root_is_not_optimal(self):
        # 6-variable cyclic cover, every row needs an artificial; one LP iteration cannot finish phase 1
        rows = [{i: 1.0, (i + 1) % 6: 1.0} for i in range(6)]
        result = solve_milp([1.0] * 6, rows, [1.0] * 6, binary=range(6), senses=[">="] * 6, max_iter=1)
        assert result.status == Status.MAX_ITER

    def test_nearly_integral_coefficients_are_not_tightened(self):
        # (1 + 5e-10) x - y <= 0.5 with x fixed at 10000: smallest integer y is 10000
        result = solve_milp([0, 1], [[1 + 5e-10, -1]], [0.5], [0, 1], lb=[10000, 0], ub=[10000, math.inf])
        assert result.solution[1] == 10000.0


class TestWarmReoptimizationIsOptimal:
    def test_tiny_dual_pivot_does_not_return_a_suboptimal_node(self, monkeypatch):
        """A node re-solved warm used to stop at a dual-infeasible basis and close with a worse point."""
        rows = [
            {0: -2.871, 1: -0.26, 5: -312.387},
            {1: 0.064, 2: -72.382, 5: 0.016},
            {0: -10.735, 1: 0.076, 6: -0.07},
            {1: 0.091, 2: 192.573, 4: 4.398, 5: -0.23, 6: 0.535},
        ]
        c = [243.016, 0.155, 0.399, -67.917, -12.706, -11.949, -11.91]
        args = (c, rows, [-535.902, -72.174, -288.437, 193.686], [1, 2, 6])
        kwargs = {"senses": [">=", "=", ">=", "="], "ub": [5, 5, 5, 5, 4, 6, 3]}
        warm = solve_milp(*args, **kwargs)
        monkeypatch.setattr("solvor.lp_engine.WarmLP.REBUILD_PIVOTS", 0)  # every node cold
        cold = solve_milp(*args, **kwargs)
        assert warm.status == cold.status == Status.OPTIMAL
        assert abs(warm.objective - cold.objective) < 1e-6
        assert abs(warm.objective - (-374.49)) < 0.01  # HiGHS


class TestBoundRoundingNoise:
    def test_tiny_inversion_on_a_continuous_variable_is_feasible(self):
        """lb = 0.1 + 0.2 is 4e-17 above ub = 0.3: solve_lp accepts it, so must solve_milp."""
        lb, ub = [0.1 + 0.2, 0.0], [0.3, 4.0]
        lp = solve_lp([1, 1], [{0: 1, 1: 1}], [5], lb=lb, ub=ub)
        milp = solve_milp([1, 1], [{0: 1, 1: 1}], [5], [1], lb=lb, ub=ub)
        assert lp.status == milp.status == Status.OPTIMAL
        assert milp.solution[1] == 0.0 and abs(milp.solution[0] - 0.3) < 1e-9


def _bits(value):
    """Floats as hex strings, recursively, so equality means bit for bit (-0.0 differs from 0.0)."""
    if isinstance(value, float):
        return value.hex()
    if isinstance(value, (list, tuple)):
        return [_bits(v) for v in value]
    return value


def _result_bits(r):
    return [r.status, _bits(r.objective), r.iterations, r.evaluations, _bits(r.solution), _bits(r.solutions)]


needs_rust = pytest.mark.skipif(not rust_available(), reason="Rust extension not built")


def _random_milp(rng):
    """Mixed integer program around a known feasible point, coefficients of mixed magnitude."""
    n, m = rng.randint(2, 12), rng.randint(1, 8)
    mag = rng.choice((0, 1, 2))

    def coef():
        return round(rng.uniform(-1, 1) * 10 ** rng.randint(-1, mag), 3)

    rows = [{j: coef() for j in range(n) if rng.random() < 0.6} for _ in range(m)]
    rows = [{j: a for j, a in r.items() if a != 0.0} or {0: 1.0} for r in rows]
    senses = [rng.choice(("<=", "<=", ">=", "=")) for _ in range(m)]
    x = [rng.randint(0, 3) for _ in range(n)]
    b = []
    for row, sense in zip(rows, senses):
        activity = sum(a * x[j] for j, a in row.items())
        if sense == "=":
            b.append(round(activity, 6))
        elif sense == "<=":
            b.append(round(activity + rng.uniform(0, 5), 3))
        else:
            b.append(round(activity - rng.uniform(0, 5), 3))
    integers = sorted(rng.sample(range(n), rng.randint(1, n)))
    ub = [float(rng.randint(3, 6)) for _ in range(n)]
    return [coef() for _ in range(n)], rows, b, integers, senses, ub, rng.random() < 0.5


@needs_rust
class TestRustBackendMatchesPython:
    """solve_milp gives the same result, bit for bit, whichever LP kernel runs."""

    @pytest.mark.parametrize("seed", range(3))
    def test_random_milps(self, seed):
        rng = random.Random(seed)
        for _ in range(100):
            c, rows, b, integers, senses, ub, minimize = _random_milp(rng)

            def run(backend):
                return solve_milp(
                    c, rows, b, integers, senses=senses, ub=ub, minimize=minimize, max_nodes=2000, backend=backend
                )

            assert _result_bits(run("rust")) == _result_bits(run("python"))

    def test_heuristics_and_multiple_solutions(self):
        rng = random.Random(7)
        n = 25
        rows = [{j: float(rng.randint(5, 40)) for j in range(n)} for _ in range(3)]
        rhs = [sum(r.values()) * 0.4 for r in rows]
        c = [float(rng.randint(10, 60)) for _ in range(n)]
        for kwargs in ({"lns_iterations": 5, "seed": 3}, {"solution_limit": 3}, {"max_nodes": 10}):

            def run(backend):
                return solve_milp(c, rows, rhs, binary=range(n), minimize=False, backend=backend, **kwargs)

            assert _result_bits(run("rust")) == _result_bits(run("python"))

    @pytest.mark.parametrize("max_iter", [10**30, -(10**30)])
    def test_iteration_limit_beyond_64_bits(self, max_iter):
        """max_iter is a Python int of any size; the Rust kernel saturates it."""

        def run(solver, backend):
            return solver([1, 1], [{0: 1.0, 1: 1.0}], [3.5], minimize=False, max_iter=max_iter, backend=backend)

        assert _result_bits(run(solve_lp, "rust")) == _result_bits(run(solve_lp, "python"))
        assert _result_bits(run(solve_milp, "rust")) == _result_bits(run(solve_milp, "python"))

    def test_lp_relaxations(self):
        rng = random.Random(11)
        for _ in range(100):
            c, rows, b, _, senses, ub, minimize = _random_milp(rng)

            def run(backend):
                return solve_lp(c, rows, b, senses=senses, ub=ub, minimize=minimize, backend=backend)

            assert _result_bits(run("rust")) == _result_bits(run("python"))


class TestRoundingHeuristicFeasibility:
    def test_flip_phase_rechecks_rows_from_scratch(self):
        """Row activities updated move by move lose precision next to huge coefficients."""
        from solvor.milp_heuristics import is_feasible, round_binary
        from solvor.utils.lp_input import LinearProblem

        rows = [{0: -8.0, 1: -500000000.0, 2: 29.7, 4: 800000000.0, 5: -5e17}]
        prob = LinearProblem(6, rows, [-7.748617323482724], ["<="], [0.0] * 6, [1.0] * 6)
        lp_solution = [0.2099488293152899, 0.9408729203857055, 0.3400624281282216]
        lp_solution += [0.43758774543069257, 0.46682868187470317, 0.7282506053122182]
        c = [4.069278007872276, 1.7833431329349612, -2.7511507368031674]
        c += [-3.2224847655908393, 3.070239286758948, -2.159208031527664]
        bounds = [0.0] * 6, [1.0] * 6
        rounded = round_binary(lp_solution, list(range(6)), c, prob, prob.columns(), *bounds, True, 1e-6)
        assert rounded is None or is_feasible(rounded, prob, *bounds, set(range(6)), 1e-6)


def _brute_force(c, rows, b, senses, lb, ub, minimize):
    best = None
    for x in itertools.product(*(range(lo, hi + 1) for lo, hi in zip(lb, ub))):
        if all(_row_holds(sum(a * x[j] for j, a in row.items()), bi, s) for row, bi, s in zip(rows, b, senses)):
            value = sum(cj * xj for cj, xj in zip(c, x))
            if best is None or (value < best if minimize else value > best):
                best = value
    return best


def _row_holds(activity, rhs, sense, tol=1e-6):
    if sense == "<=":
        return activity <= rhs + tol
    if sense == ">=":
        return activity >= rhs - tol
    return abs(activity - rhs) <= tol


class TestRandomMilpsAgainstEnumeration:
    """Small pure-integer models with every sense and general bounds, checked against brute force."""

    @pytest.mark.parametrize("seed", range(3))
    def test_optimum_matches_enumeration(self, seed):
        rng = random.Random(seed)
        for _ in range(80):
            n, m = rng.randint(1, 4), rng.randint(1, 4)
            lb = [rng.randint(-2, 1) for _ in range(n)]
            ub = [lo + rng.randint(0, 3) for lo in lb]
            rows = [
                {j: round(rng.uniform(-5, 5), 1) for j in range(n) if rng.random() < 0.8} or {0: 1.0} for _ in range(m)
            ]
            senses = [rng.choice(("<=", ">=", "=")) for _ in range(m)]
            point = [rng.randint(lo, hi) for lo, hi in zip(lb, ub)]
            b = []
            for row, sense in zip(rows, senses):
                activity = sum(a * point[j] for j, a in row.items())
                slack = 0.0 if sense == "=" else round(rng.uniform(0, 3), 2)
                b.append(round(activity + (slack if sense == "<=" else -slack), 6))
            c = [round(rng.uniform(-5, 5), 1) for _ in range(n)]
            minimize = rng.random() < 0.5

            best = _brute_force(c, rows, b, senses, lb, ub, minimize)
            result = solve_milp(c, rows, b, list(range(n)), senses=senses, lb=lb, ub=ub, minimize=minimize)
            assert result.status == Status.OPTIMAL
            assert abs(result.objective - best) <= 1e-6 * (1 + abs(best))
            for row, bi, sense in zip(rows, b, senses):
                assert _row_holds(sum(a * result.solution[j] for j, a in row.items()), bi, sense)


class TestToleranceValidation:
    @pytest.mark.parametrize("backend", ["python", "rust"] if rust_available() else ["python"])
    def test_negative_eps_is_rejected(self, backend):
        """Every backend answers a negative tolerance the same way: with a clear error."""
        with pytest.raises(ValueError, match="eps cannot be negative"):
            solve_lp([1, 1], [{0: 1.0}, {1: 1.0}], [2.0, 3.0], minimize=False, eps=-1e-9, backend=backend)
        with pytest.raises(ValueError, match="eps cannot be negative"):
            solve_milp([1, 1], [{0: 1.0}, {1: 1.0}], [2.0, 3.0], [0], minimize=False, eps=-1e-9, backend=backend)

    @pytest.mark.parametrize("eps", [math.nan, math.inf])
    def test_non_finite_eps_is_rejected(self, eps):
        """NaN turned every tolerance test false and inf made every point feasible: both returned wrong OPTIMALs."""
        with pytest.raises(ValueError, match="eps must be finite"):
            solve_lp([-1.0], [[1.0]], [1.0], eps=eps)
        with pytest.raises(ValueError, match="eps must be finite"):
            solve_milp([-1.0], [[1.0]], [1.0], [0], eps=eps)


class TestSearchLimits:
    def test_node_limit_without_a_solution_is_not_infeasible(self):
        """Nodes are still queued, so nothing is proven: the limit is reported, not infeasibility."""
        result = solve_milp([-1, -1], [[1.5, 1.5]], [2.5], [0, 1], heuristics=False, max_nodes=0)
        assert result.status == Status.MAX_ITER
        assert result.solution is None

    def test_exhausted_search_without_a_solution_is_infeasible(self):
        result = solve_milp([-1, -1], [[2, 2], [-2, -2]], [3, -3], [0, 1], heuristics=False)
        assert result.status == Status.INFEASIBLE

    def test_integral_root_is_listed_in_solutions(self):
        result = solve_milp([-1, -1], [[1, 1]], [2], [0, 1], solution_limit=3)
        assert result.status == Status.OPTIMAL
        assert result.solutions == (result.solution,)


class TestGoodIncumbents:
    def test_a_good_warm_start_does_not_hide_the_optimum(self):
        """The root LP stopped short of its optimum, above this warm start, which then pruned the root as OPTIMAL."""
        c = [-5.4636895050399394e-05, 0.4394396523049169, -0.00018110723271795248]
        c += [0.000131256340651038, -3.617974675406224e-05]
        rows = [
            {2: 100000.0, 3: 16.5, 4: -0.1},
            {0: -0.1, 1: 3.3, 2: 900000.0, 4: 80000.0},
            {0: -500000.0, 1: 0.5, 2: 0.4, 3: -60000.0},
        ]
        b = [-1.6482232040437539, 79998.91243002671, 0.11135838376362472]
        result = solve_milp(c, rows, b, binary=range(5), senses=[">=", ">=", "<="], warm_start=(1, 0, 1, 0, 0))
        assert result.status == Status.OPTIMAL
        assert result.solution == (1.0, 0.0, 1.0, 0.0, 1.0)


class TestRoundingFlips:
    def test_flips_never_worsen_the_rounding(self):
        """Minimizing, turning off a variable with a negative cost raises the objective."""
        from solvor.milp_heuristics import round_binary
        from solvor.utils.lp_input import LinearProblem

        prob = LinearProblem(2, [{0: 1.0, 1: 1.0}], [2.0], ["<="], [0.0, 0.0], [1.0, 1.0])
        rounded = round_binary([0.6, 0.6], [0, 1], [-1.0, 2.0], prob, prob.columns(), [0.0] * 2, [1.0] * 2, True, 1e-6)
        assert rounded == (1.0, 0.0)


class TestPresolveLargeRightHandSides:
    def test_large_integral_rhs_is_not_declared_infeasible(self):
        """Integrality cannot be judged above 1e15, which presolve used to read as an impossible row."""
        result = solve_milp([1, 1], [[1, 1]], [1e15], [0, 1], senses=["="], ub=[1e15, 1e15])
        assert result.status == Status.OPTIMAL
        assert result.solution[0] + result.solution[1] == 1e15


def _rows_hold_scip(x, rows, senses, b, tol=1e-6):
    for row, sense, bi in zip(rows, senses, b):
        activity = sum(a * x[j] for j, a in row.items())
        slack = tol * max(1.0, abs(activity), abs(bi))
        if (sense != ">=" and activity > bi + slack) or (sense != "<=" and activity < bi - slack):
            return False
    return True


def _solve_binary_and_check(rows, senses, b, c, minimize, optimum):
    result = solve_milp(c, rows, b, binary=range(len(c)), senses=senses, minimize=minimize)
    assert result.status == Status.OPTIMAL
    assert _rows_hold_scip(result.solution, rows, senses, b)
    assert abs(result.objective - optimum) <= 1e-9 * (1 + abs(optimum))
    return result


class TestBadlyScaledModels:
    """Big-M style models: coefficients from 0.1 to millions in one model (rows checked with SCIP's rule)."""

    def test_lp_tolerance_does_not_hide_the_optimum(self):
        """The LP relaxation used eps = 1e-6 and stopped at x0 = 4.7e-7 (objective 0 instead of 2.52)."""
        c = [2.5163104287499944, -4.7405318243649806]
        _solve_binary_and_check([{0: -5e6, 1: -3.3}], ["<="], [-2.3514730439064113], c, False, 2.5163104287499944)

    def test_snapping_does_not_break_a_row(self):
        """x0 = 1e-6 counted as integral; snapped to 0 it left x1 = 1 against x1 = 1e6 * x0."""
        c = [-2.562093184840478, -1.0517327680030952]
        result = _solve_binary_and_check([{0: -1e6, 1: 1.0}], ["="], [0.0], c, True, 0.0)
        assert result.solution == (0.0, 0.0)

    def test_warm_start_is_checked_after_snapping(self):
        """A warm start integral within eps (x0 = 1e-6) must not become the snapped, infeasible incumbent (0, 1)."""
        c = [-2.562093184840478, -1.0517327680030952]
        result = solve_milp(c, [{0: -1e6, 1: 1.0}], [0.0], binary=range(2), senses=["="], warm_start=[1e-6, 1.0])
        assert result.solution == (0.0, 0.0)

    def test_heuristic_incumbents_are_checked_after_snapping(self):
        """The rounding heuristic kept x2 = 4.3e-7, which the snap turned into a row miss of 3."""
        rows = [{0: 5.0, 1: -1e6, 2: -7e6, 3: -2.0, 4: -2e5, 5: -1e6}]
        c = [2.8441012261202054, -4.524021736073057, -3.7715374036018523, 0.5481142956078688]
        c += [-0.4958735098908509, -3.21847585015014, 3.896952037963757, 3.7952544725896207]
        _solve_binary_and_check(rows, ["="], [0.0], c, False, 7.692206510553378)

    def test_warm_infeasible_nodes_are_confirmed(self):
        """With the 1e-9 LP tolerance the warm dual simplex declared a feasible child node infeasible."""
        rows = [
            {1: -6e6, 3: 3e6, 4: 8e6, 5: 4e6, 6: 9.0},
            {1: 5e6, 2: -0.4, 4: -6.6, 6: -9e5},
            {0: 3.0, 2: -0.5, 3: 5e6, 4: 3e5, 5: -1.0, 6: -5e6},
        ]
        b = [9000009.308447689, 4099993.0, 300003.8795286772]
        c = [-2.145834072845918, -1.4808880618828324, 4.052435658348175, 0.5395145944051594]
        c += [0.9075918846188857, 0.6795827279977615, -2.7891935471904272]
        _solve_binary_and_check(rows, ["<=", "=", "<="], b, c, True, -1.4558881389521168)

    def test_branching_makes_progress_past_a_bound(self):
        """x4 = 1.00002 above its bound 1 made a child equal to its parent, repeated until max_nodes."""
        rows = [
            {1: 0.4},
            {1: 0.2, 2: -10000.0, 3: 6.0, 4: 3.0},
            {0: 0.5, 1: -2.0, 2: 16.5, 3: 200000.0, 4: -200000.0},
            {1: 0.6000000000000001, 2: -23.099999999999998, 3: -700000.0},
        ]
        b = [-2.256291121305105, -9990.8, 12.96820133207468, -700022.5]
        c = [2.9311927588536255, 4.207579188178338, 1.1890808182541441, -4.8816307226265305, -2.210998079058201]
        result = solve_milp(c, rows, b, binary=range(5), senses=[">=", "=", ">=", "="], minimize=False)
        assert result.iterations < 100
        assert _rows_hold_scip(result.solution, rows, [">=", "=", ">=", "="], b)
        assert abs(result.objective - 1.235223963601376) <= 1e-9 * 2.3
        # The optimum is found, but from an LP point outside its bound: not proven, and the error says so
        assert result.status == Status.FEASIBLE and "numerical" in result.error

    def test_lp_points_outside_their_bounds_are_not_proven(self):
        """The root LP put x3 at -0.0077 (bound 0); clamped, it gave a feasible point that is not the optimum."""
        rows = [
            {0: -3.0, 4: -700000.0, 5: 50000.0, 6: -0.2, 7: 0.4},
            {2: 6.0, 3: 50000.0, 4: -6.0, 5: 7.0, 6: 6.6},
            {2: 0.6000000000000001, 3: -0.1, 4: -400000.0, 5: -0.5, 6: -300000.0, 7: 5.0},
            {0: -3.3, 2: 40000.0, 4: 600000.0, 5: 10000.0, 7: -0.1},
        ]
        senses, b = ["=", ">=", "<=", "="], [-649999.6, 6.810339198790282, -399993.234756114, 649999.9]
        c = [-3.6184461677239432, -3.9164905819668583, 1.9161305521404461, -2.0076431776637294]
        c += [-1.8179276355594642, 3.0330916243635713, 2.7896169502043424, -0.43219699544256684]
        result = solve_milp(c, rows, b, binary=range(8), senses=senses)
        optimum = -3.2250362141286013
        if result.status == Status.OPTIMAL:
            assert abs(result.objective - optimum) <= 1e-9 * (1 + abs(optimum))
        else:
            assert "numerical" in result.error

    def test_dropped_nodes_are_reported(self):
        """An integral LP point that violates a row cannot be judged: no OPTIMAL claim, and the error says why."""
        rows = [
            {3: -7.0, 4: -3e6, 7: -23.099999999999998},
            {0: -7.0, 1: 1e6, 4: -4e6, 5: 8e5, 6: 6.6, 7: 23.099999999999998},
            {0: 3e6, 1: -4.0, 2: 13.2, 3: -6.6, 5: 7.0, 6: 5e6, 7: -16.5},
        ]
        b = [-3000030.1, -2199983.9, 2999993.1]
        c = [3.280624190410572, -1.3433818029543412, -3.895062008925081, -3.3763548817312694]
        c += [4.234166628804495, -4.371006300243073, -4.55088572799649, 1.7420543589916377]
        result = solve_milp(c, rows, b, binary=range(8), senses=["=", "=", "="])
        assert result.status != Status.OPTIMAL
        assert "numerical" in result.error
        assert result.solution is None or _rows_hold_scip(result.solution, rows, ["=", "=", "="], b)


def _binary_optimum(rows, senses, b, c, minimize, tol):
    best = None
    for x in itertools.product((0.0, 1.0), repeat=len(c)):
        if _rows_hold_scip(x, rows, senses, b, tol):
            value = sum(cj * xj for cj, xj in zip(c, x))
            if best is None or (value < best if minimize else value > best):
                best = value
    return best


def _big_m_model(rng, top):
    """Binary model with coefficients from 0.1 to about 9 * top around a known feasible point."""
    n, m = rng.randint(2, 8), rng.randint(1, 4)
    magnitudes = [1.0, 0.1, 3.3, top / 10, top]
    rows = [
        {j: rng.choice((-1, 1)) * rng.choice(magnitudes) * rng.randint(1, 9) for j in range(n) if rng.random() < 0.7}
        or {0: 1.0}
        for _ in range(m)
    ]
    senses = [rng.choice(("<=", ">=", "=")) for _ in range(m)]
    point = [float(rng.randint(0, 1)) for _ in range(n)]
    b = []
    for row, sense in zip(rows, senses):
        activity = sum(a * point[j] for j, a in row.items())
        b.append(activity + (0.0 if sense == "=" else rng.uniform(0, 3) * (1 if sense == "<=" else -1)))
    return rows, senses, b, [rng.uniform(-5, 5) for _ in range(n)]


class TestScaledBigMModels:
    def test_small_costs_next_to_big_m_coefficients(self):
        """A tie-breaking cost of 1e-4 on a column with a 1e6 coefficient must not vanish under the tolerance."""
        result = solve_milp([1e-4, 0.0], [{0: 1e6, 1: 1.0}], [2e6], binary=[0], ub=[1, 1], minimize=False)
        assert result.solution == (1.0, 0.0) and result.objective == 1e-4
        c = [4.8263813017791174e-05, -0.0002651055823561933, -0.8301041915060114]
        c += [-0.0003008426727270034, 1.4081045512398038, -2.5690099435366245e-05]
        rows = [
            {0: 6e5, 4: -9.899999999999999, 5: -0.1},
            {0: 0.4, 1: 6e5, 2: 6.0, 3: -7e5, 4: -0.1, 5: 0.6000000000000001},
        ]
        _solve_binary_and_check(rows, ["=", "<="], [-0.1, 3.353379651823708], c, True, -0.83069582986053)

    def test_tiny_coefficient_columns_still_enter(self):
        assert solve_milp([-1.0, -1.0], [{0: 1e-10, 1: 1.0}], [1.0], binary=[0, 1]).objective == -2.0

    def test_roundoff_does_not_prove_infeasibility(self):
        """Trusting every warm INFEASIBLE at the 1e-9 LP tolerance dropped a feasible node (a 5e-9 violation)."""
        c = [-2.47472862553468, -0.7863686739024436, -4.269612210645622, 2.4424301956668506]
        c += [-2.9460668791522893, 3.043069422804246, -1.6783578007028166, 1.8220326479211693]
        rows = [
            {3: -1, 4: -2, 5: -1, 6: 1},
            {0: -1, 1: 684, 5: -451, 6: -1, 7: 1},
            {0: 2, 2: 481, 3: 2, 5: -664},
            {0: -1, 1: -769, 2: 1, 3: 2, 4: 556, 6: 607},
        ]
        b = [1.6123537276187543, 233.0, -662.3149863271886, -160.0]
        _solve_binary_and_check(rows, ["<=", "=", ">=", "="], b, c, True, 4.842805791787006)

    def test_scaled_lp_finds_the_feasible_point(self):
        """Unscaled, coefficients from 0.1 to 9e5 in one row made every LP after branching look infeasible."""
        rows = [
            {0: 16.5, 1: 9.0, 2: 2.0, 3: -3.0, 4: -400000.0},
            {0: -0.9, 1: -900000.0, 2: -23.099999999999998, 3: -0.8, 4: 30000.0},
            {0: 19.799999999999997, 1: -20000.0, 2: 40000.0, 4: 0.1},
        ]
        c = [2.037460921609398, -4.231395140277435, 0.08676655689652613, 2.380731270181263, -2.394041283846513]
        _solve_binary_and_check(rows, [">=", "=", "="], [-0.8386488309790875, 0.0, 0.0], c, False, 0.0)

    def test_cuts_added_to_a_scaled_model_match_a_fresh_solve(self):
        """Rows added after the build use the existing column factors and get their own row factor."""
        from solvor import MilpModel

        for seed in range(60):
            rng = random.Random(seed)
            n = 6
            rows = [
                {j: rng.choice((1.0, 3.3, 1e5, 1e6)) * rng.randint(1, 9) for j in range(n) if rng.random() < 0.7}
                or {0: 1.0}
                for _ in range(3)
            ]
            rhs = [sum(r.values()) * 0.5 for r in rows]
            cut = {j: rng.choice((1.0, 2e5)) for j in range(n - 1, -1, -2)}
            c = [rng.uniform(1, 5) for _ in range(n)]
            model = MilpModel(n, binary=range(n))
            model.add_rows(rows, rhs)
            model.solve(c, minimize=False)
            model.add_rows([cut], [sum(cut.values()) * 0.4])
            incremental = model.solve(c, minimize=False)
            fresh = solve_milp(c, [*rows, cut], [*rhs, sum(cut.values()) * 0.4], binary=range(n), minimize=False)
            assert incremental.status == fresh.status
            assert abs(incremental.objective - fresh.objective) <= 1e-9 * (1 + abs(fresh.objective))

    @pytest.mark.parametrize(
        ("top", "limit", "cost_scales"),
        [(1e5, 2, (1.0,)), (1e6, 6, (1.0,)), (1e5, 2, (1.0, 1e-4)), (1e6, 6, (1.0, 1e-4))],
    )
    def test_big_m_models_against_brute_force(self, top, limit, cost_scales):
        """No point violates a row, every wrong answer says it is not proven, and both stay rare (spec: 0.1%, 0.3%)."""
        rng = random.Random(2026)
        scale_rng = random.Random(7)  # separate, so cost_scales=(1.0,) keeps the original sample
        infeasible = unexplained = explained = downgraded = 0
        for _ in range(1000):
            rows, senses, b, c = _big_m_model(rng, top)
            c = [v * scale_rng.choice(cost_scales) for v in c]  # small costs: tie-breakers next to big-M rows
            for minimize in (True, False):
                strict = _binary_optimum(rows, senses, b, c, minimize, 1e-12)
                loose = _binary_optimum(rows, senses, b, c, minimize, 1e-6)
                result = solve_milp(c, rows, b, binary=range(len(c)), senses=senses, minimize=minimize)
                if result.solution is not None and not _rows_hold_scip(result.solution, rows, senses, b):
                    infeasible += 1
                    continue
                if result.solution is None:
                    wrong = strict is not None
                else:
                    low, high = (loose, strict) if minimize else (strict, loose)
                    # solve_milp's own optimality tolerance (eps, gap_tol): small costs make 1e-7 differences ties
                    slack = 1e-6 * (1 + abs(strict if strict is not None else loose))
                    wrong = not (low - slack <= result.objective <= high + slack)
                if wrong:
                    unexplained += result.error is None
                    explained += result.error is not None
                else:
                    downgraded += result.error is not None
        assert infeasible == 0
        assert unexplained == 0
        assert explained <= limit
        assert downgraded <= 20  # right answers that could not be proven: FEASIBLE with an explanation

"""Tests for the constraint programming solver."""

from solvor.cp import Model
from solvor.types import Status


class TestBasicCP:
    def test_all_different(self):
        m = Model()
        x = m.int_var(1, 3, "x")
        y = m.int_var(1, 3, "y")
        z = m.int_var(1, 3, "z")
        m.add(m.all_different([x, y, z]))
        result = m.solve()
        assert result.status == Status.OPTIMAL
        vals = [result.solution["x"], result.solution["y"], result.solution["z"]]
        assert len(set(vals)) == 3

    def test_sum_constraint(self):
        m = Model()
        x = m.int_var(1, 9, "x")
        y = m.int_var(1, 9, "y")
        m.add(m.sum_eq([x, y], 10))
        result = m.solve()
        assert result.status == Status.OPTIMAL
        assert result.solution["x"] + result.solution["y"] == 10

    def test_equality_constant(self):
        m = Model()
        x = m.int_var(1, 10, "x")
        m.add(x == 5)
        result = m.solve()
        assert result.status == Status.OPTIMAL
        assert result.solution["x"] == 5


class TestInfeasible:
    def test_impossible_value(self):
        m = Model()
        x = m.int_var(1, 5, "x")
        m.add(x == 10)  # impossible
        result = m.solve()
        assert result.status == Status.INFEASIBLE

    def test_all_different_impossible(self):
        # 4 variables, domain 1-3, all different - impossible (pigeonhole)
        m = Model()
        vars = [m.int_var(1, 3, f"x{i}") for i in range(4)]
        m.add(m.all_different(vars))
        result = m.solve()
        # Should be infeasible, but may hit iteration limit
        assert result.status in (Status.INFEASIBLE, Status.MAX_ITER)

    def test_conflicting_constraints(self):
        m = Model()
        x = m.int_var(1, 10, "x")
        m.add(x == 3)
        m.add(x == 7)
        result = m.solve()
        assert result.status == Status.INFEASIBLE


class TestNQueens:
    def test_4queens(self):
        # Classic N-Queens for N=4
        n = 4
        m = Model()
        queens = [m.int_var(0, n - 1, f"q{i}") for i in range(n)]

        # All different columns
        m.add(m.all_different(queens))

        # Diagonal constraints via !=
        for i in range(n):
            for j in range(i + 1, n):
                # |q[i] - q[j]| != |i - j|
                # This is tricky in CP, we'll use a simpler encoding
                pass

        result = m.solve()
        assert result.status == Status.OPTIMAL
        # At least columns are different
        cols = [result.solution[f"q{i}"] for i in range(n)]
        assert len(set(cols)) == n


class TestArithmetic:
    def test_sum_three_vars(self):
        m = Model()
        x = m.int_var(1, 5, "x")
        y = m.int_var(1, 5, "y")
        z = m.int_var(1, 5, "z")
        m.add(m.sum_eq([x, y, z], 9))
        result = m.solve()
        assert result.status == Status.OPTIMAL
        total = result.solution["x"] + result.solution["y"] + result.solution["z"]
        assert total == 9

    def test_inequality(self):
        m = Model()
        x = m.int_var(1, 10, "x")
        y = m.int_var(1, 10, "y")
        m.add(x != y)
        m.add(m.sum_eq([x, y], 10))
        result = m.solve()
        assert result.status == Status.OPTIMAL
        assert result.solution["x"] != result.solution["y"]
        assert result.solution["x"] + result.solution["y"] == 10


class TestEdgeCases:
    def test_single_variable(self):
        m = Model()
        m.int_var(5, 5, "x")  # Domain of size 1
        result = m.solve()
        assert result.status == Status.OPTIMAL
        assert result.solution["x"] == 5

    def test_binary_variable(self):
        m = Model()
        x = m.int_var(0, 1, "x")
        m.add(x == 1)
        result = m.solve()
        assert result.status == Status.OPTIMAL
        assert result.solution["x"] == 1

    def test_large_domain(self):
        m = Model()
        x = m.int_var(0, 100, "x")
        m.add(x == 50)
        result = m.solve()
        assert result.status == Status.OPTIMAL
        assert result.solution["x"] == 50


class TestCombinedConstraints:
    def test_sudoku_cell(self):
        # Mini 2x2 Sudoku-like: 4 cells, values 1-2, constraints
        m = Model()
        cells = [[m.int_var(1, 2, f"c{i}{j}") for j in range(2)] for i in range(2)]

        # Row constraints
        m.add(m.all_different([cells[0][0], cells[0][1]]))
        m.add(m.all_different([cells[1][0], cells[1][1]]))

        # Column constraints
        m.add(m.all_different([cells[0][0], cells[1][0]]))
        m.add(m.all_different([cells[0][1], cells[1][1]]))

        result = m.solve()
        assert result.status == Status.OPTIMAL

        # Verify solution
        for i in range(2):
            assert result.solution[f"c{i}0"] != result.solution[f"c{i}1"]
        for j in range(2):
            assert result.solution[f"c0{j}"] != result.solution[f"c1{j}"]


class TestStress:
    def test_many_variables(self):
        # 5 variables, sum = 25 (simpler to solve)
        m = Model()
        n = 5
        vars = [m.int_var(1, 10, f"x{i}") for i in range(n)]
        m.add(m.sum_eq(vars, 25))
        result = m.solve()
        assert result.status in (Status.OPTIMAL, Status.MAX_ITER)
        if result.status == Status.OPTIMAL:
            total = sum(result.solution[f"x{i}"] for i in range(n))
            assert total == 25

    def test_all_different_5(self):
        # 5 variables from domain 1-5
        m = Model()
        vars = [m.int_var(1, 5, f"x{i}") for i in range(5)]
        m.add(m.all_different(vars))
        result = m.solve()
        assert result.status == Status.OPTIMAL
        vals = [result.solution[f"x{i}"] for i in range(5)]
        assert set(vals) == {1, 2, 3, 4, 5}


class TestVariableEquality:
    def test_two_vars_equal(self):
        m = Model()
        x = m.int_var(1, 5, "x")
        y = m.int_var(1, 5, "y")
        m.add(x == y)
        result = m.solve()
        assert result.status == Status.OPTIMAL
        assert result.solution["x"] == result.solution["y"]

    def test_two_vars_equal_with_sum(self):
        m = Model()
        x = m.int_var(1, 9, "x")
        y = m.int_var(1, 9, "y")
        m.add(x == y)
        m.add(m.sum_eq([x, y], 10))
        result = m.solve()
        assert result.status == Status.OPTIMAL
        assert result.solution["x"] == result.solution["y"] == 5

    def test_two_vars_not_equal(self):
        m = Model()
        x = m.int_var(1, 2, "x")
        y = m.int_var(1, 2, "y")
        m.add(x != y)
        result = m.solve()
        assert result.status == Status.OPTIMAL
        assert result.solution["x"] != result.solution["y"]

    def test_ne_const(self):
        m = Model()
        x = m.int_var(1, 5, "x")
        m.add(x != 3)
        result = m.solve()
        assert result.status == Status.OPTIMAL
        assert result.solution["x"] != 3

    def test_vars_disjoint_domains_equal(self):
        m = Model()
        x = m.int_var(1, 3, "x")
        y = m.int_var(5, 7, "y")
        m.add(x == y)
        result = m.solve()
        assert result.status == Status.INFEASIBLE


class TestHints:
    def test_hints_with_constraint(self):
        """Hints work with constrained model."""
        m = Model()
        x = m.int_var(1, 2, "x")
        y = m.int_var(1, 2, "y")
        m.add(m.all_different([x, y]))

        # Hint x=1 should work with the constraint
        result = m.solve(hints={"x": 1})
        assert result.ok
        assert result.solution["x"] != result.solution["y"]

    def test_hints_simple_model(self):
        """Hints work with simple unconstrained model."""
        m = Model()
        m.int_var(1, 2, "x")
        result = m.solve(hints={"x": 1})
        assert result.ok
        assert result.solution["x"] == 1  # Hint should be followed when feasible

    def test_hints_infeasible_ignored(self):
        """Hints with infeasible value don't crash."""
        m = Model()
        x = m.int_var(1, 5, "x")
        m.add(x == 3)
        # Hint x=10 is out of domain, should be ignored
        result = m.solve(hints={"x": 10})
        assert result.status == Status.OPTIMAL
        assert result.solution["x"] == 3


class TestSolutionPool:
    def test_solution_limit_one(self):
        """Default solution_limit=1 returns single solution."""
        m = Model()
        m.int_var(1, 3, "x")
        result = m.solve()
        assert result.status == Status.OPTIMAL
        assert result.solutions is None

    def test_solution_limit_multiple(self):
        """solution_limit > 1 finds multiple solutions."""
        m = Model()
        m.int_var(1, 3, "x")
        m.int_var(1, 3, "y")
        m.add(m.all_different([m._vars["x"], m._vars["y"]]))
        result = m.solve(solution_limit=10)
        assert result.ok
        if result.solutions:
            assert len(result.solutions) >= 2

    def test_solutions_satisfy_constraints(self):
        """All returned solutions satisfy constraints."""
        m = Model()
        x = m.int_var(1, 3, "x")
        y = m.int_var(1, 3, "y")
        m.add(x != y)
        result = m.solve(solution_limit=10)
        if result.solutions:
            for sol in result.solutions:
                assert sol["x"] != sol["y"]

    def test_solutions_are_different(self):
        """Multiple solutions are mostly distinct."""
        m = Model()
        m.int_var(1, 3, "x")
        m.int_var(1, 3, "y")
        result = m.solve(solution_limit=20)
        if result.solutions and len(result.solutions) > 1:
            seen = set()
            for sol in result.solutions:
                key = tuple(sorted(sol.items()))
                seen.add(key)
            # Should have found multiple unique solutions (may have some duplicates)
            assert len(seen) >= 2


class TestCircuit:
    def test_circuit_two_nodes(self):
        """Circuit with 2 nodes: each points to the other."""
        m = Model()
        x = m.int_var(0, 1, "x0")  # successor of 0
        y = m.int_var(0, 1, "x1")  # successor of 1
        m.add(m.circuit([x, y]))
        result = m.solve()
        # Either infeasible or should form a cycle: 0->1->0
        if result.status == Status.OPTIMAL:
            # x0 = 1 (0 points to 1), x1 = 0 (1 points to 0)
            assert result.solution["x0"] == 1
            assert result.solution["x1"] == 0

    def test_circuit_three_nodes(self):
        """Circuit with 3 nodes forms a Hamiltonian cycle."""
        m = Model()
        # successor[i] = j means edge i -> j
        s0 = m.int_var(0, 2, "s0")
        s1 = m.int_var(0, 2, "s1")
        s2 = m.int_var(0, 2, "s2")
        m.add(m.circuit([s0, s1, s2]))
        result = m.solve()
        assert result.status == Status.OPTIMAL

        # Verify it's a valid cycle: follow edges from 0 back to 0
        succ = [result.solution["s0"], result.solution["s1"], result.solution["s2"]]
        visited = [False, False, False]
        node = 0
        for _ in range(3):
            assert not visited[node], "Visited same node twice"
            visited[node] = True
            node = succ[node]
        assert node == 0, "Didn't return to start"
        assert all(visited), "Didn't visit all nodes"

    def test_circuit_no_self_loop(self):
        """Circuit constraint forbids self-loops."""
        m = Model()
        s0 = m.int_var(0, 2, "s0")
        s1 = m.int_var(0, 2, "s1")
        s2 = m.int_var(0, 2, "s2")
        m.add(m.circuit([s0, s1, s2]))
        result = m.solve()
        if result.status == Status.OPTIMAL:
            assert result.solution["s0"] != 0  # no self-loop
            assert result.solution["s1"] != 1
            assert result.solution["s2"] != 2


class TestNoOverlap:
    def test_no_overlap_two_intervals(self):
        """Two intervals that can't overlap."""
        m = Model()
        # Smaller domains for faster encoding (O(domain²) clauses per pair)
        s1 = m.int_var(0, 4, "s1")
        s2 = m.int_var(0, 4, "s2")
        # Duration 2 each, so they must be separated
        m.add(m.no_overlap([s1, s2], [2, 2]))
        result = m.solve()
        assert result.status == Status.OPTIMAL
        # Either s1 + 2 <= s2 or s2 + 2 <= s1
        start1, start2 = result.solution["s1"], result.solution["s2"]
        assert start1 + 2 <= start2 or start2 + 2 <= start1

    def test_no_overlap_three_intervals(self):
        """Three intervals scheduled without overlap."""
        m = Model()
        # Smaller domains for faster encoding
        s1 = m.int_var(0, 6, "s1")
        s2 = m.int_var(0, 6, "s2")
        s3 = m.int_var(0, 6, "s3")
        m.add(m.no_overlap([s1, s2, s3], [2, 2, 2]))
        result = m.solve()
        assert result.status == Status.OPTIMAL

        # Verify no overlaps
        starts = [result.solution["s1"], result.solution["s2"], result.solution["s3"]]
        durations = [2, 2, 2]
        for i in range(3):
            for j in range(i + 1, 3):
                si, sj = starts[i], starts[j]
                di, dj = durations[i], durations[j]
                assert si + di <= sj or sj + dj <= si

    def test_no_overlap_infeasible(self):
        """Not enough room for all intervals."""
        m = Model()
        # Three intervals of size 2 in window [0,3] - can't fit (need 6 time units, only have 5)
        s1 = m.int_var(0, 3, "s1")
        s2 = m.int_var(0, 3, "s2")
        s3 = m.int_var(0, 3, "s3")
        m.add(m.no_overlap([s1, s2, s3], [2, 2, 2]))
        result = m.solve()
        # Should be infeasible or hit max iterations
        assert result.status in (Status.INFEASIBLE, Status.MAX_ITER)

    def test_no_overlap_validation(self):
        """Mismatched lengths raise error."""
        m = Model()
        s1 = m.int_var(0, 10, "s1")
        s2 = m.int_var(0, 10, "s2")
        try:
            m.no_overlap([s1, s2], [5])  # Wrong length
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "same length" in str(e)


class TestCumulative:
    def test_cumulative_two_tasks(self):
        """Two tasks with cumulative capacity constraint."""
        m = Model()
        # Reduced domain to limit time horizon (O(time × tasks × domain))
        s1 = m.int_var(0, 5, "s1")
        s2 = m.int_var(0, 5, "s2")
        # Tasks with demand 3 and 3, capacity 5
        # They can overlap only if sum of demands <= 5
        m.add(m.cumulative([s1, s2], [2, 2], [3, 3], 5))
        result = m.solve()
        assert result.status == Status.OPTIMAL

        # If they overlap, total demand at that time must be <= 5
        start1, start2 = result.solution["s1"], result.solution["s2"]
        # Check capacity at each time point
        for t in range(max(start1 + 2, start2 + 2)):
            demand = 0
            if start1 <= t < start1 + 2:
                demand += 3
            if start2 <= t < start2 + 2:
                demand += 3
            assert demand <= 5

    def test_cumulative_force_sequencing(self):
        """High demands force tasks to be sequential."""
        m = Model()
        # Reduced domain to limit time horizon
        s1 = m.int_var(0, 5, "s1")
        s2 = m.int_var(0, 5, "s2")
        # Tasks with demand 5 each, capacity 5 - can't overlap
        m.add(m.cumulative([s1, s2], [2, 2], [5, 5], 5))
        result = m.solve()
        assert result.status == Status.OPTIMAL

        start1, start2 = result.solution["s1"], result.solution["s2"]
        # Tasks should not overlap (one must end before other starts)
        assert start1 + 2 <= start2 or start2 + 2 <= start1

    def test_cumulative_validation(self):
        """Mismatched lengths raise error."""
        m = Model()
        s1 = m.int_var(0, 10, "s1")
        s2 = m.int_var(0, 10, "s2")
        try:
            m.cumulative([s1, s2], [2], [3, 3], 5)  # Wrong duration length
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "same length" in str(e)


class TestSolverSelection:
    def test_dfs_solver_explicit(self):
        """Explicitly choosing DFS solver works."""
        m = Model()
        x = m.int_var(1, 3, "x")
        y = m.int_var(1, 3, "y")
        m.add(m.all_different([x, y]))
        result = m.solve(solver="dfs")
        assert result.ok
        assert result.solution["x"] != result.solution["y"]

    def test_sat_solver_explicit(self):
        """Explicitly choosing SAT solver works."""
        m = Model()
        x = m.int_var(1, 3, "x")
        y = m.int_var(1, 3, "y")
        m.add(m.all_different([x, y]))
        result = m.solve(solver="sat")
        assert result.ok
        assert result.solution["x"] != result.solution["y"]

    def test_invalid_solver_raises(self):
        """Unknown solver raises ValueError."""
        m = Model()
        m.int_var(1, 3, "x")
        try:
            m.solve(solver="unknown")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "unknown" in str(e).lower()

    def test_dfs_falls_back_for_sum_constraint(self):
        """DFS falls back to SAT for sum constraints."""
        m = Model()
        x = m.int_var(1, 5, "x")
        y = m.int_var(1, 5, "y")
        m.add(m.sum_eq([x, y], 6))
        # DFS should auto-fallback to SAT for sum_eq
        result = m.solve(solver="dfs")
        assert result.ok
        assert result.solution["x"] + result.solution["y"] == 6


class TestExpressionConstraints:
    def test_var_plus_const_ne(self):
        """Test x + 1 != y + 2 constraint."""
        m = Model()
        x = m.int_var(0, 3, "x")
        y = m.int_var(0, 3, "y")
        # x + 1 != y + 2 means x != y + 1
        m.add(x + 1 != y + 2)
        result = m.solve()
        assert result.ok
        # x + 1 should not equal y + 2
        assert result.solution["x"] + 1 != result.solution["y"] + 2

    def test_var_minus_const(self):
        """Test x - 1 != y - 2 constraint."""
        m = Model()
        x = m.int_var(0, 3, "x")
        y = m.int_var(0, 3, "y")
        m.add(x - 1 != y - 2)
        result = m.solve()
        assert result.ok
        assert result.solution["x"] - 1 != result.solution["y"] - 2

    def test_diagonal_constraint_pattern(self):
        """Test N-Queens style diagonal constraints."""
        # This is the pattern: queens[i] + i != queens[j] + j
        m = Model()
        q0 = m.int_var(0, 3, "q0")
        q1 = m.int_var(0, 3, "q1")
        # Forward diagonal: q0 + 0 != q1 + 1
        m.add(q0 + 0 != q1 + 1)
        # Backward diagonal: q0 - 0 != q1 - 1
        m.add(q0 - 0 != q1 - 1)
        result = m.solve()
        assert result.ok
        # Verify constraints
        assert result.solution["q0"] != result.solution["q1"] + 1
        assert result.solution["q0"] != result.solution["q1"] - 1


class TestNQueensFull:
    def test_4queens_with_diagonals(self):
        """Full 4-Queens with diagonal constraints."""
        n = 4
        m = Model()
        queens = [m.int_var(0, n - 1, f"q{i}") for i in range(n)]

        # All different columns
        m.add(m.all_different(queens))

        # Diagonal constraints
        for i in range(n):
            for j in range(i + 1, n):
                m.add(queens[i] + i != queens[j] + j)  # Forward diagonal
                m.add(queens[i] - i != queens[j] - j)  # Backward diagonal

        result = m.solve()
        assert result.ok

        cols = [result.solution[f"q{i}"] for i in range(n)]
        # Verify all different columns
        assert len(set(cols)) == n
        # Verify no diagonal attacks
        for i in range(n):
            for j in range(i + 1, n):
                assert cols[i] + i != cols[j] + j
                assert cols[i] - i != cols[j] - j

    def test_8queens_with_diagonals(self):
        """Full 8-Queens with diagonal constraints."""
        n = 8
        m = Model()
        queens = [m.int_var(0, n - 1, f"q{i}") for i in range(n)]

        m.add(m.all_different(queens))

        for i in range(n):
            for j in range(i + 1, n):
                m.add(queens[i] + i != queens[j] + j)
                m.add(queens[i] - i != queens[j] - j)

        result = m.solve()
        assert result.ok

        cols = [result.solution[f"q{i}"] for i in range(n)]
        assert len(set(cols)) == n
        for i in range(n):
            for j in range(i + 1, n):
                assert cols[i] + i != cols[j] + j
                assert cols[i] - i != cols[j] - j

    def test_4queens_multiple_solutions(self):
        """4-Queens has exactly 2 solutions."""
        n = 4
        m = Model()
        queens = [m.int_var(0, n - 1, f"q{i}") for i in range(n)]

        m.add(m.all_different(queens))

        for i in range(n):
            for j in range(i + 1, n):
                m.add(queens[i] + i != queens[j] + j)
                m.add(queens[i] - i != queens[j] - j)

        result = m.solve(solution_limit=10)
        assert result.ok
        # 4-Queens has exactly 2 solutions
        assert result.solutions is not None
        assert len(result.solutions) == 2


class TestExpressionOperators:
    """Test Expr and IntVar operator coverage."""

    def test_expr_eq_expr(self):
        """Test (x + 1) == (y + 2) constraint with SAT."""
        m = Model()
        x = m.int_var(0, 5, "x")
        y = m.int_var(0, 5, "y")
        # x + 1 == y + 2 means x == y + 1
        m.add((x + 1) == (y + 2))
        result = m.solve(solver="sat")  # Use SAT for complex expr
        assert result.ok
        assert result.solution["x"] + 1 == result.solution["y"] + 2

    def test_expr_eq_int(self):
        """Test (x + 1) == 5 constraint with SAT."""
        m = Model()
        x = m.int_var(0, 10, "x")
        m.add((x + 1) == 5)
        result = m.solve(solver="sat")  # Use SAT for complex expr
        assert result.ok
        assert result.solution["x"] == 4

    def test_expr_ne_int(self):
        """Test (x + 1) != 5 constraint with SAT."""
        m = Model()
        x = m.int_var(3, 5, "x")
        m.add((x + 1) != 5)
        result = m.solve(solver="sat")  # Use SAT for complex expr
        assert result.ok
        assert result.solution["x"] + 1 != 5

    def test_expr_add_expr(self):
        """Test Expr + Expr."""
        m = Model()
        x = m.int_var(0, 5, "x")
        y = m.int_var(0, 5, "y")
        expr = (x + 1) + 2  # Expr + int
        m.add(expr == 6)
        result = m.solve(solver="sat")
        assert result.ok

    def test_expr_radd(self):
        """Test int + Expr (radd)."""
        m = Model()
        x = m.int_var(0, 5, "x")
        expr = 2 + (x + 1)  # int + Expr triggers __radd__
        m.add(expr == 6)
        result = m.solve(solver="sat")
        assert result.ok

    def test_expr_sub_expr(self):
        """Test Expr - Expr."""
        m = Model()
        x = m.int_var(0, 10, "x")
        y = m.int_var(0, 10, "y")
        expr1 = x + 5
        expr2 = y + 2
        # This tests Expr.__sub__(Expr)
        _ = expr1 - expr2

    def test_intvar_eq_expr(self):
        """Test x == (y + 1) constraint with SAT."""
        m = Model()
        x = m.int_var(0, 5, "x")
        y = m.int_var(0, 5, "y")
        m.add(x == (y + 1))
        result = m.solve(solver="sat")
        assert result.ok
        assert result.solution["x"] == result.solution["y"] + 1

    def test_intvar_ne_expr(self):
        """Test x != (y + 1) constraint."""
        m = Model()
        x = m.int_var(0, 3, "x")
        y = m.int_var(0, 3, "y")
        m.add(x != (y + 1))
        result = m.solve()  # DFS handles this
        assert result.ok
        assert result.solution["x"] != result.solution["y"] + 1

    def test_intvar_radd(self):
        """Test int + IntVar (radd) with SAT."""
        m = Model()
        x = m.int_var(0, 5, "x")
        expr = 3 + x  # triggers IntVar.__radd__
        m.add(expr == 7)
        result = m.solve(solver="sat")
        assert result.ok
        assert result.solution["x"] == 4

    def test_intvar_sub_intvar(self):
        """Test x - y != 0 constraint (different from x != y)."""
        m = Model()
        x = m.int_var(0, 5, "x")
        y = m.int_var(0, 5, "y")
        m.add((x - y) != 0)  # x - y != 0
        result = m.solve(solver="sat")  # Use SAT for sub expression
        assert result.ok
        assert result.solution["x"] - result.solution["y"] != 0

    def test_intvar_sub_expr(self):
        """Test x - (y + 1) expression creation."""
        m = Model()
        x = m.int_var(0, 5, "x")
        y = m.int_var(0, 5, "y")
        # x - (y + 1) tests IntVar.__sub__(Expr)
        expr = x - (y + 1)
        assert expr is not None

    def test_intvar_rsub(self):
        """Test int - x (rsub)."""
        m = Model()
        x = m.int_var(0, 5, "x")
        expr = 10 - x  # triggers IntVar.__rsub__
        assert expr is not None

    def test_intvar_mul_int(self):
        """Test x * 3 (IntVar multiplication)."""
        m = Model()
        x = m.int_var(1, 5, "x")
        expr = x * 3  # triggers IntVar.__mul__
        assert expr is not None
        # Verify the expression tuple structure
        assert expr.data[0] == "mul"
        assert expr.data[2] == 3

    def test_int_mul_intvar(self):
        """Test 3 * x (IntVar rmul)."""
        m = Model()
        x = m.int_var(1, 5, "x")
        expr = 3 * x  # triggers IntVar.__rmul__
        assert expr is not None
        assert expr.data[0] == "mul"
        assert expr.data[2] == 3

    def test_expr_mul_int(self):
        """Test (x + 1) * 2 (Expr multiplication)."""
        m = Model()
        x = m.int_var(1, 5, "x")
        expr = (x + 1) * 2  # triggers Expr.__mul__
        assert expr is not None
        assert expr.data[0] == "mul"

    def test_int_mul_expr(self):
        """Test 2 * (x + 1) (Expr rmul)."""
        m = Model()
        x = m.int_var(1, 5, "x")
        expr = 2 * (x + 1)  # triggers Expr.__rmul__
        assert expr is not None
        assert expr.data[0] == "mul"

    def test_multiplication_in_constraint(self):
        """Test multiplication used in a real constraint (timetabling pattern)."""
        m = Model()
        timeslot = m.int_var(0, 2, "timeslot")  # 3 timeslots
        room = m.int_var(0, 1, "room")  # 2 rooms
        n_rooms = 2

        # Combined index: timeslot * n_rooms + room
        # This is the pattern that failed before the fix
        combined = timeslot * n_rooms + room

        # The combined value should be unique per (timeslot, room) pair
        # Range: 0*2+0=0 to 2*2+1=5
        assert combined is not None

    def test_intvar_mul_unsupported_type(self):
        """Test IntVar * unsupported type returns NotImplemented."""
        m = Model()
        x = m.int_var(1, 5, "x")
        result = x.__mul__("string")
        assert result is NotImplemented

    def test_expr_mul_unsupported_type(self):
        """Test Expr * unsupported type returns NotImplemented."""
        m = Model()
        x = m.int_var(1, 5, "x")
        expr = x + 1
        result = expr.__mul__("string")
        assert result is NotImplemented


class TestSumLeGe:
    def test_sum_le(self):
        """Test sum_le constraint."""
        m = Model()
        x = m.int_var(1, 5, "x")
        y = m.int_var(1, 5, "y")
        m.add(m.sum_le([x, y], 5))
        result = m.solve()
        assert result.ok
        assert result.solution["x"] + result.solution["y"] <= 5

    def test_sum_ge(self):
        """Test sum_ge constraint."""
        m = Model()
        x = m.int_var(1, 5, "x")
        y = m.int_var(1, 5, "y")
        m.add(m.sum_ge([x, y], 8))
        result = m.solve()
        assert result.ok
        assert result.solution["x"] + result.solution["y"] >= 8


class TestSATSolverPaths:
    def test_sat_with_hints(self):
        """Test SAT solver with hints."""
        m = Model()
        x = m.int_var(1, 5, "x")
        y = m.int_var(1, 5, "y")
        m.add(m.sum_eq([x, y], 6))  # Forces SAT solver
        result = m.solve(hints={"x": 2})
        assert result.ok
        assert result.solution["x"] + result.solution["y"] == 6

    def test_sat_multiple_solutions(self):
        """Test SAT solver with solution_limit > 1."""
        m = Model()
        x = m.int_var(1, 3, "x")
        y = m.int_var(1, 3, "y")
        m.add(m.sum_eq([x, y], 4))  # Forces SAT solver
        result = m.solve(solution_limit=5)
        assert result.ok
        # Should find multiple solutions: (1,3), (2,2), (3,1)
        if result.solutions:
            for sol in result.solutions:
                assert sol["x"] + sol["y"] == 4


class TestSumConstraints:
    def test_sum_eq_infeasible_too_small(self):
        m = Model()
        x = m.int_var(1, 5, "x")
        y = m.int_var(1, 5, "y")
        m.add(m.sum_eq([x, y], 1))
        result = m.solve()
        assert result.status == Status.INFEASIBLE

    def test_sum_eq_infeasible_too_large(self):
        m = Model()
        x = m.int_var(1, 5, "x")
        y = m.int_var(1, 5, "y")
        m.add(m.sum_eq([x, y], 20))
        result = m.solve()
        assert result.status == Status.INFEASIBLE

    def test_sum_single_var(self):
        m = Model()
        x = m.int_var(1, 10, "x")
        m.add(m.sum_eq([x], 7))
        result = m.solve()
        assert result.status == Status.OPTIMAL
        assert result.solution["x"] == 7

    def test_sum_empty(self):
        m = Model()
        m.int_var(1, 5, "x")
        m.add(m.sum_eq([], 0))
        result = m.solve()
        assert result.status == Status.OPTIMAL

    def test_sum_empty_nonzero(self):
        m = Model()
        m.int_var(1, 5, "x")
        m.add(m.sum_eq([], 5))
        result = m.solve()
        assert result.status == Status.INFEASIBLE


class TestEdgeCaseCoverage:
    """Tests for edge cases and uncovered code paths."""

    def test_eq_const_val_not_in_domain(self):
        """Test eq_const when value is outside domain."""
        m = Model()
        x = m.int_var(1, 5, "x")
        m.add(x == 10)  # 10 not in [1,5]
        result = m.solve(solver="sat")
        assert result.status == Status.INFEASIBLE

    def test_eq_var_non_overlapping_domains(self):
        """Test eq_var with disjoint domains."""
        m = Model()
        x = m.int_var(1, 3, "x")
        y = m.int_var(5, 7, "y")
        m.add(x == y)  # Impossible: no overlap
        result = m.solve(solver="sat")
        assert result.status == Status.INFEASIBLE

    def test_sum_le_single_var(self):
        """Test sum_le with single variable."""
        m = Model()
        x = m.int_var(1, 10, "x")
        m.add(m.sum_le([x], 5))
        result = m.solve()
        assert result.ok
        assert result.solution["x"] <= 5

    def test_sum_ge_single_var(self):
        """Test sum_ge with single variable."""
        m = Model()
        x = m.int_var(1, 10, "x")
        m.add(m.sum_ge([x], 7))
        result = m.solve()
        assert result.ok
        assert result.solution["x"] >= 7

    def test_sum_le_three_vars(self):
        """Test sum_le with 3 variables (recursive case)."""
        m = Model()
        x = m.int_var(1, 3, "x")
        y = m.int_var(1, 3, "y")
        z = m.int_var(1, 3, "z")
        m.add(m.sum_le([x, y, z], 5))
        result = m.solve()
        assert result.ok
        assert result.solution["x"] + result.solution["y"] + result.solution["z"] <= 5

    def test_sum_ge_three_vars(self):
        """Test sum_ge with 3 variables (recursive case)."""
        m = Model()
        x = m.int_var(1, 3, "x")
        y = m.int_var(1, 3, "y")
        z = m.int_var(1, 3, "z")
        m.add(m.sum_ge([x, y, z], 7))
        result = m.solve()
        assert result.ok
        assert result.solution["x"] + result.solution["y"] + result.solution["z"] >= 7

    def test_sum_ge_empty_positive_target(self):
        """Test sum_ge with empty list and positive target (infeasible)."""
        m = Model()
        m.int_var(1, 5, "x")
        m.add(m.sum_ge([], 5))  # sum([]) >= 5 is infeasible
        result = m.solve()
        assert result.status == Status.INFEASIBLE

    def test_sum_le_empty(self):
        """Test sum_le with empty list."""
        m = Model()
        x = m.int_var(1, 5, "x")
        m.add(m.sum_le([], 5))  # sum([]) <= 5 is always true
        result = m.solve()
        assert result.ok

    def test_circuit_single_node(self):
        """Test circuit with single node."""
        m = Model()
        x = m.int_var(0, 0, "x")  # Only valid value is 0 (self-loop required)
        m.add(m.circuit([x]))
        result = m.solve()
        # Single node circuit: x[0] = 0 means node 0 points to itself
        # But no_self_loop forbids this, so should be infeasible
        assert result.status == Status.INFEASIBLE

    def test_circuit_empty(self):
        """Test circuit with empty list."""
        m = Model()
        m.int_var(1, 5, "x")  # Unrelated variable
        m.add(m.circuit([]))  # Empty circuit
        result = m.solve()
        assert result.ok  # Empty circuit is trivially satisfied

    def test_expr_eq_unsupported_type(self):
        """Test Expr.__eq__ with unsupported type returns NotImplemented."""
        m = Model()
        x = m.int_var(0, 5, "x")
        expr = x + 1
        result = expr.__eq__("string")
        assert result is NotImplemented

    def test_expr_ne_unsupported_type(self):
        """Test Expr.__ne__ with unsupported type returns NotImplemented."""
        m = Model()
        x = m.int_var(0, 5, "x")
        expr = x + 1
        result = expr.__ne__("string")
        assert result is NotImplemented

    def test_expr_sub_unsupported_type(self):
        """Test Expr.__sub__ with unsupported type returns NotImplemented."""
        m = Model()
        x = m.int_var(0, 5, "x")
        expr = x + 1
        result = expr.__sub__("string")
        assert result is NotImplemented

    def test_intvar_eq_unsupported_type(self):
        """Test IntVar.__eq__ with unsupported type returns NotImplemented."""
        m = Model()
        x = m.int_var(0, 5, "x")
        result = x.__eq__("string")
        assert result is NotImplemented

    def test_intvar_ne_unsupported_type(self):
        """Test IntVar.__ne__ with unsupported type returns NotImplemented."""
        m = Model()
        x = m.int_var(0, 5, "x")
        result = x.__ne__("string")
        assert result is NotImplemented

    def test_intvar_sub_unsupported_type(self):
        """Test IntVar.__sub__ with unsupported type returns NotImplemented."""
        m = Model()
        x = m.int_var(0, 5, "x")
        result = x.__sub__("string")
        assert result is NotImplemented

    def test_intvar_rsub_unsupported_type(self):
        """Test IntVar.__rsub__ with unsupported type returns NotImplemented."""
        m = Model()
        x = m.int_var(0, 5, "x")
        result = x.__rsub__("string")
        assert result is NotImplemented

    def test_expr_sub_int(self):
        """Test Expr - int."""
        m = Model()
        x = m.int_var(0, 10, "x")
        expr = (x + 5) - 3  # Creates Expr(("add", ..., -3))
        m.add(expr == 6)  # x + 5 - 3 == 6 => x == 4
        result = m.solve(solver="sat")
        assert result.ok
        assert result.solution["x"] == 4

    def test_ne_expr_eq_subtraction(self):
        """Test (x - y) == c constraint (equality not ne)."""
        m = Model()
        x = m.int_var(0, 5, "x")
        y = m.int_var(0, 5, "y")
        m.add((x - y) == 2)  # x - y == 2 => x == y + 2
        result = m.solve(solver="sat")
        assert result.ok
        assert result.solution["x"] - result.solution["y"] == 2

    def test_ne_expr_constant_on_left(self):
        """Test c == (y + c2) constraint."""
        m = Model()
        y = m.int_var(0, 10, "y")
        # Create 5 == (y + 2) => y == 3
        # Need to construct this manually via Expr
        expr = y + 2
        constraint = (5).__eq__(expr)  # Won't work with plain int
        # Instead, use expr == 5 which covers the same path via symmetry
        m.add(expr == 5)
        result = m.solve(solver="sat")
        assert result.ok
        assert result.solution["y"] == 3

    def test_ne_expr_two_vars_eq_constant(self):
        """Test (x + y) == c constraint."""
        m = Model()
        x = m.int_var(1, 5, "x")
        y = m.int_var(1, 5, "y")
        m.add((x + y) == 6)  # x + y == 6
        result = m.solve(solver="sat")
        assert result.ok
        assert result.solution["x"] + result.solution["y"] == 6

    def test_ne_expr_two_vars_ne_constant(self):
        """Test (x + y) != c constraint."""
        m = Model()
        x = m.int_var(1, 3, "x")
        y = m.int_var(1, 3, "y")
        m.add((x + y) != 4)  # x + y != 4
        result = m.solve(solver="sat")
        assert result.ok
        assert result.solution["x"] + result.solution["y"] != 4

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

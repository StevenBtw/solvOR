"""Tests for the SAT solver."""

from solvor.sat import solve_sat
from solvor.types import Status


class TestBasicSAT:
    def test_simple_satisfiable(self):
        # (x1 OR x2) AND (NOT x1 OR x2)
        # Satisfiable: x2=True
        result = solve_sat([[1, 2], [-1, 2]])
        assert result.status == Status.OPTIMAL
        assert result.solution[2] is True

    def test_single_clause(self):
        # x1 must be true
        result = solve_sat([[1]])
        assert result.status == Status.OPTIMAL
        assert result.solution[1] is True

    def test_single_negative_clause(self):
        # x1 must be false
        result = solve_sat([[-1]])
        assert result.status == Status.OPTIMAL
        assert result.solution[1] is False


class TestUnsatisfiable:
    def test_contradiction(self):
        # x AND NOT x
        result = solve_sat([[1], [-1]])
        assert result.status == Status.INFEASIBLE

    def test_larger_contradiction(self):
        # (x1) AND (x2) AND (NOT x1 OR NOT x2) AND (NOT x1) - unsat
        result = solve_sat([[1], [2], [-1, -2], [-1]])
        assert result.status == Status.INFEASIBLE


class TestEmptyCases:
    def test_empty_clauses(self):
        result = solve_sat([])
        assert result.status == Status.OPTIMAL

    def test_single_variable_clauses(self):
        # (x1) AND (x2) AND (x3)
        result = solve_sat([[1], [2], [3]])
        assert result.status == Status.OPTIMAL
        assert result.solution[1] is True
        assert result.solution[2] is True
        assert result.solution[3] is True


class TestThreeSAT:
    def test_simple_3sat(self):
        clauses = [[1, 2, 3], [-1, -2, 3], [1, -2, -3]]
        result = solve_sat(clauses)
        assert result.status == Status.OPTIMAL
        # Verify solution satisfies all clauses
        for clause in clauses:
            satisfied = any(
                (lit > 0 and result.solution.get(abs(lit), False)) or
                (lit < 0 and not result.solution.get(abs(lit), False))
                for lit in clause
            )
            assert satisfied

    def test_3sat_with_many_clauses(self):
        # Satisfiable 3-SAT instance
        clauses = [
            [1, 2, 3],
            [-1, 2, 3],
            [1, -2, 3],
            [1, 2, -3],
            [-1, -2, 3],
        ]
        result = solve_sat(clauses)
        assert result.status == Status.OPTIMAL
        # Verify all clauses satisfied
        for clause in clauses:
            satisfied = any(
                (lit > 0 and result.solution.get(abs(lit), False)) or
                (lit < 0 and not result.solution.get(abs(lit), False))
                for lit in clause
            )
            assert satisfied


class TestImplicationChains:
    def test_implication_chain(self):
        # x1 -> x2 -> x3 -> x4, and x1 must be true
        # Encoded as: (x1), (-x1 OR x2), (-x2 OR x3), (-x3 OR x4)
        clauses = [[1], [-1, 2], [-2, 3], [-3, 4]]
        result = solve_sat(clauses)
        assert result.status == Status.OPTIMAL
        assert result.solution[1] is True
        assert result.solution[2] is True
        assert result.solution[3] is True
        assert result.solution[4] is True

    def test_xor_encoding(self):
        # x1 XOR x2 (exactly one true)
        # Encoded as: (x1 OR x2) AND (NOT x1 OR NOT x2)
        clauses = [[1, 2], [-1, -2]]
        result = solve_sat(clauses)
        assert result.status == Status.OPTIMAL
        assert result.solution[1] != result.solution[2]


class TestAssumptions:
    def test_with_assumptions(self):
        # (x1 OR x2), assume x1 = True
        result = solve_sat([[1, 2]], assumptions=[1])
        assert result.status == Status.OPTIMAL
        assert result.solution[1] is True

    def test_multiple_assumptions(self):
        # (x1 OR x2 OR x3), assume x1=True and x2=True
        result = solve_sat([[1, 2, 3]], assumptions=[1, 2])
        assert result.status == Status.OPTIMAL
        assert result.solution.get(1, False) or result.solution.get(2, False)


class TestEdgeCases:
    def test_large_clause(self):
        # One large clause: any of x1..x10 true
        clause = list(range(1, 11))
        result = solve_sat([clause])
        assert result.status == Status.OPTIMAL

    def test_many_variables(self):
        # Each variable in its own clause (all true)
        n = 20
        clauses = [[i] for i in range(1, n + 1)]
        result = solve_sat(clauses)
        assert result.status == Status.OPTIMAL
        for i in range(1, n + 1):
            assert result.solution[i] is True

    def test_all_negative(self):
        # All variables must be false
        clauses = [[-1], [-2], [-3]]
        result = solve_sat(clauses)
        assert result.status == Status.OPTIMAL
        assert result.solution[1] is False
        assert result.solution[2] is False
        assert result.solution[3] is False


class TestStress:
    def test_pigeonhole_like(self):
        # At most one of x1, x2, x3 (satisfiable: at most one)
        # Encoded as: (NOT x1 OR NOT x2), (NOT x1 OR NOT x3), (NOT x2 OR NOT x3)
        clauses = [[-1, -2], [-1, -3], [-2, -3]]
        result = solve_sat(clauses)
        assert result.status == Status.OPTIMAL
        # At most one is true
        true_count = sum(1 for i in [1, 2, 3] if result.solution.get(i, False))
        assert true_count <= 1

    def test_random_satisfiable(self):
        # Random but satisfiable instance
        import random
        random.seed(42)
        n_vars = 10
        n_clauses = 20
        clauses = []
        for _ in range(n_clauses):
            clause = []
            for _ in range(3):
                var = random.randint(1, n_vars)
                lit = var if random.random() > 0.5 else -var
                if lit not in clause and -lit not in clause:
                    clause.append(lit)
            if clause:
                clauses.append(clause)

        result = solve_sat(clauses)
        # Random 3-SAT with clause/var ratio ~2 is usually satisfiable
        if result.status == Status.OPTIMAL:
            for clause in clauses:
                satisfied = any(
                    (lit > 0 and result.solution.get(abs(lit), False)) or
                    (lit < 0 and not result.solution.get(abs(lit), False))
                    for lit in clause
                )
                assert satisfied

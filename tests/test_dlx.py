"""Tests for the DLX (Dancing Links / Algorithm X) exact cover solver."""

import pytest
from solvor.dlx import solve_exact_cover, Status


class TestBasicExactCover:
    def test_knuth_example(self):
        # Knuth's example from "Dancing Links" paper
        matrix = [
            [1, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 1],
            [0, 0, 1, 0, 1, 1, 0],
            [0, 1, 1, 0, 0, 1, 1],
            [0, 1, 0, 0, 0, 0, 1],
        ]
        result = solve_exact_cover(matrix)
        assert result.status == Status.OPTIMAL
        # Rows 1, 3, 5 form exact cover (0-indexed)
        assert set(result.solution) == {1, 3, 5}

    def test_identity_matrix(self):
        # Identity matrix - trivial exact cover
        matrix = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
        result = solve_exact_cover(matrix)
        assert result.status == Status.OPTIMAL
        assert set(result.solution) == {0, 1, 2}

    def test_simple_cover(self):
        # Simple 2-row cover
        matrix = [
            [1, 1, 0],
            [0, 0, 1],
        ]
        result = solve_exact_cover(matrix)
        assert result.status == Status.OPTIMAL
        assert set(result.solution) == {0, 1}


class TestInfeasible:
    def test_no_solution_missing_column(self):
        # No row covers column 2
        matrix = [
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ]
        result = solve_exact_cover(matrix)
        assert result.status == Status.INFEASIBLE

    def test_overlapping_only(self):
        # All rows overlap - no exact cover possible
        matrix = [
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
        ]
        result = solve_exact_cover(matrix)
        assert result.status == Status.INFEASIBLE

    def test_pentomino_impossible(self):
        matrix = [
            [1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1],
            [1, 0, 0, 0, 1],
        ]
        result = solve_exact_cover(matrix)
        assert result.status == Status.INFEASIBLE


class TestEmptyCases:
    def test_empty_matrix(self):
        result = solve_exact_cover([])
        assert result.status == Status.OPTIMAL
        assert result.solution == ()

    def test_single_row(self):
        matrix = [[1, 1, 1]]
        result = solve_exact_cover(matrix)
        assert result.status == Status.OPTIMAL
        assert result.solution == (0,)

    def test_single_column(self):
        matrix = [[1], [0], [1]]
        result = solve_exact_cover(matrix)
        assert result.status == Status.OPTIMAL
        # Either row 0 or row 2
        assert result.solution[0] in (0, 2)


class TestMultipleSolutions:
    def test_find_all_solutions(self):
        matrix = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [0, 1, 1],
        ]
        result = solve_exact_cover(matrix, find_all=True)
        assert result.status == Status.OPTIMAL
        # Multiple solutions exist: {0,1,2}, {0,4}, {3,2}
        assert len(result.solution) >= 2

    def test_max_solutions_limit(self):
        matrix = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
        ]
        result = solve_exact_cover(matrix, find_all=True, max_solutions=2)
        assert result.status == Status.FEASIBLE
        assert len(result.solution) == 2

    def test_unique_solution(self):
        # Only one solution exists
        matrix = [
            [1, 0],
            [0, 1],
        ]
        result = solve_exact_cover(matrix, find_all=True)
        assert result.status == Status.OPTIMAL
        assert len(result.solution) == 1


class TestColumnNames:
    def test_with_column_names(self):
        matrix = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
        result = solve_exact_cover(matrix, columns=['A', 'B', 'C'])
        assert result.status == Status.OPTIMAL
        assert set(result.solution) == {0, 1, 2}

    def test_column_names_no_effect(self):
        # Column names shouldn't affect the solution
        matrix = [
            [1, 1, 0],
            [0, 0, 1],
        ]
        result1 = solve_exact_cover(matrix)
        result2 = solve_exact_cover(matrix, columns=['X', 'Y', 'Z'])
        assert result1.solution == result2.solution


class TestSudokuLike:
    def test_mini_sudoku_2x2(self):
        # 2x2 Sudoku has 4 cells, values 1-2
        # Encode as exact cover (simplified)
        # Columns: cell00, cell01, cell10, cell11
        # Rows: each possible (cell, value) assignment
        matrix = [
            [1, 0, 0, 0],  # cell00 = 1
            [1, 0, 0, 0],  # cell00 = 2 (duplicate column, choose one)
            [0, 1, 0, 0],  # cell01 = 1
            [0, 0, 1, 0],  # cell10 = 1
            [0, 0, 0, 1],  # cell11 = 1
        ]
        result = solve_exact_cover(matrix)
        assert result.status == Status.OPTIMAL


class TestNQueensLike:
    def test_4_queens_cover(self):
        # N-Queens as exact cover (each row and column covered exactly once)
        # For N=2, this is infeasible
        # For N=4, represent columns as positions, rows as queen placements
        # Simplified: just column coverage
        matrix = [
            [1, 0, 0, 0],  # Queen in col 0
            [0, 1, 0, 0],  # Queen in col 1
            [0, 0, 1, 0],  # Queen in col 2
            [0, 0, 0, 1],  # Queen in col 3
        ]
        result = solve_exact_cover(matrix)
        assert result.status == Status.OPTIMAL
        assert len(result.solution) == 4


class TestEdgeCases:
    def test_all_ones_row(self):
        # One row covers everything
        matrix = [
            [1, 1, 1, 1],
            [1, 0, 0, 0],
        ]
        result = solve_exact_cover(matrix)
        assert result.status == Status.OPTIMAL
        assert result.solution == (0,)

    def test_sparse_matrix(self):
        # Mostly zeros
        matrix = [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ]
        result = solve_exact_cover(matrix)
        assert result.status == Status.OPTIMAL
        assert set(result.solution) == {0, 1, 2, 3, 4}

    def test_redundant_rows(self):
        # Multiple rows that could work
        matrix = [
            [1, 0, 0],
            [1, 0, 0],  # Duplicate
            [0, 1, 1],
        ]
        result = solve_exact_cover(matrix)
        assert result.status == Status.OPTIMAL
        # Should use either row 0 or 1, plus row 2
        assert 2 in result.solution
        assert len(result.solution) == 2


class TestStress:
    def test_larger_matrix(self):
        # 10x10 identity matrix
        n = 10
        matrix = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        result = solve_exact_cover(matrix)
        assert result.status == Status.OPTIMAL
        assert set(result.solution) == set(range(n))

    def test_many_rows(self):
        # Many rows, few columns
        n_rows = 20
        n_cols = 5
        import random
        random.seed(42)

        # Ensure at least one solution exists
        matrix = []
        # Add identity-like rows first
        for j in range(n_cols):
            row = [0] * n_cols
            row[j] = 1
            matrix.append(row)
        # Add random rows
        for _ in range(n_rows - n_cols):
            row = [random.randint(0, 1) for _ in range(n_cols)]
            if sum(row) > 0:  # Non-empty row
                matrix.append(row)

        result = solve_exact_cover(matrix)
        assert result.status == Status.OPTIMAL

    def test_verify_solution(self):
        matrix = [
            [1, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 1],
            [0, 0, 1, 0, 1, 1, 0],
            [0, 1, 1, 0, 0, 1, 1],
            [0, 1, 0, 0, 0, 0, 1],
        ]
        result = solve_exact_cover(matrix)
        assert result.status == Status.OPTIMAL

        # Verify the solution is valid
        n_cols = len(matrix[0])
        covered = [0] * n_cols
        for row_idx in result.solution:
            for j in range(n_cols):
                covered[j] += matrix[row_idx][j]

        # Each column should be covered exactly once
        assert all(c == 1 for c in covered)

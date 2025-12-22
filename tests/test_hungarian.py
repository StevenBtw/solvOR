"""Tests for Hungarian Algorithm."""

from solvor.hungarian import hungarian
from solvor.types import Status


class TestBasic:
    def test_simple_2x2(self):
        costs = [[1, 2], [3, 4]]
        result = hungarian(costs)
        assert result.status == Status.OPTIMAL
        assert result.objective == 5
        assert result.solution[0] == 0
        assert result.solution[1] == 1

    def test_swap_better(self):
        costs = [[1, 10], [10, 1]]
        result = hungarian(costs)
        assert result.status == Status.OPTIMAL
        assert result.objective == 2
        assert result.solution[0] == 0
        assert result.solution[1] == 1

    def test_3x3(self):
        costs = [[10, 5, 13], [3, 9, 18], [10, 6, 12]]
        result = hungarian(costs)
        assert result.status == Status.OPTIMAL
        assert len(result.solution) == 3
        assert len(set(result.solution)) == 3
        # Optimal: 0→1 (5), 1→0 (3), 2→2 (12) = 20
        assert result.objective == 20


class TestMaximize:
    def test_maximize_2x2(self):
        costs = [[1, 4], [3, 2]]
        result = hungarian(costs, minimize=False)
        assert result.status == Status.OPTIMAL
        assert result.objective == 7
        assert result.solution[0] == 1
        assert result.solution[1] == 0

    def test_maximize_3x3(self):
        costs = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        result = hungarian(costs, minimize=False)
        assert result.status == Status.OPTIMAL
        assert result.objective == 15


class TestRectangular:
    def test_more_cols(self):
        costs = [[1, 2, 3, 4], [5, 6, 7, 8]]
        result = hungarian(costs)
        assert result.status == Status.OPTIMAL
        assert len(result.solution) == 2

    def test_more_rows(self):
        costs = [[1, 2], [3, 4], [5, 6], [7, 8]]
        result = hungarian(costs)
        assert result.status == Status.OPTIMAL
        assigned = [a for a in result.solution if a != -1]
        assert len(assigned) == 2


class TestUniform:
    def test_all_same(self):
        costs = [[5, 5, 5], [5, 5, 5], [5, 5, 5]]
        result = hungarian(costs)
        assert result.status == Status.OPTIMAL
        assert result.objective == 15
        assert len(set(result.solution)) == 3

    def test_all_zeros(self):
        costs = [[0, 0], [0, 0]]
        result = hungarian(costs)
        assert result.status == Status.OPTIMAL
        assert result.objective == 0


class TestEdgeCases:
    def test_single_element(self):
        costs = [[5]]
        result = hungarian(costs)
        assert result.status == Status.OPTIMAL
        assert result.objective == 5
        assert result.solution == [0]

    def test_empty(self):
        result = hungarian([])
        assert result.status == Status.OPTIMAL
        assert result.solution == []

    def test_empty_row(self):
        result = hungarian([[]])
        assert result.status == Status.OPTIMAL


class TestClassicProblems:
    def test_assignment_problem(self):
        costs = [[9, 2, 7, 8], [6, 4, 3, 7], [5, 8, 1, 8], [7, 6, 9, 4]]
        result = hungarian(costs)
        assert result.status == Status.OPTIMAL
        assert len(set(result.solution)) == 4
        assert result.objective == 13

    def test_job_assignment(self):
        costs = [[82, 83, 69, 92], [77, 37, 49, 92], [11, 69, 5, 86], [8, 9, 98, 23]]
        result = hungarian(costs)
        assert result.status == Status.OPTIMAL
        assert len(set(result.solution)) == 4


class TestStress:
    def test_larger_matrix(self):
        import random

        random.seed(42)
        n = 20
        costs = [[random.randint(1, 100) for _ in range(n)] for _ in range(n)]
        result = hungarian(costs)
        assert result.status == Status.OPTIMAL
        assert len(result.solution) == n
        assert len(set(result.solution)) == n

    def test_large_values(self):
        costs = [[1000000, 2000000], [3000000, 4000000]]
        result = hungarian(costs)
        assert result.status == Status.OPTIMAL
        assert result.objective == 5000000

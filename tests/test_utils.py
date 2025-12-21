"""Tests for utility functions."""

from solvor.utils import (
    assignment_cost,
    is_feasible,
    random_permutation,
    pairwise_swap_neighbors,
    fenwick_build,
    fenwick_update,
    fenwick_prefix,
)


class TestAssignmentCost:
    def test_simple(self):
        matrix = [[1, 2], [3, 4]]
        assert assignment_cost(matrix, [0, 1]) == 5
        assert assignment_cost(matrix, [1, 0]) == 5

    def test_with_unassigned(self):
        matrix = [[1, 2], [3, 4]]
        assert assignment_cost(matrix, [0, -1]) == 1

    def test_empty(self):
        assert assignment_cost([], []) == 0


class TestIsFeasible:
    def test_feasible(self):
        A = [[1, 1], [2, 1]]
        b = [4, 5]
        x = [1, 1]
        assert is_feasible(A, b, x) is True

    def test_infeasible(self):
        A = [[1, 1]]
        b = [2]
        x = [2, 2]
        assert is_feasible(A, b, x) is False

    def test_boundary(self):
        A = [[1, 0]]
        b = [5]
        x = [5, 0]
        assert is_feasible(A, b, x) is True

class TestRandomPermutation:
    def test_length(self):
        perm = random_permutation(10)
        assert len(perm) == 10

    def test_contains_all(self):
        perm = random_permutation(10)
        assert set(perm) == set(range(10))

    def test_empty(self):
        assert random_permutation(0) == []

    def test_single(self):
        assert random_permutation(1) == [0]

class TestPairwiseSwapNeighbors:
    def test_count(self):
        perm = [0, 1, 2]
        neighbors = list(pairwise_swap_neighbors(perm))
        assert len(neighbors) == 3

    def test_swaps(self):
        perm = [0, 1, 2]
        neighbors = list(pairwise_swap_neighbors(perm))
        assert [1, 0, 2] in neighbors
        assert [2, 1, 0] in neighbors
        assert [0, 2, 1] in neighbors

    def test_original_unchanged(self):
        perm = [0, 1, 2]
        list(pairwise_swap_neighbors(perm))
        assert perm == [0, 1, 2]

class TestFenwick:
    def test_build_and_prefix(self):
        values = [1, 2, 3, 4, 5]
        tree = fenwick_build(values)
        assert fenwick_prefix(tree, 0) == 1
        assert fenwick_prefix(tree, 1) == 3
        assert fenwick_prefix(tree, 2) == 6
        assert fenwick_prefix(tree, 4) == 15

    def test_update(self):
        values = [1, 2, 3, 4, 5]
        tree = fenwick_build(values)
        fenwick_update(tree, 2, 10)
        assert fenwick_prefix(tree, 2) == 16
        assert fenwick_prefix(tree, 4) == 25

    def test_empty(self):
        tree = fenwick_build([])
        assert tree == []

    def test_single(self):
        tree = fenwick_build([5])
        assert fenwick_prefix(tree, 0) == 5
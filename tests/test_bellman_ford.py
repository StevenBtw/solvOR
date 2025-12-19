"""Tests for Bellman-Ford algorithm."""

from solvor.bellman_ford import bellman_ford
from solvor.types import Status


class TestBasic:
    def test_simple_path(self):
        edges = [(0, 1, 1), (1, 2, 2)]
        result = bellman_ford(0, edges, 3, target=2)
        assert result.status == Status.OPTIMAL
        assert result.solution == [0, 1, 2]
        assert result.objective == 3

    def test_shortest_path(self):
        edges = [
            (0, 1, 4),
            (0, 2, 5),
            (1, 2, -3),
            (2, 3, 4)
        ]
        result = bellman_ford(0, edges, 4, target=3)
        assert result.status == Status.OPTIMAL
        assert result.objective == 5

    def test_all_distances(self):
        edges = [(0, 1, 1), (1, 2, 2), (0, 2, 5)]
        result = bellman_ford(0, edges, 3)
        assert result.status == Status.OPTIMAL
        assert result.solution[0] == 0
        assert result.solution[1] == 1
        assert result.solution[2] == 3


class TestNegativeWeights:
    def test_negative_edge(self):
        edges = [(0, 1, 5), (1, 2, -3), (2, 3, 1)]
        result = bellman_ford(0, edges, 4, target=3)
        assert result.status == Status.OPTIMAL
        assert result.objective == 3

    def test_multiple_negative_edges(self):
        edges = [
            (0, 1, 2),
            (1, 2, -1),
            (2, 3, -1),
            (0, 3, 5)
        ]
        result = bellman_ford(0, edges, 4, target=3)
        assert result.status == Status.OPTIMAL
        assert result.objective == 0


class TestNegativeCycle:
    def test_simple_negative_cycle(self):
        edges = [
            (0, 1, 1),
            (1, 2, -1),
            (2, 0, -1)
        ]
        result = bellman_ford(0, edges, 3)
        assert result.status == Status.UNBOUNDED

    def test_negative_cycle_detection(self):
        edges = [
            (0, 1, 4),
            (1, 2, -6),
            (2, 1, 2)
        ]
        result = bellman_ford(0, edges, 3)
        assert result.status == Status.UNBOUNDED

    def test_unreachable_negative_cycle(self):
        edges = [
            (0, 1, 1),
            (2, 3, -2),
            (3, 2, -2)
        ]
        result = bellman_ford(0, edges, 4, target=1)
        assert result.status == Status.OPTIMAL


class TestNoPath:
    def test_disconnected(self):
        edges = [(0, 1, 1)]
        result = bellman_ford(0, edges, 3, target=2)
        assert result.status == Status.INFEASIBLE

    def test_no_edges(self):
        result = bellman_ford(0, [], 3, target=2)
        assert result.status == Status.INFEASIBLE


class TestEdgeCases:
    def test_single_vertex(self):
        result = bellman_ford(0, [], 1)
        assert result.status == Status.OPTIMAL
        assert result.solution == {0: 0}

    def test_self_loop(self):
        edges = [(0, 0, 5), (0, 1, 1)]
        result = bellman_ford(0, edges, 2, target=1)
        assert result.status == Status.OPTIMAL
        assert result.objective == 1

    def test_parallel_edges(self):
        edges = [(0, 1, 5), (0, 1, 3)]
        result = bellman_ford(0, edges, 2, target=1)
        assert result.status == Status.OPTIMAL
        assert result.objective == 3

    def test_zero_weight(self):
        edges = [(0, 1, 0), (1, 2, 0)]
        result = bellman_ford(0, edges, 3, target=2)
        assert result.status == Status.OPTIMAL
        assert result.objective == 0


class TestStress:
    def test_linear_graph(self):
        n = 100
        edges = [(i, i + 1, 1) for i in range(n - 1)]
        result = bellman_ford(0, edges, n, target=n - 1)
        assert result.status == Status.OPTIMAL
        assert result.objective == n - 1

    def test_dense_graph(self):
        n = 20
        edges = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    edges.append((i, j, abs(i - j)))
        result = bellman_ford(0, edges, n, target=n - 1)
        assert result.status == Status.OPTIMAL
        assert result.objective == n - 1

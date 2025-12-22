"""Tests for Dijkstra's algorithm."""

from solvor.dijkstra import dijkstra
from solvor.types import Status


class TestBasic:
    def test_direct_path(self):
        graph = {"A": [("B", 1)], "B": [("C", 2)], "C": []}
        result = dijkstra("A", "C", lambda n: graph.get(n, []))
        assert result.status == Status.OPTIMAL
        assert result.solution == ["A", "B", "C"]
        assert result.objective == 3

    def test_shortest_weighted(self):
        graph = {"A": [("B", 1), ("C", 4)], "B": [("C", 2), ("D", 5)], "C": [("D", 1)], "D": []}
        result = dijkstra("A", "D", lambda n: graph.get(n, []))
        assert result.status == Status.OPTIMAL
        assert result.objective == 4
        assert result.solution == ["A", "B", "C", "D"]

    def test_start_is_goal(self):
        graph = {"A": [("B", 1)]}
        result = dijkstra("A", "A", lambda n: graph.get(n, []))
        assert result.status == Status.OPTIMAL
        assert result.solution == ["A"]
        assert result.objective == 0

    def test_no_path(self):
        graph = {"A": [("B", 1)], "B": [], "C": []}
        result = dijkstra("A", "C", lambda n: graph.get(n, []))
        assert result.status == Status.INFEASIBLE


class TestGoalPredicate:
    def test_find_any_large(self):
        graph = {0: [(1, 1), (2, 1)], 1: [(10, 1)], 2: [(20, 1)], 10: [], 20: []}
        result = dijkstra(0, lambda s: s >= 10, lambda n: graph.get(n, []))
        assert result.status == Status.OPTIMAL
        assert result.solution[-1] >= 10

    def test_find_specific_value(self):
        graph = {i: [(i + 1, 1)] for i in range(10)}
        graph[10] = []
        result = dijkstra(0, lambda s: s == 5, lambda n: graph.get(n, []))
        assert result.solution[-1] == 5


class TestMaxCost:
    def test_within_budget(self):
        graph = {"A": [("B", 2)], "B": [("C", 2)], "C": []}
        result = dijkstra("A", "C", lambda n: graph.get(n, []), max_cost=10)
        assert result.status == Status.OPTIMAL
        assert result.objective == 4

    def test_pruned_by_max_cost(self):
        graph = {"A": [("B", 10)], "B": [("C", 1)], "C": []}
        result = dijkstra("A", "C", lambda n: graph.get(n, []), max_cost=5)
        assert result.status == Status.INFEASIBLE


class TestEdgeCases:
    def test_single_node(self):
        result = dijkstra("A", "A", lambda n: [])
        assert result.solution == ["A"]
        assert result.objective == 0

    def test_parallel_edges(self):
        graph = {"A": [("B", 5), ("B", 3)], "B": []}
        result = dijkstra("A", "B", lambda n: graph.get(n, []))
        assert result.objective == 3

    def test_zero_weight_edges(self):
        graph = {"A": [("B", 0)], "B": [("C", 0)], "C": []}
        result = dijkstra("A", "C", lambda n: graph.get(n, []))
        assert result.objective == 0

    def test_max_iter_limit(self):
        def infinite_neighbors(n):
            return [(n + 1, 1)]

        result = dijkstra(0, 1000000, infinite_neighbors, max_iter=50)
        assert result.status == Status.MAX_ITER


class TestComplex:
    def test_diamond_graph(self):
        graph = {"S": [("A", 1), ("B", 4)], "A": [("B", 2), ("T", 5)], "B": [("T", 1)], "T": []}
        result = dijkstra("S", "T", lambda n: graph.get(n, []))
        assert result.status == Status.OPTIMAL
        assert result.objective == 4
        assert result.solution == ["S", "A", "B", "T"]

    def test_grid_navigation(self):
        grid = {}
        for r in range(5):
            for c in range(5):
                neighbors = []
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 5 and 0 <= nc < 5:
                        neighbors.append(((nr, nc), 1))
                grid[(r, c)] = neighbors

        result = dijkstra((0, 0), (4, 4), lambda n: grid.get(n, []))
        assert result.status == Status.OPTIMAL
        assert result.objective == 8


class TestStress:
    def test_large_graph(self):
        n = 100
        graph = {}
        for i in range(n):
            graph[i] = [(i + 1, 1)] if i < n - 1 else []

        result = dijkstra(0, n - 1, lambda node: graph.get(node, []))
        assert result.status == Status.OPTIMAL
        assert result.objective == n - 1

    def test_dense_graph(self):
        n = 20
        graph = {}
        for i in range(n):
            graph[i] = [(j, abs(i - j)) for j in range(n) if j != i]

        result = dijkstra(0, n - 1, lambda node: graph.get(node, []))
        assert result.status == Status.OPTIMAL
        assert result.objective == n - 1

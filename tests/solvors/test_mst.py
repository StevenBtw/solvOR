"""Tests for Minimum Spanning Tree algorithms."""

from solvor.mst import kruskal, prim
from solvor.types import Status


class TestKruskalBasic:
    def test_simple_triangle(self):
        edges = [(0, 1, 1), (1, 2, 2), (0, 2, 3)]
        result = kruskal(3, edges)
        assert result.status == Status.OPTIMAL
        assert result.objective == 3
        assert len(result.solution) == 2

    def test_line_graph(self):
        edges = [(0, 1, 1), (1, 2, 1), (2, 3, 1)]
        result = kruskal(4, edges)
        assert result.status == Status.OPTIMAL
        assert result.objective == 3
        assert len(result.solution) == 3

    def test_choose_lighter_edge(self):
        edges = [(0, 1, 10), (0, 1, 5), (1, 2, 3)]
        result = kruskal(3, edges)
        assert result.status == Status.OPTIMAL
        assert result.objective == 8

    def test_single_vertex(self):
        result = kruskal(1, [])
        assert result.status == Status.OPTIMAL
        assert result.objective == 0
        assert result.solution == []


class TestKruskalDisconnected:
    def test_disconnected_graph(self):
        edges = [(0, 1, 1)]
        result = kruskal(3, edges)
        assert result.status == Status.INFEASIBLE

    def test_isolated_vertices(self):
        edges = [(0, 1, 1), (2, 3, 1)]
        result = kruskal(4, edges)
        assert result.status == Status.INFEASIBLE


class TestKruskalAllowForest:
    def test_allow_forest_disconnected(self):
        edges = [(0, 1, 1)]
        result = kruskal(3, edges, allow_forest=True)
        assert result.status == Status.FEASIBLE
        assert result.solution == [(0, 1, 1)]
        assert result.objective == 1

    def test_allow_forest_two_components(self):
        edges = [(0, 1, 1), (2, 3, 2)]
        result = kruskal(4, edges, allow_forest=True)
        assert result.status == Status.FEASIBLE
        assert len(result.solution) == 2
        assert result.objective == 3

    def test_allow_forest_connected_same_as_regular(self):
        edges = [(0, 1, 1), (1, 2, 2), (0, 2, 3)]
        result_regular = kruskal(3, edges)
        result_forest = kruskal(3, edges, allow_forest=True)
        assert result_regular.objective == result_forest.objective
        assert result_regular.solution == result_forest.solution

    def test_allow_forest_no_edges(self):
        result = kruskal(3, [], allow_forest=True)
        assert result.status == Status.FEASIBLE
        assert result.solution == []
        assert result.objective == 0

    def test_allow_forest_multiple_components(self):
        # 6 nodes in 3 components: {0,1}, {2,3}, {4,5}
        edges = [(0, 1, 1), (2, 3, 2), (4, 5, 3)]
        result = kruskal(6, edges, allow_forest=True)
        assert result.status == Status.FEASIBLE
        assert len(result.solution) == 3
        assert result.objective == 6

    def test_allow_forest_false_default(self):
        edges = [(0, 1, 1)]
        result = kruskal(3, edges)  # allow_forest defaults to False
        assert result.status == Status.INFEASIBLE


class TestPrimBasic:
    def test_simple_triangle(self):
        graph = {0: [(1, 1), (2, 3)], 1: [(0, 1), (2, 2)], 2: [(0, 3), (1, 2)]}
        result = prim(graph)
        assert result.status == Status.OPTIMAL
        assert result.objective == 3
        assert len(result.solution) == 2

    def test_line_graph(self):
        graph = {0: [(1, 1)], 1: [(0, 1), (2, 1)], 2: [(1, 1), (3, 1)], 3: [(2, 1)]}
        result = prim(graph, start=0)
        assert result.status == Status.OPTIMAL
        assert result.objective == 3

    def test_empty_graph(self):
        result = prim({})
        assert result.status == Status.OPTIMAL
        assert result.objective == 0

    def test_single_vertex(self):
        result = prim({0: []})
        assert result.status == Status.OPTIMAL
        assert result.solution == []


class TestPrimDisconnected:
    def test_disconnected_graph(self):
        graph = {0: [(1, 1)], 1: [(0, 1)], 2: [(3, 1)], 3: [(2, 1)]}
        result = prim(graph, start=0)
        assert result.status == Status.INFEASIBLE


class TestBothAlgorithms:
    def test_same_result_square(self):
        edges = [(0, 1, 1), (1, 2, 2), (2, 3, 3), (3, 0, 4), (0, 2, 5), (1, 3, 6)]
        graph = {
            0: [(1, 1), (3, 4), (2, 5)],
            1: [(0, 1), (2, 2), (3, 6)],
            2: [(1, 2), (3, 3), (0, 5)],
            3: [(2, 3), (0, 4), (1, 6)],
        }

        k_result = kruskal(4, edges)
        p_result = prim(graph)

        assert k_result.objective == p_result.objective
        assert k_result.objective == 6

    def test_same_result_complete_5(self):
        import random

        random.seed(42)
        n = 5
        edges = []
        graph = {i: [] for i in range(n)}

        for i in range(n):
            for j in range(i + 1, n):
                w = random.randint(1, 20)
                edges.append((i, j, w))
                graph[i].append((j, w))
                graph[j].append((i, w))

        k_result = kruskal(n, edges)
        p_result = prim(graph)

        assert k_result.objective == p_result.objective


class TestEdgeCases:
    def test_two_vertices(self):
        edges = [(0, 1, 5)]
        result = kruskal(2, edges)
        assert result.objective == 5
        assert len(result.solution) == 1

    def test_parallel_edges(self):
        edges = [(0, 1, 10), (0, 1, 5), (0, 1, 8)]
        result = kruskal(2, edges)
        assert result.objective == 5

    def test_zero_weight(self):
        edges = [(0, 1, 0), (1, 2, 0)]
        result = kruskal(3, edges)
        assert result.objective == 0


class TestStress:
    def test_large_sparse(self):
        import random

        random.seed(42)
        n = 100
        edges = []
        for i in range(n - 1):
            edges.append((i, i + 1, random.randint(1, 10)))
        for _ in range(50):
            i, j = random.sample(range(n), 2)
            edges.append((i, j, random.randint(1, 20)))

        result = kruskal(n, edges)
        assert result.status == Status.OPTIMAL
        assert len(result.solution) == n - 1

    def test_large_dense(self):
        import random

        random.seed(42)
        n = 30
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j, random.randint(1, 100)))

        result = kruskal(n, edges)
        assert result.status == Status.OPTIMAL
        assert len(result.solution) == n - 1

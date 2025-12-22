"""Tests for Floyd-Warshall algorithm."""

from solvor.floyd_warshall import floyd_warshall
from solvor.types import Status


class TestBasic:
    def test_simple_graph(self):
        edges = [(0, 1, 3), (1, 2, 1), (0, 2, 6)]
        result = floyd_warshall(3, edges)
        assert result.status == Status.OPTIMAL
        dist = result.solution
        assert dist[0][0] == 0
        assert dist[0][1] == 3
        assert dist[0][2] == 4
        assert dist[1][2] == 1

    def test_all_pairs(self):
        edges = [(0, 1, 1), (1, 2, 1), (2, 3, 1)]
        result = floyd_warshall(4, edges)
        assert result.status == Status.OPTIMAL
        dist = result.solution
        assert dist[0][3] == 3
        assert dist[1][3] == 2
        assert dist[2][3] == 1

    def test_shortest_through_intermediate(self):
        edges = [(0, 1, 5), (1, 2, 5), (0, 2, 100)]
        result = floyd_warshall(3, edges)
        dist = result.solution
        assert dist[0][2] == 10


class TestUndirected:
    def test_undirected_graph(self):
        edges = [(0, 1, 1), (1, 2, 2)]
        result = floyd_warshall(3, edges, directed=False)
        dist = result.solution
        assert dist[0][1] == 1
        assert dist[1][0] == 1
        assert dist[0][2] == 3
        assert dist[2][0] == 3

    def test_triangle(self):
        edges = [(0, 1, 1), (1, 2, 1), (0, 2, 3)]
        result = floyd_warshall(3, edges, directed=False)
        dist = result.solution
        assert dist[0][2] == 2


class TestNegativeWeights:
    def test_negative_edge(self):
        edges = [(0, 1, 2), (1, 2, -1)]
        result = floyd_warshall(3, edges)
        assert result.status == Status.OPTIMAL
        dist = result.solution
        assert dist[0][2] == 1

    def test_negative_cycle(self):
        edges = [(0, 1, 1), (1, 2, -3), (2, 0, 1)]
        result = floyd_warshall(3, edges)
        assert result.status == Status.UNBOUNDED


class TestDisconnected:
    def test_disconnected_vertices(self):
        edges = [(0, 1, 1)]
        result = floyd_warshall(3, edges)
        dist = result.solution
        assert dist[0][1] == 1
        assert dist[0][2] == float("inf")
        assert dist[2][0] == float("inf")

    def test_isolated_vertex(self):
        edges = [(0, 1, 1), (1, 2, 1)]
        result = floyd_warshall(4, edges)
        dist = result.solution
        assert dist[3][0] == float("inf")
        assert dist[0][3] == float("inf")


class TestEdgeCases:
    def test_single_vertex(self):
        result = floyd_warshall(1, [])
        assert result.status == Status.OPTIMAL
        assert result.solution[0][0] == 0

    def test_two_vertices_no_edge(self):
        result = floyd_warshall(2, [])
        dist = result.solution
        assert dist[0][0] == 0
        assert dist[1][1] == 0
        assert dist[0][1] == float("inf")

    def test_parallel_edges(self):
        edges = [(0, 1, 5), (0, 1, 3)]
        result = floyd_warshall(2, edges)
        assert result.solution[0][1] == 3

    def test_self_loop(self):
        edges = [(0, 0, 1), (0, 1, 2)]
        result = floyd_warshall(2, edges)
        assert result.solution[0][0] == 0
        assert result.solution[0][1] == 2


class TestComplex:
    def test_complete_graph(self):
        n = 4
        edges = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    edges.append((i, j, 1))
        result = floyd_warshall(n, edges)
        dist = result.solution
        for i in range(n):
            for j in range(n):
                if i == j:
                    assert dist[i][j] == 0
                else:
                    assert dist[i][j] == 1

    def test_chain_graph(self):
        n = 5
        edges = [(i, i + 1, i + 1) for i in range(n - 1)]
        result = floyd_warshall(n, edges)
        dist = result.solution
        assert dist[0][4] == 1 + 2 + 3 + 4


class TestStress:
    def test_medium_graph(self):
        n = 30
        edges = []
        for i in range(n - 1):
            edges.append((i, i + 1, 1))
        result = floyd_warshall(n, edges)
        assert result.status == Status.OPTIMAL
        assert result.solution[0][n - 1] == n - 1

    def test_dense_small(self):
        n = 10
        edges = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    edges.append((i, j, abs(i - j)))
        result = floyd_warshall(n, edges)
        assert result.status == Status.OPTIMAL
        for i in range(n):
            for j in range(n):
                assert result.solution[i][j] == abs(i - j)

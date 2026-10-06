"""Tests for articulation points and bridges."""

from solvor.articulation import articulation_points, bridges
from solvor.types import Status


class TestArticulationPointsBasic:
    def test_single_node(self):
        """Single node has no articulation points."""
        result = articulation_points(["A"], lambda n: [])
        assert result.status == Status.OPTIMAL
        assert result.solution == set()

    def test_two_nodes(self):
        """Two connected nodes: neither is articulation point."""
        graph = {"A": ["B"], "B": ["A"]}
        result = articulation_points(graph.keys(), lambda n: graph.get(n, []))
        assert result.solution == set()

    def test_chain_of_three(self):
        """A - B - C: B is articulation point."""
        graph = {"A": ["B"], "B": ["A", "C"], "C": ["B"]}
        result = articulation_points(graph.keys(), lambda n: graph.get(n, []))
        assert result.solution == {"B"}

    def test_triangle(self):
        """Triangle: no articulation points."""
        graph = {
            "A": ["B", "C"],
            "B": ["A", "C"],
            "C": ["A", "B"],
        }
        result = articulation_points(graph.keys(), lambda n: graph.get(n, []))
        assert result.solution == set()

    def test_star(self):
        """Star topology: center is articulation point."""
        graph = {
            "hub": ["A", "B", "C", "D"],
            "A": ["hub"],
            "B": ["hub"],
            "C": ["hub"],
            "D": ["hub"],
        }
        result = articulation_points(graph.keys(), lambda n: graph.get(n, []))
        assert result.solution == {"hub"}


class TestArticulationPointsComplex:
    def test_two_triangles_sharing_vertex(self):
        """Two triangles sharing one vertex: that vertex is articulation point."""
        graph = {
            "A": ["B", "C"],
            "B": ["A", "C"],
            "C": ["A", "B", "D", "E"],  # Bridge vertex
            "D": ["C", "E"],
            "E": ["C", "D"],
        }
        result = articulation_points(graph.keys(), lambda n: graph.get(n, []))
        assert "C" in result.solution

    def test_multiple_articulation_points(self):
        """Chain of triangles has multiple articulation points."""
        graph = {
            "A": ["B", "C"],
            "B": ["A", "C"],
            "C": ["A", "B", "D"],
            "D": ["C", "E", "F"],
            "E": ["D", "F"],
            "F": ["D", "E"],
        }
        result = articulation_points(graph.keys(), lambda n: graph.get(n, []))
        assert "C" in result.solution
        assert "D" in result.solution


class TestBridgesBasic:
    def test_single_node(self):
        """Single node has no bridges."""
        result = bridges(["A"], lambda n: [])
        assert result.status == Status.OPTIMAL
        assert result.solution == []

    def test_two_nodes(self):
        """A - B: the edge is a bridge."""
        graph = {"A": ["B"], "B": ["A"]}
        result = bridges(graph.keys(), lambda n: graph.get(n, []))
        assert len(result.solution) == 1
        assert result.solution[0] in [("A", "B"), ("B", "A")]

    def test_chain_of_three(self):
        """A - B - C: two bridges."""
        graph = {"A": ["B"], "B": ["A", "C"], "C": ["B"]}
        result = bridges(graph.keys(), lambda n: graph.get(n, []))
        assert len(result.solution) == 2

    def test_triangle(self):
        """Triangle: no bridges."""
        graph = {
            "A": ["B", "C"],
            "B": ["A", "C"],
            "C": ["A", "B"],
        }
        result = bridges(graph.keys(), lambda n: graph.get(n, []))
        assert result.solution == []


class TestBridgesComplex:
    def test_two_triangles_connected(self):
        """Two triangles connected by one edge: that edge is a bridge."""
        graph = {
            "A": ["B", "C"],
            "B": ["A", "C"],
            "C": ["A", "B", "D"],
            "D": ["C", "E", "F"],
            "E": ["D", "F"],
            "F": ["D", "E"],
        }
        result = bridges(graph.keys(), lambda n: graph.get(n, []))
        assert len(result.solution) == 1
        bridge = result.solution[0]
        assert set(bridge) == {"C", "D"}


class TestEdgeCases:
    def test_empty_graph_articulation(self):
        result = articulation_points([], lambda n: [])
        assert result.solution == set()

    def test_empty_graph_bridges(self):
        result = bridges([], lambda n: [])
        assert result.solution == []

    def test_disconnected_components(self):
        """Two disconnected pairs."""
        graph = {"A": ["B"], "B": ["A"], "C": ["D"], "D": ["C"]}
        result_ap = articulation_points(graph.keys(), lambda n: graph.get(n, []))
        result_br = bridges(graph.keys(), lambda n: graph.get(n, []))
        assert result_ap.solution == set()
        assert len(result_br.solution) == 2

    def test_numeric_nodes(self):
        """Works with integer nodes."""
        graph = {0: [1], 1: [0, 2], 2: [1]}
        result = articulation_points(graph.keys(), lambda n: graph.get(n, []))
        assert result.solution == {1}


class TestStress:
    def test_large_chain(self):
        """200 node chain: all middle nodes are articulation points."""
        n = 200
        nodes = list(range(n))
        graph = {}
        for i in range(n):
            neighbors = []
            if i > 0:
                neighbors.append(i - 1)
            if i < n - 1:
                neighbors.append(i + 1)
            graph[i] = neighbors

        result = articulation_points(nodes, lambda v: graph.get(v, []))
        # All nodes except endpoints are articulation points
        assert result.solution == set(range(1, n - 1))

    def test_large_cycle(self):
        """200 node cycle: no articulation points or bridges."""
        n = 200
        nodes = list(range(n))
        graph = {i: [(i - 1) % n, (i + 1) % n] for i in range(n)}

        result_ap = articulation_points(nodes, lambda v: graph.get(v, []))
        result_br = bridges(nodes, lambda v: graph.get(v, []))
        assert result_ap.solution == set()
        assert result_br.solution == []


class TestDeepGraphs:
    """DFS depth far beyond Python's default recursion limit (1000)."""

    def test_articulation_points_on_a_long_path(self):
        n = 12_000
        graph = {v: [w for w in (v - 1, v + 1) if 0 <= w < n] for v in range(n)}
        result = articulation_points(range(n), lambda v: graph[v])
        assert result.solution == set(range(1, n - 1))

    def test_bridges_on_a_long_path(self):
        n = 12_000
        graph = {v: [w for w in (v - 1, v + 1) if 0 <= w < n] for v in range(n)}
        result = bridges(range(n), lambda v: graph[v])
        assert sorted(result.solution) == [(v, v + 1) for v in range(n - 1)]

    def test_deep_random_tree(self):
        """In a tree every edge is a bridge and every node of degree 2 or more is a cut vertex."""
        import random

        rnd = random.Random(7)
        n = 100_000
        graph: dict[int, list[int]] = {v: [] for v in range(n)}
        for v in range(1, n):
            parent = rnd.randrange(max(0, v - 50), v)
            graph[v].append(parent)
            graph[parent].append(v)
        assert articulation_points(range(n), lambda v: graph[v]).solution == {v for v in range(n) if len(graph[v]) >= 2}
        assert len(bridges(range(n), lambda v: graph[v]).solution) == n - 1

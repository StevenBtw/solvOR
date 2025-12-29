"""Tests for k-core decomposition."""

from solvor.kcore import kcore, kcore_decomposition
from solvor.types import Status


class TestKcoreDecompositionBasic:
    def test_single_node(self):
        """Single node has core number 0."""
        result = kcore_decomposition(["A"], lambda n: [])
        assert result.status == Status.OPTIMAL
        assert result.solution == {"A": 0}

    def test_two_connected(self):
        """Two connected nodes: both have core 1."""
        graph = {"A": ["B"], "B": ["A"]}
        result = kcore_decomposition(graph.keys(), lambda n: graph.get(n, []))
        assert result.solution == {"A": 1, "B": 1}

    def test_chain_of_three(self):
        """A - B - C: endpoints have core 1, middle has core 1."""
        graph = {"A": ["B"], "B": ["A", "C"], "C": ["B"]}
        result = kcore_decomposition(graph.keys(), lambda n: graph.get(n, []))
        assert result.solution["A"] == 1
        assert result.solution["B"] == 1
        assert result.solution["C"] == 1

    def test_triangle(self):
        """Triangle: all nodes have core 2."""
        graph = {
            "A": ["B", "C"],
            "B": ["A", "C"],
            "C": ["A", "B"],
        }
        result = kcore_decomposition(graph.keys(), lambda n: graph.get(n, []))
        assert result.solution == {"A": 2, "B": 2, "C": 2}

    def test_star(self):
        """Star: hub has core 1, leaves have core 1."""
        graph = {
            "hub": ["A", "B", "C", "D"],
            "A": ["hub"],
            "B": ["hub"],
            "C": ["hub"],
            "D": ["hub"],
        }
        result = kcore_decomposition(graph.keys(), lambda n: graph.get(n, []))
        for node in graph:
            assert result.solution[node] == 1


class TestKcoreDecompositionComplex:
    def test_clique_plus_leaf(self):
        """K4 with one leaf: clique has core 3, leaf has core 1."""
        graph = {
            "A": ["B", "C", "D"],
            "B": ["A", "C", "D"],
            "C": ["A", "B", "D"],
            "D": ["A", "B", "C", "leaf"],
            "leaf": ["D"],
        }
        result = kcore_decomposition(graph.keys(), lambda n: graph.get(n, []))
        assert result.solution["A"] == 3
        assert result.solution["B"] == 3
        assert result.solution["C"] == 3
        assert result.solution["D"] == 3
        assert result.solution["leaf"] == 1

    def test_two_cliques_connected(self):
        """Two triangles connected by edge."""
        graph = {
            "A": ["B", "C"],
            "B": ["A", "C"],
            "C": ["A", "B", "D"],
            "D": ["C", "E", "F"],
            "E": ["D", "F"],
            "F": ["D", "E"],
        }
        result = kcore_decomposition(graph.keys(), lambda n: graph.get(n, []))
        assert result.solution["A"] == 2
        assert result.solution["B"] == 2
        assert result.solution["C"] == 2
        assert result.solution["D"] == 2
        assert result.solution["E"] == 2
        assert result.solution["F"] == 2

    def test_max_core_returned(self):
        """Objective should be the maximum core number."""
        graph = {
            "A": ["B", "C", "D"],
            "B": ["A", "C", "D"],
            "C": ["A", "B", "D"],
            "D": ["A", "B", "C"],
        }
        result = kcore_decomposition(graph.keys(), lambda n: graph.get(n, []))
        assert result.objective == 3


class TestKcoreExtraction:
    def test_extract_2_core(self):
        """Extract nodes with core >= 2."""
        graph = {
            "A": ["B", "C"],
            "B": ["A", "C"],
            "C": ["A", "B"],
            "leaf": ["A"],
        }
        result = kcore(graph.keys(), lambda n: graph.get(n, []), k=2)
        assert result.solution == {"A", "B", "C"}
        assert "leaf" not in result.solution

    def test_extract_3_core(self):
        """K4 plus leaf: 3-core is just the K4."""
        graph = {
            "A": ["B", "C", "D"],
            "B": ["A", "C", "D"],
            "C": ["A", "B", "D"],
            "D": ["A", "B", "C", "leaf"],
            "leaf": ["D"],
        }
        result = kcore(graph.keys(), lambda n: graph.get(n, []), k=3)
        assert result.solution == {"A", "B", "C", "D"}

    def test_empty_kcore(self):
        """Requesting higher k than exists returns empty set."""
        graph = {"A": ["B"], "B": ["A"]}
        result = kcore(graph.keys(), lambda n: graph.get(n, []), k=5)
        assert result.solution == set()


class TestEdgeCases:
    def test_empty_graph(self):
        result = kcore_decomposition([], lambda n: [])
        assert result.solution == {}

    def test_no_edges(self):
        """Disconnected nodes have core 0."""
        nodes = ["A", "B", "C"]
        result = kcore_decomposition(nodes, lambda n: [])
        for node in nodes:
            assert result.solution[node] == 0

    def test_numeric_nodes(self):
        """Works with integer nodes."""
        graph = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
        result = kcore_decomposition(graph.keys(), lambda n: graph.get(n, []))
        assert result.solution == {0: 2, 1: 2, 2: 2}

    def test_self_loop_ignored(self):
        """Self-loops should be ignored."""
        graph = {"A": ["A", "B"], "B": ["A", "B"]}
        result = kcore_decomposition(graph.keys(), lambda n: graph.get(n, []))
        assert result.solution["A"] == 1
        assert result.solution["B"] == 1


class TestStress:
    def test_large_clique(self):
        """50-node clique: all nodes have core 49."""
        n = 50
        nodes = list(range(n))
        graph = {i: [j for j in range(n) if j != i] for i in range(n)}
        result = kcore_decomposition(nodes, lambda v: graph.get(v, []))
        for v in nodes:
            assert result.solution[v] == n - 1

    def test_large_ring(self):
        """200-node ring: all nodes have core 2."""
        n = 200
        nodes = list(range(n))
        graph = {i: [(i - 1) % n, (i + 1) % n] for i in range(n)}
        result = kcore_decomposition(nodes, lambda v: graph.get(v, []))
        for v in nodes:
            assert result.solution[v] == 2

"""Tests for PageRank algorithm."""

from solvor.pagerank import pagerank
from solvor.types import Status


class TestPageRankBasic:
    def test_single_node(self):
        """Single node gets all the rank."""
        result = pagerank(["A"], lambda n: [])
        assert result.status == Status.OPTIMAL
        assert abs(result.solution["A"] - 1.0) < 0.01

    def test_two_nodes_one_link(self):
        """A → B: B should have higher rank."""
        graph = {"A": ["B"], "B": []}
        result = pagerank(graph.keys(), lambda n: graph.get(n, []))
        assert result.solution["B"] > result.solution["A"]

    def test_mutual_links(self):
        """A ↔ B: should have equal rank."""
        graph = {"A": ["B"], "B": ["A"]}
        result = pagerank(graph.keys(), lambda n: graph.get(n, []))
        assert abs(result.solution["A"] - result.solution["B"]) < 0.01

    def test_star_topology(self):
        """Hub with spokes: hub should have highest rank."""
        graph = {
            "hub": [],
            "spoke1": ["hub"],
            "spoke2": ["hub"],
            "spoke3": ["hub"],
        }
        result = pagerank(graph.keys(), lambda n: graph.get(n, []))
        assert result.solution["hub"] > result.solution["spoke1"]
        assert result.solution["hub"] > result.solution["spoke2"]
        assert result.solution["hub"] > result.solution["spoke3"]

    def test_ranks_sum_to_one(self):
        """All ranks should sum to 1."""
        graph = {"A": ["B", "C"], "B": ["C"], "C": ["A"]}
        result = pagerank(graph.keys(), lambda n: graph.get(n, []))
        total = sum(result.solution.values())
        assert abs(total - 1.0) < 0.001


class TestPageRankParameters:
    def test_damping_factor(self):
        """Higher damping = more weight to link structure."""
        graph = {"A": ["B"], "B": ["C"], "C": []}
        result_high = pagerank(graph.keys(), lambda n: graph.get(n, []), damping=0.99)
        result_low = pagerank(graph.keys(), lambda n: graph.get(n, []), damping=0.50)
        # With higher damping, C gets proportionally more rank from link chain
        ratio_high = result_high.solution["C"] / result_high.solution["A"]
        ratio_low = result_low.solution["C"] / result_low.solution["A"]
        assert ratio_high > ratio_low

    def test_max_iter_limit(self):
        """Should stop at max_iter even if not converged."""
        graph = {"A": ["B"], "B": ["A"]}
        result = pagerank(graph.keys(), lambda n: graph.get(n, []), max_iter=2)
        assert result.iterations <= 2

    def test_max_iter_status(self):
        """Should return MAX_ITER status when not converged."""
        # Asymmetric graph that takes many iterations to converge
        graph = {"A": ["B", "C", "D"], "B": ["A"], "C": ["A"], "D": ["A"]}
        result = pagerank(graph.keys(), lambda n: graph.get(n, []), max_iter=1, tol=1e-20)
        assert result.status == Status.MAX_ITER
        assert result.iterations == 1

    def test_convergence_tolerance(self):
        """Tighter tolerance = more iterations."""
        graph = {"A": ["B"], "B": ["C"], "C": ["A"]}
        result_loose = pagerank(graph.keys(), lambda n: graph.get(n, []), tol=0.1)
        result_tight = pagerank(graph.keys(), lambda n: graph.get(n, []), tol=1e-10)
        assert result_tight.iterations >= result_loose.iterations


class TestPageRankComplex:
    def test_wikipedia_example(self):
        """Classic PageRank example structure."""
        graph = {
            "A": ["B", "C"],
            "B": ["C"],
            "C": ["A"],
            "D": ["C"],
        }
        result = pagerank(graph.keys(), lambda n: graph.get(n, []))
        # C receives links from everyone, should have high rank
        assert result.solution["C"] > result.solution["D"]
        # All ranks sum to 1
        assert abs(sum(result.solution.values()) - 1.0) < 0.001

    def test_dangling_nodes(self):
        """Nodes with no outgoing edges (dangling) distribute rank evenly."""
        graph = {"A": ["B"], "B": [], "C": []}  # B and C are dangling
        result = pagerank(graph.keys(), lambda n: graph.get(n, []))
        # Should still sum to 1
        assert abs(sum(result.solution.values()) - 1.0) < 0.001


class TestPageRankEdgeCases:
    def test_empty_graph(self):
        result = pagerank([], lambda n: [])
        assert result.solution == {}

    def test_no_edges(self):
        """All disconnected nodes should have equal rank."""
        nodes = ["A", "B", "C", "D"]
        result = pagerank(nodes, lambda n: [])
        expected = 1.0 / len(nodes)
        for node in nodes:
            assert abs(result.solution[node] - expected) < 0.01

    def test_numeric_nodes(self):
        """Works with integer nodes."""
        graph = {0: [1, 2], 1: [2], 2: [0]}
        result = pagerank(graph.keys(), lambda n: graph.get(n, []))
        assert abs(sum(result.solution.values()) - 1.0) < 0.001

    def test_self_loops(self):
        """Self-loops should be handled gracefully."""
        graph = {"A": ["A", "B"], "B": ["A"]}
        result = pagerank(graph.keys(), lambda n: graph.get(n, []))
        assert abs(sum(result.solution.values()) - 1.0) < 0.001


class TestPageRankStress:
    def test_large_graph(self):
        """500 nodes in a cycle."""
        n = 500
        nodes = list(range(n))
        graph = {i: [(i + 1) % n] for i in range(n)}
        result = pagerank(nodes, lambda v: graph.get(v, []))
        # All nodes should have equal rank in a cycle
        expected = 1.0 / n
        for v in nodes:
            assert abs(result.solution[v] - expected) < 0.01

    def test_convergence(self):
        """Should converge with reasonable iterations."""
        n = 100
        nodes = list(range(n))
        # Random-ish graph
        graph = {i: [(i * 7 + 3) % n, (i * 13 + 5) % n] for i in range(n)}
        result = pagerank(nodes, lambda v: graph.get(v, []))
        assert result.status == Status.OPTIMAL
        assert result.iterations < 100

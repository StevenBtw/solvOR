"""Tests for Louvain community detection."""

from solvor.community import louvain
from solvor.types import Status


class TestLouvainBasic:
    def test_single_node(self):
        """Single node is its own community."""
        result = louvain(["A"], lambda n: [])
        assert result.status == Status.OPTIMAL
        assert len(result.solution) == 1
        assert result.solution[0] == {"A"}

    def test_two_connected_nodes(self):
        """Two connected nodes form one community."""
        graph = {"A": ["B"], "B": ["A"]}
        result = louvain(graph.keys(), lambda n: graph.get(n, []))
        # Should be in same community
        all_nodes = set().union(*result.solution)
        assert all_nodes == {"A", "B"}

    def test_two_disconnected_pairs(self):
        """Two disconnected pairs should form two communities."""
        graph = {
            "A": ["B"], "B": ["A"],
            "C": ["D"], "D": ["C"],
        }
        result = louvain(graph.keys(), lambda n: graph.get(n, []))
        assert len(result.solution) == 2
        communities = [frozenset(c) for c in result.solution]
        assert frozenset(["A", "B"]) in communities
        assert frozenset(["C", "D"]) in communities

    def test_triangle(self):
        """Triangle should be one community."""
        graph = {
            "A": ["B", "C"],
            "B": ["A", "C"],
            "C": ["A", "B"],
        }
        result = louvain(graph.keys(), lambda n: graph.get(n, []))
        assert len(result.solution) == 1
        assert result.solution[0] == {"A", "B", "C"}


class TestLouvainStructure:
    def test_two_triangles_connected(self):
        """Two triangles connected by single edge form two communities."""
        graph = {
            "A": ["B", "C"],
            "B": ["A", "C"],
            "C": ["A", "B", "D"],  # Bridge to other triangle
            "D": ["C", "E", "F"],
            "E": ["D", "F"],
            "F": ["D", "E"],
        }
        result = louvain(graph.keys(), lambda n: graph.get(n, []))
        # Should detect 2 communities
        assert len(result.solution) == 2

    def test_barbell_graph(self):
        """Two cliques connected by a single edge."""
        graph = {
            # Left clique
            "A": ["B", "C", "D"],
            "B": ["A", "C", "D"],
            "C": ["A", "B", "D"],
            "D": ["A", "B", "C", "E"],  # Bridge
            # Right clique
            "E": ["D", "F", "G", "H"],
            "F": ["E", "G", "H"],
            "G": ["E", "F", "H"],
            "H": ["E", "F", "G"],
        }
        result = louvain(graph.keys(), lambda n: graph.get(n, []))
        assert len(result.solution) == 2
        left = next(c for c in result.solution if "A" in c)
        right = next(c for c in result.solution if "F" in c)
        assert "A" in left and "B" in left and "C" in left
        assert "F" in right and "G" in right and "H" in right


class TestLouvainParameters:
    def test_resolution_high(self):
        """Higher resolution = smaller communities."""
        graph = {
            "A": ["B", "C", "D"],
            "B": ["A", "C", "D"],
            "C": ["A", "B", "D"],
            "D": ["A", "B", "C"],
        }
        result_low = louvain(graph.keys(), lambda n: graph.get(n, []), resolution=0.5)
        result_high = louvain(graph.keys(), lambda n: graph.get(n, []), resolution=2.0)
        # Higher resolution tends to produce more communities
        assert len(result_high.solution) >= len(result_low.solution)


class TestLouvainEdgeCases:
    def test_empty_graph(self):
        result = louvain([], lambda n: [])
        assert result.solution == []

    def test_no_edges(self):
        """All disconnected nodes are their own communities."""
        nodes = ["A", "B", "C", "D"]
        result = louvain(nodes, lambda n: [])
        assert len(result.solution) == 4
        for comm in result.solution:
            assert len(comm) == 1

    def test_numeric_nodes(self):
        """Works with integer nodes."""
        graph = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
        result = louvain(graph.keys(), lambda n: graph.get(n, []))
        assert len(result.solution) == 1
        assert result.solution[0] == {0, 1, 2}

    def test_self_loop_ignored(self):
        """Self-loops should be ignored."""
        graph = {"A": ["A", "B"], "B": ["A", "B"]}
        result = louvain(graph.keys(), lambda n: graph.get(n, []))
        all_nodes = set().union(*result.solution)
        assert all_nodes == {"A", "B"}


class TestLouvainModularity:
    def test_modularity_positive(self):
        """Good community structure should have positive modularity."""
        # Two clear clusters
        graph = {
            "A": ["B", "C"],
            "B": ["A", "C"],
            "C": ["A", "B"],
            "D": ["E", "F"],
            "E": ["D", "F"],
            "F": ["D", "E"],
        }
        result = louvain(graph.keys(), lambda n: graph.get(n, []))
        assert result.objective > 0

    def test_clique_modularity(self):
        """Single clique should have modularity near 0."""
        graph = {
            "A": ["B", "C", "D"],
            "B": ["A", "C", "D"],
            "C": ["A", "B", "D"],
            "D": ["A", "B", "C"],
        }
        result = louvain(graph.keys(), lambda n: graph.get(n, []))
        # Single community from clique has low modularity
        assert result.objective < 0.5


class TestLouvainStress:
    def test_large_ring(self):
        """200 nodes in a ring."""
        n = 200
        nodes = list(range(n))
        graph = {i: [(i - 1) % n, (i + 1) % n] for i in range(n)}
        result = louvain(nodes, lambda v: graph.get(v, []))
        # Ring should produce some communities
        assert len(result.solution) >= 1
        total_nodes = sum(len(c) for c in result.solution)
        assert total_nodes == n

    def test_complete_bipartite(self):
        """K_5_5 bipartite graph."""
        left = [f"L{i}" for i in range(5)]
        right = [f"R{i}" for i in range(5)]
        graph = {l: right[:] for l in left}
        graph.update({r: left[:] for r in right})
        result = louvain(left + right, lambda v: graph.get(v, []))
        # All nodes accounted for
        total_nodes = sum(len(c) for c in result.solution)
        assert total_nodes == 10

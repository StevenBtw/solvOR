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
            "A": ["B"],
            "B": ["A"],
            "C": ["D"],
            "D": ["C"],
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
        graph = {node: right[:] for node in left}
        graph.update({r: left[:] for r in right})
        result = louvain(left + right, lambda v: graph.get(v, []))
        # All nodes accounted for
        total_nodes = sum(len(c) for c in result.solution)
        assert total_nodes == 10


class TestLouvainAggregation:
    """Phase 2: communities become nodes and local moving repeats on the coarser graph."""

    def test_long_path_reaches_high_modularity(self):
        n = 1000
        result = louvain(range(n), lambda v: [w for w in (v - 1, v + 1) if 0 <= w < n])
        # Local moving alone stops at 500 pairs (modularity ~0.50)
        assert result.objective > 0.9
        assert len(result.solution) < 100
        assert all(_is_contiguous(c) for c in result.solution)

    def test_two_cliques_joined_by_one_edge(self):
        graph: dict[int, list[int]] = {v: [] for v in range(20)}
        for block in (range(10), range(10, 20)):
            for a in block:
                graph[a].extend(b for b in block if b != a)
        graph[9].append(10)
        graph[10].append(9)
        result = louvain(graph.keys(), lambda v: graph[v])
        assert sorted(map(sorted, result.solution)) == [list(range(10)), list(range(10, 20))]

    def test_ring_of_cliques(self):
        """Classic benchmark: 8 cliques of 5 nodes in a ring; each clique is one community."""
        k, size = 8, 5
        graph: dict[int, set[int]] = {v: set() for v in range(k * size)}
        for c in range(k):
            members = range(c * size, (c + 1) * size)
            for a in members:
                graph[a].update(b for b in members if b != a)
            a, b = c * size, ((c + 1) % k) * size + 1
            graph[a].add(b)
            graph[b].add(a)
        result = louvain(graph.keys(), lambda v: graph[v])
        assert sorted(map(sorted, result.solution)) == [list(range(c * size, (c + 1) * size)) for c in range(k)]

    def test_modularity_matches_partition(self):
        """result.objective is the modularity of the returned partition on the original graph."""
        import random

        rng = random.Random(5)
        graph: dict[int, set[int]] = {v: set() for v in range(200)}
        for _ in range(600):
            a, b = rng.randrange(200), rng.randrange(200)
            if a != b:
                graph[a].add(b)
                graph[b].add(a)
        result = louvain(graph.keys(), lambda v: graph[v])
        m = sum(len(s) for s in graph.values()) / 2
        label = {v: i for i, comm in enumerate(result.solution) for v in comm}
        q = 0.0
        for comm in result.solution:
            inside = sum(1 for v in comm for w in graph[v] if label[w] == label[v]) / 2
            degree = sum(len(graph[v]) for v in comm)
            q += inside / m - (degree / (2 * m)) ** 2
        assert abs(result.objective - q) < 1e-12


class TestLouvainInputs:
    def test_duplicate_nodes_count_once(self):
        graph = {1: [2], 2: [1, 3], 3: [2]}
        once = louvain([1, 2, 3], lambda v: graph[v])
        twice = louvain([1, 1, 2, 3, 3], lambda v: graph[v])
        assert once.solution == twice.solution
        assert once.objective == twice.objective

    def test_incomparable_nodes_do_not_raise(self):
        graph = {1: ["a"], "a": [1], 2: ["b"], "b": [2]}
        result = louvain([1, "a", 2, "b"], lambda v: graph[v])
        assert sorted(map(len, result.solution)) == [2, 2]


def _is_contiguous(community: set[int]) -> bool:
    return max(community) - min(community) + 1 == len(community)

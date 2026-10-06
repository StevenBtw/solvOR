"""Tests for PageRank algorithm."""

import pytest

from solvor.pagerank import pagerank, pagerank_edges
from solvor.rust import rust_available
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


class TestPageRankEdges:
    """Tests for edge-list PageRank variant."""

    def test_simple(self):
        edges = [(0, 1), (1, 2), (2, 0)]
        result = pagerank_edges(3, edges)
        # Cycle should have equal ranks
        assert abs(result.solution[0] - result.solution[1]) < 0.01

    def test_star(self):
        # All point to node 0
        edges = [(1, 0), (2, 0), (3, 0)]
        result = pagerank_edges(4, edges)
        assert result.solution[0] > result.solution[1]

    def test_sums_to_one(self):
        edges = [(0, 1), (1, 2), (2, 0), (0, 2)]
        result = pagerank_edges(3, edges)
        assert abs(sum(result.solution.values()) - 1.0) < 0.01


class TestPageRankEdgesPython:
    """Test Python backend explicitly."""

    def test_simple_python(self):
        edges = [(0, 1), (1, 2), (2, 0)]
        result = pagerank_edges(3, edges, backend="python")
        assert abs(sum(result.solution.values()) - 1.0) < 0.01

    def test_star_python(self):
        edges = [(1, 0), (2, 0), (3, 0)]
        result = pagerank_edges(4, edges, backend="python")
        assert result.solution[0] > result.solution[1]


class TestPageRankEdgesRust:
    """Test Rust backend explicitly."""

    @pytest.fixture(autouse=True)
    def require_rust(self):
        from solvor.rust import rust_available

        if not rust_available():
            pytest.skip("Rust backend not available")

    def test_simple_rust(self):
        edges = [(0, 1), (1, 2), (2, 0)]
        result = pagerank_edges(3, edges, backend="rust")
        assert abs(sum(result.solution.values()) - 1.0) < 0.01

    def test_star_rust(self):
        edges = [(1, 0), (2, 0), (3, 0)]
        result = pagerank_edges(4, edges, backend="rust")
        assert result.solution[0] > result.solution[1]

    def test_large_graph_rust(self):
        """Test Rust handles larger graphs correctly."""
        n = 100
        edges = [(i, (i + 1) % n) for i in range(n)]
        result = pagerank_edges(n, edges, backend="rust")
        assert abs(sum(result.solution.values()) - 1.0) < 0.01


class TestPageRankBackendsAgree:
    """pagerank() routes through pagerank_edges(); both backends must agree."""

    @pytest.fixture(autouse=True)
    def require_rust(self):
        from solvor.rust import rust_available

        if not rust_available():
            pytest.skip("Rust backend not available")

    def test_random_graphs(self):
        import random

        rng = random.Random(3)
        for _ in range(100):
            n = rng.randint(1, 400)
            edges = [(rng.randrange(n), rng.randrange(n)) for _ in range(rng.randint(0, 5 * n))]
            for damping in (0.85, 0.5):
                py = pagerank_edges(n, edges, damping=damping, backend="python")
                rs = pagerank_edges(n, edges, damping=damping, backend="rust")
                # Bit-identical, not just close: downstream caches key on these values
                assert py.solution == rs.solution
                assert py.iterations == rs.iterations
                assert py.objective == rs.objective
                assert py.status == rs.status

    def test_callback_default_backend_matches_python(self):
        graph = {"a": ["b", "c"], "b": ["c"], "c": ["a"], "d": ["c"]}
        auto = pagerank(graph.keys(), lambda v: graph[v])
        python = pagerank(graph.keys(), lambda v: graph[v], backend="python")
        assert auto.solution == python.solution
        assert auto.iterations == python.iterations


class TestPageRankCallbackMapping:
    def test_callback_api_matches_edge_api(self):
        graph = {"a": ["b", "c"], "b": ["c"], "c": ["a"], "d": ["c"]}
        result = pagerank(graph.keys(), lambda v: graph[v], backend="python")
        edges = [(0, 1), (0, 2), (1, 2), (2, 0), (3, 2)]
        expected = pagerank_edges(4, edges, backend="python")
        assert [result.solution[v] for v in "abcd"] == [expected.solution[i] for i in range(4)]

    def test_out_of_range_edges_are_ignored(self):
        with_bad = pagerank_edges(3, [(0, 1), (1, 2), (2, 0), (5, 1), (1, -1)], backend="python")
        clean = pagerank_edges(3, [(0, 1), (1, 2), (2, 0)], backend="python")
        assert with_bad.solution == clean.solution


class TestPageRankValidation:
    @pytest.mark.parametrize("backend", ["python", "auto"])
    def test_rejects_bad_damping(self, backend):
        with pytest.raises(ValueError, match="damping"):
            pagerank(["a"], lambda v: [], damping=1.0, backend=backend)

    @pytest.mark.parametrize("backend", ["python", "auto"])
    def test_rejects_zero_max_iter(self, backend):
        with pytest.raises(ValueError, match="max_iter must be positive"):
            pagerank(["a"], lambda v: [], max_iter=0, backend=backend)

    def test_rejects_non_positive_tol(self):
        with pytest.raises(ValueError, match="tol must be positive"):
            pagerank_edges(1, [], tol=0.0, backend="python")


_BACKENDS = ["python", "rust"] if rust_available() else ["python"]


class TestPageRankInvalidInputParity:
    """Invalid input raises the same exception type and message on every backend."""

    @pytest.mark.parametrize("backend", _BACKENDS)
    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"damping": 1.0}, "damping 1.0 must be in [0, 1)"),
            ({"damping": -0.5}, "damping -0.5 must be in [0, 1)"),
            ({"max_iter": 0}, "max_iter must be positive"),
            ({"max_iter": -1}, "max_iter must be positive"),
            ({"tol": 0.0}, "tol must be positive"),
        ],
    )
    def test_bad_parameters(self, backend, kwargs, message):
        with pytest.raises(ValueError) as excinfo:
            pagerank_edges(3, [(0, 1)], backend=backend, **kwargs)
        assert str(excinfo.value) == message

    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_negative_node_count(self, backend):
        with pytest.raises(ValueError) as excinfo:
            pagerank_edges(-1, [], backend=backend)
        assert str(excinfo.value) == "n_nodes must be non-negative"

    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_negative_and_out_of_range_endpoints_are_ignored(self, backend):
        dirty = pagerank_edges(
            3, [(0, 1), (1, 2), (2, 0), (5, 1), (1, -1), (-2, 0), (2**64, 0), (0, -(2**70))], backend=backend
        )
        clean = pagerank_edges(3, [(0, 1), (1, 2), (2, 0)], backend="python")
        assert dirty.solution == clean.solution

"""Tests for SCC and topological sort."""

from solvor.scc import condense, strongly_connected_components, topological_sort
from solvor.types import Status


class TestTopologicalSortBasic:
    def test_linear_chain(self):
        """A → B → C should sort to [A, B, C]."""
        graph = {"A": ["B"], "B": ["C"], "C": []}
        result = topological_sort(graph.keys(), lambda n: graph.get(n, []))
        assert result.status == Status.OPTIMAL
        assert result.solution == ["A", "B", "C"]

    def test_diamond(self):
        """A → B, A → C, B → D, C → D."""
        graph = {"A": ["B", "C"], "B": ["D"], "C": ["D"], "D": []}
        result = topological_sort(graph.keys(), lambda n: graph.get(n, []))
        assert result.status == Status.OPTIMAL
        sol = result.solution
        assert sol.index("A") < sol.index("B")
        assert sol.index("A") < sol.index("C")
        assert sol.index("B") < sol.index("D")
        assert sol.index("C") < sol.index("D")

    def test_multiple_sources(self):
        """A → C, B → C (A and B are both sources)."""
        graph = {"A": ["C"], "B": ["C"], "C": []}
        result = topological_sort(graph.keys(), lambda n: graph.get(n, []))
        assert result.status == Status.OPTIMAL
        sol = result.solution
        assert sol.index("A") < sol.index("C")
        assert sol.index("B") < sol.index("C")

    def test_single_node(self):
        result = topological_sort(["A"], lambda n: [])
        assert result.status == Status.OPTIMAL
        assert result.solution == ["A"]

    def test_disconnected(self):
        """Two separate chains: A → B and C → D."""
        graph = {"A": ["B"], "B": [], "C": ["D"], "D": []}
        result = topological_sort(graph.keys(), lambda n: graph.get(n, []))
        assert result.status == Status.OPTIMAL
        sol = result.solution
        assert sol.index("A") < sol.index("B")
        assert sol.index("C") < sol.index("D")


class TestTopologicalSortCycles:
    def test_simple_cycle(self):
        """A → B → C → A should fail."""
        graph = {"A": ["B"], "B": ["C"], "C": ["A"]}
        result = topological_sort(graph.keys(), lambda n: graph.get(n, []))
        assert result.status == Status.INFEASIBLE
        assert result.solution is None

    def test_self_loop(self):
        """A → A should fail."""
        graph = {"A": ["A"]}
        result = topological_sort(graph.keys(), lambda n: graph.get(n, []))
        assert result.status == Status.INFEASIBLE

    def test_cycle_in_larger_graph(self):
        """A → B → C, C → B creates cycle but A is reachable."""
        graph = {"A": ["B"], "B": ["C"], "C": ["B"]}
        result = topological_sort(graph.keys(), lambda n: graph.get(n, []))
        assert result.status == Status.INFEASIBLE


class TestSCCBasic:
    def test_linear_chain(self):
        """A → B → C: each node is its own SCC."""
        graph = {"A": ["B"], "B": ["C"], "C": []}
        result = strongly_connected_components(graph.keys(), lambda n: graph.get(n, []))
        assert result.status == Status.OPTIMAL
        assert result.objective == 3  # 3 components
        # Each component should be a single node
        assert all(len(c) == 1 for c in result.solution)

    def test_single_cycle(self):
        """A → B → C → A: all in one SCC."""
        graph = {"A": ["B"], "B": ["C"], "C": ["A"]}
        result = strongly_connected_components(graph.keys(), lambda n: graph.get(n, []))
        assert result.status == Status.OPTIMAL
        assert result.objective == 1
        assert set(result.solution[0]) == {"A", "B", "C"}

    def test_two_sccs(self):
        """A → B → A, C → D → C with B → C connecting them."""
        graph = {"A": ["B"], "B": ["A", "C"], "C": ["D"], "D": ["C"]}
        result = strongly_connected_components(graph.keys(), lambda n: graph.get(n, []))
        assert result.status == Status.OPTIMAL
        assert result.objective == 2
        sccs = [set(c) for c in result.solution]
        assert {"A", "B"} in sccs
        assert {"C", "D"} in sccs

    def test_self_loop(self):
        """A → A is an SCC."""
        graph = {"A": ["A"]}
        result = strongly_connected_components(graph.keys(), lambda n: graph.get(n, []))
        assert result.objective == 1
        assert result.solution[0] == ["A"]

    def test_single_node_no_edges(self):
        result = strongly_connected_components(["A"], lambda n: [])
        assert result.objective == 1
        assert result.solution[0] == ["A"]


class TestSCCComplex:
    def test_wikipedia_example(self):
        """Classic SCC example with 8 nodes, 3 SCCs."""
        # Graph with known SCC structure
        graph = {
            "a": ["b"],
            "b": ["c", "e", "f"],
            "c": ["d", "g"],
            "d": ["c", "h"],
            "e": ["a", "f"],
            "f": ["g"],
            "g": ["f"],
            "h": ["d", "g"],
        }
        result = strongly_connected_components(graph.keys(), lambda n: graph.get(n, []))
        sccs = [set(c) for c in result.solution]
        # a→b→e→a forms a cycle
        assert {"a", "b", "e"} in sccs
        # c↔d and d↔h forms a cycle
        assert {"c", "d", "h"} in sccs
        # f↔g forms a cycle
        assert {"f", "g"} in sccs
        assert result.objective == 3

    def test_reverse_topological_order(self):
        """SCCs are returned in reverse topological order (sinks first)."""
        # A → B → C (linear, no cycles)
        graph = {"A": ["B"], "B": ["C"], "C": []}
        result = strongly_connected_components(graph.keys(), lambda n: graph.get(n, []))
        # Should be [["C"], ["B"], ["A"]] - sinks first
        flat = [c[0] for c in result.solution]
        assert flat.index("C") < flat.index("B") < flat.index("A")


class TestCondense:
    def test_cycle_becomes_single_node(self):
        """A → B → C → A becomes one condensed node."""
        graph = {"A": ["B"], "B": ["C"], "C": ["A"]}
        result = condense(graph.keys(), lambda n: graph.get(n, []))
        condensed_nodes, adjacency = result.solution
        assert len(condensed_nodes) == 1
        assert frozenset(["A", "B", "C"]) in condensed_nodes

    def test_two_sccs_with_edge(self):
        """SCC1 → SCC2 becomes edge in DAG."""
        graph = {"A": ["B"], "B": ["A", "C"], "C": ["D"], "D": ["C"]}
        result = condense(graph.keys(), lambda n: graph.get(n, []))
        condensed_nodes, adjacency = result.solution
        assert len(condensed_nodes) == 2

        # Find which condensed node contains A (should point to C's component)
        ab_node = next(n for n in condensed_nodes if "A" in n)
        cd_node = next(n for n in condensed_nodes if "C" in n)
        assert cd_node in adjacency[ab_node]

    def test_dag_unchanged(self):
        """DAG without cycles: each node becomes its own condensed node."""
        graph = {"A": ["B"], "B": ["C"], "C": []}
        result = condense(graph.keys(), lambda n: graph.get(n, []))
        condensed_nodes, adjacency = result.solution
        assert len(condensed_nodes) == 3
        for node in condensed_nodes:
            assert len(node) == 1


class TestEdgeCases:
    def test_empty_graph(self):
        result = topological_sort([], lambda n: [])
        assert result.status == Status.OPTIMAL
        assert result.solution == []

        result = strongly_connected_components([], lambda n: [])
        assert result.solution == []
        assert result.objective == 0

    def test_numeric_nodes(self):
        """Works with integer nodes."""
        graph = {0: [1], 1: [2], 2: [0]}
        result = strongly_connected_components(graph.keys(), lambda n: graph.get(n, []))
        assert result.objective == 1
        assert set(result.solution[0]) == {0, 1, 2}

    def test_tuple_nodes(self):
        """Works with tuple nodes (like grid coordinates)."""
        graph = {(0, 0): [(0, 1)], (0, 1): [(1, 1)], (1, 1): [(0, 0)]}
        result = strongly_connected_components(graph.keys(), lambda n: graph.get(n, []))
        assert result.objective == 1


class TestStress:
    def test_large_dag(self):
        """1000 node linear chain."""
        n = 1000
        nodes = list(range(n))
        graph = {i: [i + 1] for i in range(n - 1)}
        graph[n - 1] = []

        result = topological_sort(nodes, lambda v: graph.get(v, []))
        assert result.status == Status.OPTIMAL
        assert result.solution == list(range(n))

    def test_large_single_scc(self):
        """500 node cycle."""
        n = 500
        nodes = list(range(n))
        graph = {i: [(i + 1) % n] for i in range(n)}

        result = strongly_connected_components(nodes, lambda v: graph.get(v, []))
        assert result.objective == 1
        assert len(result.solution[0]) == n

    def test_many_small_sccs(self):
        """100 separate 2-cycles."""
        nodes = list(range(200))
        graph = {}
        for i in range(0, 200, 2):
            graph[i] = [i + 1]
            graph[i + 1] = [i]

        result = strongly_connected_components(nodes, lambda v: graph.get(v, []))
        assert result.objective == 100
        for scc in result.solution:
            assert len(scc) == 2

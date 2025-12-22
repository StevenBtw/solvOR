"""Tests for Network Simplex algorithm."""

from solvor.network_simplex import network_simplex
from solvor.types import Status


class TestBasic:
    def test_simple_flow(self):
        arcs = [(0, 1, 10, 1)]
        supplies = [5, -5]
        result = network_simplex(2, arcs, supplies)
        assert result.status == Status.OPTIMAL
        assert result.objective == 5

    def test_two_paths(self):
        arcs = [(0, 1, 10, 2), (0, 2, 10, 1), (1, 3, 10, 1), (2, 3, 10, 2)]
        supplies = [5, 0, 0, -5]
        result = network_simplex(4, arcs, supplies)
        assert result.status == Status.OPTIMAL
        assert result.objective == 15

    def test_choose_cheaper_path(self):
        arcs = [(0, 1, 10, 5), (0, 2, 10, 1), (1, 3, 10, 1), (2, 3, 10, 1)]
        supplies = [5, 0, 0, -5]
        result = network_simplex(4, arcs, supplies)
        assert result.status == Status.OPTIMAL
        assert result.objective == 10


class TestInfeasible:
    def test_unbalanced_supply(self):
        arcs = [(0, 1, 10, 1)]
        supplies = [5, -3]
        result = network_simplex(2, arcs, supplies)
        assert result.status == Status.INFEASIBLE

    def test_insufficient_capacity(self):
        arcs = [(0, 1, 3, 1)]
        supplies = [5, -5]
        result = network_simplex(2, arcs, supplies)
        assert result.status == Status.INFEASIBLE

    def test_no_path(self):
        arcs = [(0, 1, 10, 1)]
        supplies = [5, 0, -5]
        result = network_simplex(3, arcs, supplies)
        assert result.status == Status.INFEASIBLE


class TestTransshipment:
    def test_transshipment_node(self):
        arcs = [(0, 1, 10, 1), (1, 2, 10, 1)]
        supplies = [5, 0, -5]
        result = network_simplex(3, arcs, supplies)
        assert result.status == Status.OPTIMAL
        assert result.objective == 10

    def test_multiple_transshipment(self):
        arcs = [(0, 1, 10, 1), (1, 2, 10, 1), (2, 3, 10, 1)]
        supplies = [5, 0, 0, -5]
        result = network_simplex(4, arcs, supplies)
        assert result.status == Status.OPTIMAL
        assert result.objective == 15


class TestMultipleSourcesSinks:
    def test_two_sources(self):
        arcs = [(0, 2, 10, 1), (1, 2, 10, 1)]
        supplies = [3, 2, -5]
        result = network_simplex(3, arcs, supplies)
        assert result.status == Status.OPTIMAL
        assert result.objective == 5

    def test_two_sinks(self):
        arcs = [(0, 1, 10, 1), (0, 2, 10, 2)]
        supplies = [5, -3, -2]
        result = network_simplex(3, arcs, supplies)
        assert result.status == Status.OPTIMAL
        assert result.objective == 7


class TestEdgeCases:
    def test_zero_flow(self):
        arcs = [(0, 1, 10, 1)]
        supplies = [0, 0]
        result = network_simplex(2, arcs, supplies)
        assert result.status == Status.OPTIMAL
        assert result.objective == 0

    def test_no_arcs_zero_supply(self):
        supplies = [0, 0]
        result = network_simplex(2, [], supplies)
        assert result.status == Status.OPTIMAL

    def test_single_node(self):
        result = network_simplex(1, [], [0])
        assert result.status == Status.OPTIMAL

    def test_zero_cost(self):
        arcs = [(0, 1, 10, 0)]
        supplies = [5, -5]
        result = network_simplex(2, arcs, supplies)
        assert result.status == Status.OPTIMAL
        assert result.objective == 0


class TestStress:
    def test_chain(self):
        n = 20
        arcs = [(i, i + 1, 100, 1) for i in range(n - 1)]
        supplies = [10] + [0] * (n - 2) + [-10]
        result = network_simplex(n, arcs, supplies)
        assert result.status == Status.OPTIMAL
        assert result.objective == 10 * (n - 1)

    def test_parallel_arcs(self):
        arcs = [(0, 1, 5, 2), (0, 1, 5, 3), (0, 1, 5, 1)]
        supplies = [10, -10]
        result = network_simplex(2, arcs, supplies)
        assert result.status == Status.OPTIMAL
        # Should use cheapest arcs first: 5*1 + 5*2 = 15
        assert result.objective == 15

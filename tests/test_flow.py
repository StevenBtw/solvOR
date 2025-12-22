"""Tests for the network flow solvers."""

from solvor.flow import max_flow, min_cost_flow, solve_assignment
from solvor.types import Status


class TestMaxFlowBasic:
    def test_single_edge(self):
        graph = {"s": [("t", 10, 0)], "t": []}
        result = max_flow(graph, "s", "t")
        assert result.status == Status.OPTIMAL
        assert result.objective == 10

    def test_two_paths(self):
        graph = {"s": [("a", 10, 0), ("b", 5, 0)], "a": [("t", 10, 0)], "b": [("t", 10, 0)], "t": []}
        result = max_flow(graph, "s", "t")
        assert result.status == Status.OPTIMAL
        assert result.objective == 15  # 10 through a, 5 through b

    def test_bottleneck(self):
        graph = {
            "s": [("a", 100, 0)],
            "a": [("b", 5, 0)],  # Bottleneck
            "b": [("t", 100, 0)],
            "t": [],
        }
        result = max_flow(graph, "s", "t")
        assert result.status == Status.OPTIMAL
        assert result.objective == 5


class TestMaxFlowComplex:
    def test_diamond(self):
        # Diamond network: s -> a,b -> c -> t
        graph = {
            "s": [("a", 10, 0), ("b", 10, 0)],
            "a": [("c", 5, 0)],
            "b": [("c", 5, 0)],
            "c": [("t", 15, 0)],
            "t": [],
        }
        result = max_flow(graph, "s", "t")
        assert result.status == Status.OPTIMAL
        assert result.objective == 10  # Limited by a->c and b->c

    def test_parallel_paths(self):
        graph = {
            "s": [("a", 5, 0), ("b", 5, 0), ("c", 5, 0)],
            "a": [("t", 5, 0)],
            "b": [("t", 5, 0)],
            "c": [("t", 5, 0)],
            "t": [],
        }
        result = max_flow(graph, "s", "t")
        assert result.status == Status.OPTIMAL
        assert result.objective == 15

    def test_multiple_edges_same_nodes(self):
        # Multiple edges between same nodes
        graph = {"s": [("t", 5, 0), ("t", 3, 0)], "t": []}
        result = max_flow(graph, "s", "t")
        assert result.status == Status.OPTIMAL
        assert result.objective == 8


class TestMinCostFlow:
    def test_basic_min_cost(self):
        graph = {"s": [("a", 10, 1), ("b", 10, 2)], "a": [("t", 10, 1)], "b": [("t", 10, 1)], "t": []}
        result = min_cost_flow(graph, "s", "t", 5)
        assert result.status == Status.OPTIMAL
        # Should prefer path through 'a' (cost 1+1=2 vs 2+1=3)

    def test_equal_cost_paths(self):
        graph = {"s": [("a", 10, 2), ("b", 10, 2)], "a": [("t", 10, 2)], "b": [("t", 10, 2)], "t": []}
        result = min_cost_flow(graph, "s", "t", 5)
        assert result.status == Status.OPTIMAL
        assert result.objective == 20  # 5 units * cost 4

    def test_insufficient_capacity(self):
        graph = {"s": [("t", 5, 1)], "t": []}
        result = min_cost_flow(graph, "s", "t", 10)
        assert result.status == Status.INFEASIBLE


class TestAssignment:
    def test_simple_3x3(self):
        costs = [[10, 5, 13], [3, 9, 18], [10, 6, 12]]
        result = solve_assignment(costs)
        assert result.status == Status.OPTIMAL
        assert len(result.solution) == 3
        assert set(result.solution) == {0, 1, 2}

    def test_2x2(self):
        costs = [[1, 10], [10, 1]]
        result = solve_assignment(costs)
        assert result.status == Status.OPTIMAL
        # Optimal: worker 0 -> task 0, worker 1 -> task 1
        assert result.solution[0] == 0
        assert result.solution[1] == 1
        assert result.objective == 2

    def test_uniform_costs(self):
        costs = [[5, 5, 5], [5, 5, 5], [5, 5, 5]]
        result = solve_assignment(costs)
        assert result.status == Status.OPTIMAL
        assert len(set(result.solution)) == 3  # All different assignments
        assert result.objective == 15  # 3 * 5

    def test_4x4(self):
        costs = [[9, 2, 7, 8], [6, 4, 3, 7], [5, 8, 1, 8], [7, 6, 9, 4]]
        result = solve_assignment(costs)
        assert result.status == Status.OPTIMAL
        assert len(result.solution) == 4
        assert set(result.solution) == {0, 1, 2, 3}


class TestEdgeCases:
    def test_no_flow_possible(self):
        # Disconnected graph
        graph = {"s": [("a", 10, 0)], "a": [], "b": [("t", 10, 0)], "t": []}
        result = max_flow(graph, "s", "t")
        assert result.objective == 0

    def test_zero_capacity(self):
        graph = {"s": [("t", 0, 0)], "t": []}
        result = max_flow(graph, "s", "t")
        assert result.objective == 0

    def test_single_node_assignment(self):
        costs = [[5]]
        result = solve_assignment(costs)
        assert result.status == Status.OPTIMAL
        assert result.solution == [0]

    def test_rectangular_assignment(self):
        # More tasks than workers
        costs = [[1, 5, 3, 2], [4, 2, 6, 1]]
        result = solve_assignment(costs)
        assert result.status == Status.OPTIMAL
        assert len(result.solution) == 2


class TestFlowValues:
    def test_flow_conservation(self):
        graph = {
            "s": [("a", 10, 0), ("b", 10, 0)],
            "a": [("c", 5, 0), ("d", 5, 0)],
            "b": [("c", 5, 0), ("d", 5, 0)],
            "c": [("t", 10, 0)],
            "d": [("t", 10, 0)],
            "t": [],
        }
        result = max_flow(graph, "s", "t")
        assert result.status == Status.OPTIMAL

        # Check flow values are returned
        assert isinstance(result.solution, dict)

    def test_saturated_edges(self):
        graph = {"s": [("t", 5, 0)], "t": []}
        result = max_flow(graph, "s", "t")
        assert result.objective == 5
        assert result.solution[("s", "t")] == 5


class TestStress:
    def test_larger_network(self):
        # 10-node network
        n = 5
        graph = {"s": [], "t": []}

        # Source to middle layer
        for i in range(n):
            graph["s"].append((f"m{i}", 10, 0))
            graph[f"m{i}"] = [(f"n{i}", 10, 0)]
            graph[f"n{i}"] = [("t", 10, 0)]

        result = max_flow(graph, "s", "t")
        assert result.status == Status.OPTIMAL
        assert result.objective == 50  # 5 paths * 10 capacity

    def test_larger_assignment(self):
        import random

        random.seed(42)
        n = 8
        costs = [[random.randint(1, 20) for _ in range(n)] for _ in range(n)]
        result = solve_assignment(costs)
        assert result.status == Status.OPTIMAL
        assert len(result.solution) == n
        assert set(result.solution) == set(range(n))

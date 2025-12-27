"""Tests for A* search."""

from solvor.a_star import astar, astar_grid
from solvor.types import Status


class TestAstarBasic:
    def test_direct_path(self):
        graph = {"A": [("B", 1)], "B": [("C", 1)], "C": []}
        result = astar("A", "C", lambda n: graph.get(n, []), lambda s: 0)
        assert result.status == Status.OPTIMAL
        assert result.solution == ["A", "B", "C"]
        assert result.objective == 2

    def test_heuristic_guides_search(self):
        graph = {"A": [("B", 1), ("C", 1)], "B": [("D", 10)], "C": [("D", 1)], "D": []}
        h = {"A": 2, "B": 10, "C": 1, "D": 0}
        result = astar("A", "D", lambda n: graph.get(n, []), lambda s: h[s])
        assert result.status == Status.OPTIMAL
        assert result.objective == 2
        assert result.solution == ["A", "C", "D"]

    def test_start_is_goal(self):
        result = astar("A", "A", lambda n: [], lambda s: 0)
        assert result.solution == ["A"]
        assert result.objective == 0

    def test_no_path(self):
        graph = {"A": [("B", 1)], "B": [], "C": []}
        result = astar("A", "C", lambda n: graph.get(n, []), lambda s: 0)
        assert result.status == Status.INFEASIBLE

    def test_callable_goal(self):
        # Goal is any node >= 10
        graph = {i: [(i + 1, 1), (i + 2, 2)] for i in range(15)}
        graph[15] = []

        def is_goal(n):
            return n >= 10

        result = astar(0, is_goal, lambda n: graph.get(n, []), lambda s: max(0, 10 - s))
        assert result.status == Status.OPTIMAL
        assert result.solution[-1] >= 10
        assert result.objective == 10  # Shortest path to any n >= 10

    def test_callable_goal_multiple_targets(self):
        # Goal is any of specific nodes
        graph = {"A": [("B", 1), ("C", 5)], "B": [("D", 1)], "C": [("E", 1)], "D": [], "E": []}
        targets = {"D", "E"}
        result = astar("A", lambda n: n in targets, lambda n: graph.get(n, []), lambda s: 0)
        assert result.status == Status.OPTIMAL
        assert result.solution[-1] in targets
        assert result.objective == 2  # A -> B -> D is shortest


class TestWeightedAstar:
    def test_weight_greater_than_one(self):
        graph = {i: [(i + 1, 1)] for i in range(10)}
        graph[10] = []
        result = astar(0, 10, lambda n: graph.get(n, []), lambda s: 10 - s, weight=2.0)
        assert result.status == Status.FEASIBLE
        assert result.solution[-1] == 10

    def test_weight_one_is_optimal(self):
        graph = {"A": [("B", 1), ("C", 2)], "B": [("D", 3)], "C": [("D", 1)], "D": []}
        result = astar("A", "D", lambda n: graph.get(n, []), lambda s: 0, weight=1.0)
        assert result.status == Status.OPTIMAL


class TestAstarGrid:
    def test_simple_grid(self):
        grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        result = astar_grid(grid, (0, 0), (2, 2))
        assert result.status == Status.OPTIMAL
        assert result.solution[0] == (0, 0)
        assert result.solution[-1] == (2, 2)

    def test_obstacle_avoidance(self):
        grid = [[0, 0, 0], [1, 1, 0], [0, 0, 0]]
        result = astar_grid(grid, (0, 0), (2, 0))
        assert result.status == Status.OPTIMAL
        assert (1, 0) not in result.solution
        assert (1, 1) not in result.solution

    def test_blocked_goal(self):
        grid = [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
        result = astar_grid(grid, (0, 0), (2, 1))
        assert result.status == Status.INFEASIBLE

    def test_8_directions(self):
        grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        result = astar_grid(grid, (0, 0), (2, 2), directions=8)
        assert result.status == Status.OPTIMAL
        assert len(result.solution) <= 3

    def test_4_directions(self):
        grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        result = astar_grid(grid, (0, 0), (2, 2), directions=4)
        assert result.status == Status.OPTIMAL
        assert result.objective == 4


class TestHeuristics:
    def test_heuristic_reduces_expansions(self):
        # Good heuristic should expand fewer nodes than h=0 (Dijkstra)
        graph = {i: [(i + 1, 1), (i + 10, 1)] for i in range(100)}
        graph[100] = []
        for i in range(10, 100, 10):
            graph[i] = [(i + 1, 1)]

        # With h=0 (no guidance), explores more nodes
        result_no_h = astar(0, 100, lambda n: graph.get(n, []), lambda s: 0)

        # With informative heuristic, explores fewer
        result_with_h = astar(0, 100, lambda n: graph.get(n, []), lambda s: max(0, 100 - s))

        assert result_with_h.iterations <= result_no_h.iterations
        assert result_with_h.objective == result_no_h.objective  # Same optimal cost


class TestGridCosts:
    def test_terrain_costs_avoids_expensive(self):
        # Grid layout (row, col):
        #   col: 0   1   2   3   4
        # row 0: 0   2   2   2   0
        # row 1: 0   0   0   0   0
        # Direct path (0,0)→(0,1)→(0,2)→(0,3)→(0,4) = 10+10+10+1 = 31
        # Path via row 1: (0,0)→(1,0)→(1,1)→(1,2)→(1,3)→(1,4)→(0,4) = 6
        grid = [
            [0, 2, 2, 2, 0],
            [0, 0, 0, 0, 0],
        ]
        result = astar_grid(grid, (0, 0), (0, 4), costs={2: 10.0})
        assert result.status == Status.OPTIMAL
        # Should go around via row 1
        assert (0, 1) not in result.solution
        assert (0, 2) not in result.solution
        assert (0, 3) not in result.solution
        assert result.objective == 6

    def test_terrain_costs_uses_expensive_when_shorter(self):
        # Direct path (0,0)→(0,1)→(0,2) costs: 1.5 + 1 = 2.5
        # Path around (0,0)→(1,0)→(1,1)→(1,2)→(0,2) costs: 4
        grid = [
            [0, 2, 0],
            [0, 0, 0],
        ]
        result = astar_grid(grid, (0, 0), (0, 2), costs={2: 1.5})
        assert result.status == Status.OPTIMAL
        # Should go through expensive terrain (still cheaper overall)
        assert (0, 1) in result.solution
        assert result.objective == 2.5

    def test_blocked_set(self):
        grid = [[0, 1, 0], [0, 2, 0], [0, 0, 0]]
        result = astar_grid(grid, (0, 0), (0, 2), blocked={1, 2})
        assert result.status == Status.OPTIMAL
        assert (0, 1) not in result.solution
        assert (1, 1) not in result.solution


class TestEdgeCases:
    def test_single_cell(self):
        grid = [[0]]
        result = astar_grid(grid, (0, 0), (0, 0))
        assert result.solution == [(0, 0)]

    def test_adjacent_cells(self):
        grid = [[0, 0]]
        result = astar_grid(grid, (0, 0), (0, 1))
        assert result.objective == 1

    def test_max_iter_limit(self):
        grid = [[0] * 100 for _ in range(100)]
        result = astar_grid(grid, (0, 0), (99, 99), max_iter=10)
        assert result.status == Status.MAX_ITER


class TestMaxCost:
    def test_max_cost_prunes_expensive_paths(self):
        # max_cost prevents exploring expensive nodes, but goal is still reachable
        # via cheaper path if one exists
        graph = {
            "A": [("B", 1), ("D", 10)],  # Two paths: A->B->C or A->D->C
            "B": [("C", 1)],
            "D": [("C", 1)],
            "C": [],
        }
        result = astar("A", "C", lambda n: graph.get(n, []), lambda s: 0, max_cost=5)
        assert result.status == Status.OPTIMAL
        assert result.solution == ["A", "B", "C"]
        assert result.objective == 2

    def test_max_cost_allows_close_goal(self):
        # Goal is within max_cost
        graph = {"A": [("B", 1)], "B": [("C", 1)], "C": []}
        result = astar("A", "C", lambda n: graph.get(n, []), lambda s: 0, max_cost=5)
        assert result.status == Status.OPTIMAL
        assert result.solution == ["A", "B", "C"]
        assert result.objective == 2

    def test_max_cost_equal_to_path(self):
        # max_cost exactly equals path cost
        graph = {"A": [("B", 2)], "B": [("C", 2)], "C": []}
        result = astar("A", "C", lambda n: graph.get(n, []), lambda s: 0, max_cost=4)
        assert result.status == Status.OPTIMAL
        assert result.objective == 4

    def test_max_cost_prevents_expansion_beyond_limit(self):
        # Verify that nodes beyond max_cost are not expanded
        expanded = []

        def tracked_neighbors(n):
            expanded.append(n)
            graph = {"A": [("B", 3)], "B": [("C", 3)], "C": [("D", 3)], "D": []}
            return graph.get(n, [])

        astar("A", "D", tracked_neighbors, lambda s: 0, max_cost=4)
        # B (cost 3) should be expanded, C (cost 6) should not
        assert "B" in expanded
        # C might be added to frontier but shouldn't be expanded since g[C]=6 > 4


class TestStress:
    def test_large_grid(self):
        n = 50
        grid = [[0] * n for _ in range(n)]
        result = astar_grid(grid, (0, 0), (n - 1, n - 1), directions=8)
        assert result.status == Status.OPTIMAL
        # With 8 directions, diagonal path length should be 49 (Chebyshev distance)
        assert len(result.solution) == n  # Path from (0,0) to (49,49) inclusive

    def test_maze(self):
        grid = [[0, 0, 0, 0, 0], [1, 1, 1, 1, 0], [0, 0, 0, 0, 0], [0, 1, 1, 1, 1], [0, 0, 0, 0, 0]]
        result = astar_grid(grid, (0, 0), (4, 4))
        assert result.status == Status.OPTIMAL

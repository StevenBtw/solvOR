"""Tests for BFS and DFS solvers."""

from solvor.bfs import bfs, dfs
from solvor.types import Status


class TestBFSBasic:
    def test_direct_path(self):
        graph = {"A": ["B"], "B": ["C"], "C": []}
        result = bfs("A", "C", lambda n: graph.get(n, []))
        assert result.status == Status.OPTIMAL
        assert result.solution == ["A", "B", "C"]
        assert result.objective == 2

    def test_shortest_path(self):
        graph = {"A": ["B", "C"], "B": ["D"], "C": ["D"], "D": []}
        result = bfs("A", "D", lambda n: graph.get(n, []))
        assert result.status == Status.OPTIMAL
        assert result.objective == 2
        assert len(result.solution) == 3

    def test_start_is_goal(self):
        graph = {"A": ["B"], "B": []}
        result = bfs("A", "A", lambda n: graph.get(n, []))
        assert result.status == Status.OPTIMAL
        assert result.solution == ["A"]
        assert result.objective == 0

    def test_no_path(self):
        graph = {"A": ["B"], "B": [], "C": ["D"], "D": []}
        result = bfs("A", "D", lambda n: graph.get(n, []))
        assert result.status == Status.INFEASIBLE
        assert result.solution is None

    def test_goal_predicate(self):
        graph = {0: [1, 2], 1: [3], 2: [4], 3: [], 4: []}
        result = bfs(0, lambda s: s > 3, lambda n: graph.get(n, []))
        assert result.status == Status.OPTIMAL
        assert result.solution[-1] == 4


class TestDFSBasic:
    def test_finds_path(self):
        graph = {"A": ["B"], "B": ["C"], "C": []}
        result = dfs("A", "C", lambda n: graph.get(n, []))
        assert result.status == Status.FEASIBLE
        assert result.solution[-1] == "C"
        assert result.solution[0] == "A"

    def test_start_is_goal(self):
        graph = {"A": ["B"], "B": []}
        result = dfs("A", "A", lambda n: graph.get(n, []))
        assert result.status == Status.FEASIBLE
        assert result.solution == ["A"]

    def test_no_path(self):
        graph = {"A": ["B"], "B": [], "C": []}
        result = dfs("A", "C", lambda n: graph.get(n, []))
        assert result.status == Status.INFEASIBLE


class TestExploration:
    def test_bfs_explore_all(self):
        graph = {"A": ["B", "C"], "B": ["D"], "C": ["D"], "D": []}
        result = bfs("A", None, lambda n: graph.get(n, []))
        assert result.status == Status.OPTIMAL
        assert result.solution == {"A", "B", "C", "D"}
        assert result.objective == 4

    def test_dfs_explore_all(self):
        graph = {"A": ["B", "C"], "B": ["D"], "C": [], "D": []}
        result = dfs("A", None, lambda n: graph.get(n, []))
        assert result.status == Status.OPTIMAL
        assert result.solution == {"A", "B", "C", "D"}

    def test_disconnected_graph(self):
        graph = {"A": ["B"], "B": [], "C": ["D"], "D": []}
        result = bfs("A", None, lambda n: graph.get(n, []))
        assert result.solution == {"A", "B"}


class TestGrid:
    def test_bfs_grid(self):
        grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]

        def neighbors(pos):
            r, c = pos
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 3 and 0 <= nc < 3 and grid[nr][nc] == 0:
                    yield (nr, nc)

        result = bfs((0, 0), (2, 2), neighbors)
        assert result.status == Status.OPTIMAL
        assert result.solution[0] == (0, 0)
        assert result.solution[-1] == (2, 2)


class TestEdgeCases:
    def test_single_node(self):
        result = bfs("A", "A", lambda n: [])
        assert result.solution == ["A"]

    def test_cycle_handling(self):
        graph = {"A": ["B"], "B": ["C"], "C": ["A", "D"], "D": []}
        result = bfs("A", "D", lambda n: graph.get(n, []))
        assert result.status == Status.OPTIMAL
        assert result.solution[-1] == "D"

    def test_max_iter_limit(self):
        def infinite_neighbors(n):
            return [n + 1]

        result = bfs(0, 1000000, infinite_neighbors, max_iter=100)
        assert result.status == Status.MAX_ITER


class TestStress:
    def test_wide_graph(self):
        n = 100
        graph = {0: list(range(1, n + 1))}
        for i in range(1, n + 1):
            graph[i] = [n + 1]
        graph[n + 1] = []

        result = bfs(0, n + 1, lambda node: graph.get(node, []))
        assert result.status == Status.OPTIMAL
        assert result.objective == 2

    def test_deep_graph(self):
        n = 500
        graph = {i: [i + 1] for i in range(n)}
        graph[n] = []

        result = bfs(0, n, lambda node: graph.get(node, []))
        assert result.status == Status.OPTIMAL
        assert result.objective == n

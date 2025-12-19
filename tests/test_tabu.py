"""Tests for the tabu search solver."""

from solvor.tabu import tabu_search, solve_tsp
from solvor.types import Status, Progress


class TestBasicTabu:
    def test_simple_search(self):
        # Find x = 10
        def objective(x):
            return abs(x - 10)

        def neighbors(x):
            return [('dec', x - 1), ('inc', x + 1)]

        result = tabu_search(0, objective, neighbors, max_iter=100)
        assert result.status == Status.FEASIBLE
        assert result.solution == 10

    def test_search_from_above(self):
        # Start above target
        def objective(x):
            return abs(x - 5)

        def neighbors(x):
            return [('dec', x - 1), ('inc', x + 1)]

        result = tabu_search(20, objective, neighbors, max_iter=100)
        assert result.status == Status.FEASIBLE
        assert result.solution == 5

    def test_minimize_vs_escape_local(self):
        # Test that tabu helps escape local optima
        def objective(x):
            # Local minimum at x=5, global at x=10
            if x == 5:
                return 1
            elif x == 10:
                return 0
            else:
                return abs(x - 5) + 2

        def neighbors(x):
            moves = []
            if x > 0:
                moves.append(('dec', x - 1))
            moves.append(('inc', x + 1))
            return moves

        result = tabu_search(0, objective, neighbors, max_iter=50, cooldown=10)
        # Should find global optimum
        assert result.objective <= 1


class TestTSP:
    def test_trivial_3city(self):
        dist = [[0, 1, 2], [1, 0, 3], [2, 3, 0]]
        result = solve_tsp(dist)
        assert result.status == Status.FEASIBLE
        assert len(result.solution) == 3
        assert set(result.solution) == {0, 1, 2}

    def test_small_4city(self):
        dist = [
            [0, 10, 15, 20],
            [10, 0, 35, 25],
            [15, 35, 0, 30],
            [20, 25, 30, 0],
        ]
        result = solve_tsp(dist)
        assert result.status == Status.FEASIBLE
        assert len(result.solution) == 4
        assert set(result.solution) == {0, 1, 2, 3}

    def test_symmetric_square(self):
        # 4 cities at corners of a square
        # Optimal tour visits them in order around the square
        dist = [
            [0, 1, 2, 1],  # (0,0)
            [1, 0, 1, 2],  # (1,0)
            [2, 1, 0, 1],  # (1,1)
            [1, 2, 1, 0],  # (0,1)
        ]
        result = solve_tsp(dist)
        assert result.status == Status.FEASIBLE
        # Optimal tour length is 4 (visiting corners in order)
        assert result.objective == 4

    def test_5city(self):
        # 5 cities
        dist = [
            [0, 2, 9, 10, 7],
            [2, 0, 6, 4, 3],
            [9, 6, 0, 8, 5],
            [10, 4, 8, 0, 6],
            [7, 3, 5, 6, 0],
        ]
        result = solve_tsp(dist)
        assert result.status == Status.FEASIBLE
        assert len(result.solution) == 5
        assert set(result.solution) == {0, 1, 2, 3, 4}


class TestEdgeCases:
    def test_2city(self):
        # Trivial 2-city case
        dist = [[0, 5], [5, 0]]
        result = solve_tsp(dist)
        assert len(result.solution) == 2
        assert result.objective == 10  # Round trip

    def test_already_optimal(self):
        # Start at optimal
        def objective(x):
            return x ** 2

        def neighbors(x):
            return [('dec', x - 1), ('inc', x + 1)]

        result = tabu_search(0, objective, neighbors, max_iter=50)
        assert result.solution == 0

    def test_large_neighborhood(self):
        # Many neighbors
        def objective(x):
            return abs(x - 50)

        def neighbors(x):
            moves = []
            for delta in range(-5, 6):
                if delta != 0:
                    moves.append((f'd{delta}', x + delta))
            return moves

        result = tabu_search(0, objective, neighbors, max_iter=50)
        assert result.objective < 10  # Should get close


class TestParameters:
    def test_cooldown_effect(self):
        # Larger cooldown prevents cycling
        def objective(x):
            return abs(x - 20)

        def neighbors(x):
            return [('dec', x - 1), ('inc', x + 1)]

        result = tabu_search(0, objective, neighbors, max_iter=100, cooldown=20)
        assert result.solution == 20

    def test_max_iter_limit(self):
        # Should stop at max_iter
        def objective(x):
            return abs(x - 1000)  # Far target

        def neighbors(x):
            return [('dec', x - 1), ('inc', x + 1)]

        result = tabu_search(0, objective, neighbors, max_iter=10)
        assert result.iterations <= 10


class TestStress:
    def test_tsp_8city(self):
        # Larger TSP
        import random
        random.seed(42)
        n = 8
        # Random symmetric distance matrix
        dist = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = random.randint(1, 20)
                dist[i][j] = d
                dist[j][i] = d

        result = solve_tsp(dist, max_iter=500)
        assert result.status == Status.FEASIBLE
        assert len(result.solution) == n
        assert set(result.solution) == set(range(n))

    def test_rapid_convergence(self):
        # Problem that should converge quickly
        def objective(x):
            return (x - 5) ** 2

        def neighbors(x):
            return [('dec', x - 1), ('inc', x + 1)]

        result = tabu_search(0, objective, neighbors, max_iter=20)
        assert result.solution == 5


class TestProgressCallback:
    def test_callback_called_at_interval(self):
        # Use a large target so we don't reach optimum early
        def objective(x):
            return abs(x - 100)

        def neighbors(x):
            return [('dec', x - 1), ('inc', x + 1)]

        calls = []

        def callback(progress):
            calls.append(progress.iteration)

        tabu_search(0, objective, neighbors, max_iter=50, on_progress=callback, progress_interval=10)
        assert calls == [10, 20, 30, 40, 50]

    def test_callback_early_stop(self):
        def objective(x):
            return abs(x - 100)

        def neighbors(x):
            return [('dec', x - 1), ('inc', x + 1)]

        def stop_at_20(progress):
            if progress.iteration >= 20:
                return True

        result = tabu_search(0, objective, neighbors, max_iter=100, on_progress=stop_at_20, progress_interval=5)
        assert result.iterations == 20

    def test_callback_receives_progress_data(self):
        def objective(x):
            return abs(x - 10)

        def neighbors(x):
            return [('dec', x - 1), ('inc', x + 1)]

        received = []

        def callback(progress):
            received.append(progress)

        tabu_search(0, objective, neighbors, max_iter=20, on_progress=callback, progress_interval=5)
        assert len(received) > 0
        p = received[0]
        assert isinstance(p, Progress)
        assert p.iteration == 5
        assert p.objective is not None
        assert p.evaluations > 0


class TestMaximize:
    def test_maximize_simple(self):
        def objective(x):
            return -abs(x - 10)  # Maximum at x=10

        def neighbors(x):
            return [('dec', x - 1), ('inc', x + 1)]

        result = tabu_search(0, objective, neighbors, minimize=False, max_iter=50)
        assert result.solution == 10
        assert result.objective == 0

    def test_maximize_tsp(self):
        # For TSP, maximize means longest path
        dist = [[0, 1, 2], [1, 0, 3], [2, 3, 0]]
        result_min = solve_tsp(dist, minimize=True)
        result_max = solve_tsp(dist, minimize=False)
        # Longest tour should have larger objective
        assert result_max.objective >= result_min.objective


class TestMaxNoImprove:
    def test_early_stop_on_no_improve(self):
        # Objective that plateaus - no improvement possible after reaching 0
        def objective(x):
            return max(0, 10 - x)  # Stays at 0 for x >= 10

        def neighbors(x):
            return [('dec', x - 1), ('inc', x + 1)]

        result = tabu_search(0, objective, neighbors, max_iter=1000, max_no_improve=20)
        # Should stop early due to no improvement
        assert result.iterations < 1000

    def test_continues_with_improvement(self):
        # Should not stop early if still improving
        def objective(x):
            return abs(x - 30)

        def neighbors(x):
            return [('dec', x - 1), ('inc', x + 1)]

        result = tabu_search(0, objective, neighbors, max_iter=100, max_no_improve=50)
        # Should reach the optimum
        assert result.solution == 30

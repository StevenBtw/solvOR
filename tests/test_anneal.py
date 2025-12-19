"""Tests for the simulated annealing solver."""

import pytest
from random import gauss, seed
from solvor.anneal import anneal, Status


def make_neighbor_fn(std=0.5):
    """Create a neighbor function with gaussian perturbation."""
    def neighbor(x):
        return [xi + gauss(0, std) for xi in x]
    return neighbor


class TestBasicAnneal:
    def test_minimize_quadratic(self):
        # Minimize x^2, starting from x=5
        seed(42)
        result = anneal(
            [5.0],
            lambda x: x[0] ** 2,
            make_neighbor_fn(0.5),
            max_iter=5000
        )
        assert result.status in (Status.FEASIBLE, Status.MAX_ITER)
        assert abs(result.solution[0]) < 1.0

    def test_maximize_quadratic(self):
        # Maximize -x^2 (peak at 0)
        seed(42)
        result = anneal(
            [5.0],
            lambda x: -x[0] ** 2,
            make_neighbor_fn(0.5),
            minimize=False,
            max_iter=5000
        )
        assert abs(result.solution[0]) < 1.0

    def test_2d_optimization(self):
        # Minimize x^2 + y^2
        seed(42)
        result = anneal(
            [3.0, 4.0],
            lambda x: x[0] ** 2 + x[1] ** 2,
            make_neighbor_fn(0.3),
            max_iter=10000
        )
        assert result.status in (Status.FEASIBLE, Status.MAX_ITER)
        assert abs(result.solution[0]) < 1.5
        assert abs(result.solution[1]) < 1.5


class TestMultiModal:
    def test_rastrigin_like(self):
        # Simple multimodal function: x^2 + sin(4*x)
        # Has local minima but global at x~0
        seed(42)
        result = anneal(
            [3.0],
            lambda x: x[0] ** 2 + 0.5 * (1 - __import__('math').cos(4 * x[0])),
            make_neighbor_fn(0.3),
            max_iter=5000,
            temperature=100.0
        )
        # Should find something near the global minimum
        assert result.objective < 2.0


class TestParameters:
    def test_high_temperature(self):
        # High temperature allows more exploration
        seed(42)
        result = anneal(
            [10.0],
            lambda x: x[0] ** 2,
            make_neighbor_fn(1.0),
            temperature=10000.0,
            max_iter=3000
        )
        assert result.status in (Status.FEASIBLE, Status.MAX_ITER)

    def test_fast_cooling(self):
        # Fast cooling converges quickly
        seed(42)
        result = anneal(
            [5.0],
            lambda x: x[0] ** 2,
            make_neighbor_fn(0.5),
            cooling=0.99,
            max_iter=1000
        )
        assert result.status in (Status.FEASIBLE, Status.MAX_ITER)

    def test_min_temp_stop(self):
        # Should stop when temperature drops below min_temp
        seed(42)
        result = anneal(
            [5.0],
            lambda x: x[0] ** 2,
            make_neighbor_fn(0.5),
            min_temp=1e-4,
            cooling=0.99,
            max_iter=100000
        )
        assert result.status == Status.FEASIBLE  # Stopped by min_temp, not max_iter


class TestEdgeCases:
    def test_already_optimal(self):
        # Start at optimal
        seed(42)
        result = anneal(
            [0.0],
            lambda x: x[0] ** 2,
            make_neighbor_fn(0.1),
            max_iter=1000
        )
        # Should stay near optimal
        assert abs(result.solution[0]) < 0.5

    def test_discrete_neighbor(self):
        # Discrete optimization
        seed(42)
        def discrete_objective(x):
            return abs(x[0] - 7)

        def discrete_neighbor(x):
            from random import choice
            delta = choice([-1, 1])
            return [x[0] + delta]

        result = anneal(
            [0],
            discrete_objective,
            discrete_neighbor,
            max_iter=1000
        )
        assert abs(result.solution[0] - 7) < 3

    def test_high_dimensional(self):
        # Higher dimensional problem
        seed(42)
        n = 10
        result = anneal(
            [5.0] * n,
            lambda x: sum(xi ** 2 for xi in x),
            make_neighbor_fn(0.2),
            max_iter=20000
        )
        # Should reduce the objective significantly
        assert result.objective < sum(5.0 ** 2 for _ in range(n)) * 0.5


class TestStress:
    def test_many_iterations(self):
        # Long run for better convergence
        seed(42)
        result = anneal(
            [10.0],
            lambda x: x[0] ** 2,
            make_neighbor_fn(0.3),
            max_iter=50000
        )
        assert abs(result.solution[0]) < 0.5

    def test_evaluations_counted(self):
        # Verify evaluations are tracked
        seed(42)
        result = anneal(
            [5.0],
            lambda x: x[0] ** 2,
            make_neighbor_fn(0.5),
            max_iter=100
        )
        # Should have at least initial + max_iter evaluations
        assert result.evaluations >= 100

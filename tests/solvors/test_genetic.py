"""Tests for the genetic algorithm solver."""

from random import randint
from random import seed as set_seed

from solvor.genetic import evolve
from solvor.types import Progress, Status


def simple_crossover(p1, p2):
    """Single-point crossover."""
    mid = len(p1) // 2
    return p1[:mid] + p2[mid:]


def bit_mutate(bits):
    """Flip a random bit."""
    bits = list(bits)
    i = randint(0, len(bits) - 1)
    bits[i] = 1 - bits[i]
    return tuple(bits)


class TestBasicGA:
    def test_minimize_sum(self):
        # Minimize sum of bits (find all zeros)
        def objective(bits):
            return sum(bits)

        population = [tuple([1] * 10) for _ in range(20)]
        result = evolve(objective, population, simple_crossover, bit_mutate, max_gen=50, seed=42)
        assert result.status == Status.FEASIBLE
        assert result.objective < 5  # Should find mostly zeros

    def test_maximize_sum(self):
        # Maximize sum of bits (find all ones)
        def objective(bits):
            return sum(bits)

        population = [tuple([0] * 10) for _ in range(20)]
        result = evolve(objective, population, simple_crossover, bit_mutate, max_gen=50, minimize=False, seed=42)
        assert result.status == Status.FEASIBLE
        assert result.objective > 5  # Should find mostly ones

    def test_target_pattern(self):
        # Find specific pattern [1,0,1,0,1]
        target = (1, 0, 1, 0, 1)

        def objective(bits):
            return sum(b != t for b, t in zip(bits, target))

        population = [tuple(randint(0, 1) for _ in range(5)) for _ in range(20)]
        result = evolve(objective, population, simple_crossover, bit_mutate, max_gen=100, seed=42)
        assert result.status == Status.FEASIBLE
        assert result.objective <= 2  # Should get close to target


class TestCrossover:
    def test_two_point_crossover(self):
        def two_point(p1, p2):
            n = len(p1)
            i, j = sorted([randint(0, n), randint(0, n)])
            return p1[:i] + p2[i:j] + p1[j:]

        def objective(bits):
            return sum(bits)

        population = [tuple([1] * 8) for _ in range(15)]
        result = evolve(objective, population, two_point, bit_mutate, max_gen=50, seed=42)
        assert result.status == Status.FEASIBLE

    def test_uniform_crossover(self):
        def uniform(p1, p2):
            return tuple(p1[i] if randint(0, 1) else p2[i] for i in range(len(p1)))

        def objective(bits):
            return sum(bits)

        population = [tuple([1] * 8) for _ in range(15)]
        result = evolve(objective, population, uniform, bit_mutate, max_gen=50, seed=42)
        assert result.status == Status.FEASIBLE


class TestMutation:
    def test_multi_bit_mutate(self):
        def multi_mutate(bits):
            bits = list(bits)
            for _ in range(2):
                i = randint(0, len(bits) - 1)
                bits[i] = 1 - bits[i]
            return tuple(bits)

        def objective(bits):
            return sum(bits)

        population = [tuple([1] * 10) for _ in range(20)]
        result = evolve(objective, population, simple_crossover, multi_mutate, max_gen=50, seed=42)
        assert result.status == Status.FEASIBLE


class TestParameters:
    def test_large_population(self):
        def objective(bits):
            return sum(bits)

        population = [tuple([1] * 10) for _ in range(50)]
        result = evolve(objective, population, simple_crossover, bit_mutate, max_gen=30, seed=42)
        assert result.status == Status.FEASIBLE

    def test_small_population(self):
        def objective(bits):
            return sum(bits)

        population = [tuple([1] * 10) for _ in range(5)]
        result = evolve(objective, population, simple_crossover, bit_mutate, max_gen=50, seed=42)
        assert result.status == Status.FEASIBLE

    def test_elitism(self):
        def objective(bits):
            return sum(bits)

        population = [tuple([1] * 10) for _ in range(20)]
        result = evolve(objective, population, simple_crossover, bit_mutate, max_gen=30, elite_size=5, seed=42)
        assert result.status == Status.FEASIBLE

    def test_mutation_rate(self):
        def objective(bits):
            return sum(bits)

        population = [tuple([1] * 10) for _ in range(20)]
        result = evolve(objective, population, simple_crossover, bit_mutate, max_gen=50, mutation_rate=0.5, seed=42)
        assert result.status == Status.FEASIBLE


class TestRealValuedGA:
    def test_float_optimization(self):
        # Real-valued GA
        def float_crossover(p1, p2):
            alpha = 0.5
            return tuple(alpha * a + (1 - alpha) * b for a, b in zip(p1, p2))

        def float_mutate(x):
            from random import gauss

            return tuple(xi + gauss(0, 0.1) for xi in x)

        def objective(x):
            return sum(xi**2 for xi in x)

        set_seed(42)
        population = [tuple(randint(-10, 10) for _ in range(3)) for _ in range(20)]
        result = evolve(objective, population, float_crossover, float_mutate, max_gen=100, seed=42)
        assert result.status == Status.FEASIBLE
        assert result.objective < 50  # Should improve from initial


class TestEdgeCases:
    def test_single_bit(self):
        def objective(bits):
            return bits[0]

        population = [tuple([1]) for _ in range(10)]
        result = evolve(objective, population, simple_crossover, bit_mutate, max_gen=20, seed=42)
        assert result.status == Status.FEASIBLE

    def test_already_optimal(self):
        # Start with optimal solution in population
        def objective(bits):
            return sum(bits)

        population = [tuple([0] * 5)] + [tuple([1] * 5) for _ in range(9)]
        result = evolve(objective, population, simple_crossover, bit_mutate, max_gen=10, seed=42)
        assert result.objective == 0

    def test_identical_population(self):
        def objective(bits):
            return sum(bits)

        population = [tuple([1, 0, 1, 0]) for _ in range(10)]
        result = evolve(objective, population, simple_crossover, bit_mutate, max_gen=30, seed=42)
        assert result.status == Status.FEASIBLE


class TestStress:
    def test_long_chromosome(self):
        def objective(bits):
            return sum(bits)

        population = [tuple([1] * 50) for _ in range(30)]
        result = evolve(objective, population, simple_crossover, bit_mutate, max_gen=100, seed=42)
        assert result.status == Status.FEASIBLE
        assert result.objective < 25  # Should reduce significantly

    def test_many_generations(self):
        def objective(bits):
            return sum(bits)

        population = [tuple([1] * 10) for _ in range(20)]
        result = evolve(objective, population, simple_crossover, bit_mutate, max_gen=200, seed=42)
        assert result.status == Status.FEASIBLE
        assert result.objective < 3


class TestProgressCallback:
    def test_callback_called_at_interval(self):
        def objective(bits):
            return sum(bits)

        population = [tuple([1] * 10) for _ in range(20)]
        calls = []

        def callback(progress):
            calls.append(progress.iteration)

        evolve(
            objective,
            population,
            simple_crossover,
            bit_mutate,
            max_gen=50,
            seed=42,
            on_progress=callback,
            progress_interval=10,
        )
        assert calls == [10, 20, 30, 40, 50]

    def test_callback_early_stop(self):
        def objective(bits):
            return sum(bits)

        population = [tuple([1] * 10) for _ in range(20)]

        def stop_at_20(progress):
            if progress.iteration >= 20:
                return True

        result = evolve(
            objective,
            population,
            simple_crossover,
            bit_mutate,
            max_gen=100,
            seed=42,
            on_progress=stop_at_20,
            progress_interval=5,
        )
        assert result.iterations == 20

    def test_callback_receives_progress_data(self):
        def objective(bits):
            return sum(bits)

        population = [tuple([1] * 10) for _ in range(20)]
        received = []

        def callback(progress):
            received.append(progress)

        evolve(
            objective,
            population,
            simple_crossover,
            bit_mutate,
            max_gen=20,
            seed=42,
            on_progress=callback,
            progress_interval=5,
        )
        assert len(received) > 0
        p = received[0]
        assert isinstance(p, Progress)
        assert p.iteration == 5
        assert isinstance(p.objective, (int, float)) and p.objective == p.objective  # finite number
        assert p.evaluations > 0


class TestTournamentK:
    def test_high_tournament_pressure(self):
        # High tournament_k = more selection pressure
        def objective(bits):
            return sum(bits)

        population = [tuple([1] * 10) for _ in range(20)]
        result = evolve(objective, population, simple_crossover, bit_mutate, max_gen=50, seed=42, tournament_k=10)
        assert result.status == Status.FEASIBLE

    def test_low_tournament_pressure(self):
        # Low tournament_k = less selection pressure, more diversity
        def objective(bits):
            return sum(bits)

        population = [tuple([1] * 10) for _ in range(20)]
        result = evolve(objective, population, simple_crossover, bit_mutate, max_gen=50, seed=42, tournament_k=2)
        assert result.status == Status.FEASIBLE

    def test_tournament_k_comparison(self):
        # Higher tournament_k should converge faster (more greedy)
        def objective(bits):
            return sum(bits)

        population_high = [tuple([1] * 15) for _ in range(30)]
        population_low = [tuple([1] * 15) for _ in range(30)]

        result_high = evolve(
            objective, population_high, simple_crossover, bit_mutate, max_gen=30, seed=42, tournament_k=15
        )
        result_low = evolve(
            objective, population_low, simple_crossover, bit_mutate, max_gen=30, seed=42, tournament_k=2
        )
        # Both should work, but high k typically converges faster
        assert result_high.status == Status.FEASIBLE
        assert result_low.status == Status.FEASIBLE

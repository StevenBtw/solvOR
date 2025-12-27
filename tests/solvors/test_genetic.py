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
        result = evolve(objective, population, simple_crossover, bit_mutate, max_iter=50, seed=42)
        assert result.status == Status.FEASIBLE
        assert result.objective < 5  # Should find mostly zeros

    def test_maximize_sum(self):
        # Maximize sum of bits (find all ones)
        def objective(bits):
            return sum(bits)

        population = [tuple([0] * 10) for _ in range(20)]
        result = evolve(objective, population, simple_crossover, bit_mutate, max_iter=50, minimize=False, seed=42)
        assert result.status == Status.FEASIBLE
        assert result.objective > 5  # Should find mostly ones

    def test_target_pattern(self):
        # Find specific pattern [1,0,1,0,1]
        target = (1, 0, 1, 0, 1)

        def objective(bits):
            return sum(b != t for b, t in zip(bits, target))

        population = [tuple(randint(0, 1) for _ in range(5)) for _ in range(20)]
        result = evolve(objective, population, simple_crossover, bit_mutate, max_iter=100, seed=42)
        assert result.status == Status.FEASIBLE
        assert result.objective <= 2  # Should get close to target


class TestParameters:
    def test_elitism_preserves_best(self):
        # Elitism should never lose the best solution found
        def objective(bits):
            return sum(bits)

        # Include one optimal solution in population
        optimal = tuple([0] * 10)
        population = [optimal] + [tuple([1] * 10) for _ in range(19)]
        result = evolve(objective, population, simple_crossover, bit_mutate, max_iter=30, elite_size=1, seed=42)
        assert result.status == Status.FEASIBLE
        assert result.objective == 0  # Elitism should preserve the optimal solution


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
        result = evolve(objective, population, float_crossover, float_mutate, max_iter=100, seed=42)
        assert result.status == Status.FEASIBLE
        assert result.objective < 50  # Should improve from initial


class TestEdgeCases:
    def test_already_optimal(self):
        # Start with optimal solution in population
        def objective(bits):
            return sum(bits)

        population = [tuple([0] * 5)] + [tuple([1] * 5) for _ in range(9)]
        result = evolve(objective, population, simple_crossover, bit_mutate, max_iter=10, seed=42)
        assert result.objective == 0

    def test_identical_population_still_improves(self):
        # Mutation should introduce diversity even with identical start
        def objective(bits):
            return sum(bits)

        # All ones, objective = 4
        population = [tuple([1, 1, 1, 1]) for _ in range(10)]
        result = evolve(objective, population, simple_crossover, bit_mutate, max_iter=50, seed=42)
        assert result.status == Status.FEASIBLE
        assert result.objective < 4  # Mutation should find improvement


class TestStress:
    def test_long_chromosome(self):
        def objective(bits):
            return sum(bits)

        population = [tuple([1] * 50) for _ in range(30)]
        result = evolve(objective, population, simple_crossover, bit_mutate, max_iter=100, seed=42)
        assert result.status == Status.FEASIBLE
        assert result.objective < 25  # Should reduce significantly

    def test_many_generations(self):
        def objective(bits):
            return sum(bits)

        population = [tuple([1] * 10) for _ in range(20)]
        result = evolve(objective, population, simple_crossover, bit_mutate, max_iter=200, seed=42)
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
            max_iter=50,
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
            max_iter=100,
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
            max_iter=20,
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


class TestAdaptiveMutation:
    """Test adaptive mutation rate adjustment."""

    def test_adaptive_mutation_improves(self):
        """Test adaptive mutation when making progress (decreases rate)."""
        def objective(bits):
            return sum(bits)

        # Start with all ones - easy to improve
        population = [tuple([1] * 10) for _ in range(20)]
        result = evolve(
            objective, population, simple_crossover, bit_mutate,
            max_iter=50, adaptive_mutation=True, seed=42
        )
        assert result.status == Status.FEASIBLE
        assert result.objective < 5  # Should make progress

    def test_adaptive_mutation_stagnation(self):
        """Test adaptive mutation when stagnating (increases rate)."""
        def objective(bits):
            # Very flat landscape - hard to improve
            return 10 if sum(bits) > 3 else sum(bits)

        # Start with sum=5 (returns 10), need to reduce to sum<=3
        population = [tuple([0] * 5 + [1] * 5) for _ in range(20)]
        result = evolve(
            objective, population, simple_crossover, bit_mutate,
            max_iter=50, adaptive_mutation=True, seed=42
        )
        assert result.status == Status.FEASIBLE
        # Adaptive mutation should help escape the penalty region
        assert result.objective <= 3  # Found a solution in the low-penalty region

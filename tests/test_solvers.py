"""Tests for all solvor solvers."""

from solvor.simplex import solve_lp, Status as LPStatus
from solvor.milp import solve_milp, Status as MILPStatus
from solvor.anneal import anneal, Status as AnnealStatus
from solvor.tabu import tabu_search, solve_tsp, Status as TabuStatus
from solvor.sat import solve_sat, Status as SATStatus
from solvor.cp import Model, Status as CPStatus
from solvor.bayesian import bayesian_opt, Status as BayesStatus
from solvor.genetic import evolve, Status as GAStatus
from solvor.flow import max_flow, min_cost_flow, solve_assignment, Status as FlowStatus
from solvor.gradient import gradient_descent, momentum, adam, Status as GradStatus


class TestSimplex:
    def test_basic_lp(self):
        # minimize x + 2y subject to x + y >= 2, x >= 0, y >= 0
        # Optimal: x=2, y=0, obj=2
        result = solve_lp(c=[1, 2], A=[[-1, -1]], b=[-2])
        assert result.status == LPStatus.OPTIMAL
        assert abs(result.objective - 2.0) < 1e-6

    def test_maximize(self):
        # maximize x + y subject to x + y <= 4, x <= 3
        result = solve_lp(c=[1, 1], A=[[1, 1], [1, 0]], b=[4, 3], minimize=False)
        assert result.status == LPStatus.OPTIMAL
        assert abs(result.objective - 4.0) < 1e-6

    def test_infeasible(self):
        # x >= 1, x <= 0 - infeasible
        result = solve_lp(c=[1], A=[[-1], [1]], b=[-1, 0])
        assert result.status == LPStatus.INFEASIBLE

    def test_unbounded(self):
        # minimize -x subject to x >= 0 - unbounded
        result = solve_lp(c=[-1], A=[[]], b=[])
        # With no constraints other than x >= 0, minimizing -x is unbounded
        # Actually need a constraint that allows unboundedness
        result = solve_lp(c=[-1, 0], A=[[0, 1]], b=[1])
        assert result.status == LPStatus.UNBOUNDED


class TestMILP:
    def test_basic_milp(self):
        # minimize x + y, x integer, x + y >= 2.5
        result = solve_milp(c=[1, 1], A=[[-1, -1]], b=[-2.5], integers=[0])
        assert result.status in (MILPStatus.OPTIMAL, MILPStatus.FEASIBLE)
        assert result.solution[0] == round(result.solution[0])  # x is integer

    def test_pure_integer(self):
        # minimize x + y, both integer, x + y >= 3
        result = solve_milp(c=[1, 1], A=[[-1, -1]], b=[-3], integers=[0, 1])
        assert result.status in (MILPStatus.OPTIMAL, MILPStatus.FEASIBLE)
        assert abs(result.objective - 3.0) < 1e-6


class TestAnneal:
    def test_simple_minimization(self):
        # Minimize x^2, starting from x=5
        def objective(x):
            return x[0] ** 2

        def neighbor(x):
            from random import gauss
            return [x[0] + gauss(0, 0.5)]

        result = anneal([5.0], objective, neighbor, max_iter=5000)
        assert result.status in (AnnealStatus.FEASIBLE, AnnealStatus.MAX_ITER)
        assert abs(result.solution[0]) < 1.0  # Should be near 0

    def test_maximize(self):
        # Maximize -x^2 (peak at 0)
        def objective(x):
            return -x[0] ** 2

        def neighbor(x):
            from random import gauss
            return [x[0] + gauss(0, 0.5)]

        result = anneal([5.0], objective, neighbor, minimize=False, max_iter=5000)
        assert abs(result.solution[0]) < 1.0


class TestTabu:
    def test_simple_search(self):
        # Simple discrete optimization
        def objective(x):
            return abs(x - 10)

        def neighbors(x):
            return [('dec', x - 1), ('inc', x + 1)]

        result = tabu_search(0, objective, neighbors, max_iter=100)
        assert result.status == TabuStatus.FEASIBLE
        assert result.solution == 10

    def test_tsp_small(self):
        # 4-city TSP
        dist = [
            [0, 10, 15, 20],
            [10, 0, 35, 25],
            [15, 35, 0, 30],
            [20, 25, 30, 0],
        ]
        result = solve_tsp(dist)
        assert result.status == TabuStatus.FEASIBLE
        assert len(result.solution) == 4
        assert set(result.solution) == {0, 1, 2, 3}

    def test_tsp_trivial(self):
        # 3-city TSP (trivial case)
        dist = [[0, 1, 2], [1, 0, 3], [2, 3, 0]]
        result = solve_tsp(dist)
        assert len(result.solution) == 3


class TestSAT:
    def test_satisfiable(self):
        # (x1 OR x2) AND (NOT x1 OR x2)
        # Satisfiable: x2=True
        result = solve_sat([[1, 2], [-1, 2]])
        assert result.status == SATStatus.OPTIMAL
        assert result.solution[2] is True

    def test_unsatisfiable(self):
        # x AND NOT x
        result = solve_sat([[1], [-1]])
        assert result.status == SATStatus.INFEASIBLE

    def test_empty(self):
        result = solve_sat([])
        assert result.status == SATStatus.OPTIMAL

    def test_three_sat(self):
        # Simple 3-SAT instance
        clauses = [[1, 2, 3], [-1, -2, 3], [1, -2, -3]]
        result = solve_sat(clauses)
        assert result.status == SATStatus.OPTIMAL
        # Verify solution satisfies all clauses
        for clause in clauses:
            satisfied = any(
                (lit > 0 and result.solution.get(abs(lit), False)) or
                (lit < 0 and not result.solution.get(abs(lit), False))
                for lit in clause
            )
            assert satisfied


class TestCP:
    def test_all_different(self):
        m = Model()
        x = m.int_var(1, 3, 'x')
        y = m.int_var(1, 3, 'y')
        z = m.int_var(1, 3, 'z')
        m.add(m.all_different([x, y, z]))
        result = m.solve()
        assert result.status == CPStatus.OPTIMAL
        vals = [result.solution['x'], result.solution['y'], result.solution['z']]
        assert len(set(vals)) == 3

    def test_sum_constraint(self):
        m = Model()
        x = m.int_var(1, 9, 'x')
        y = m.int_var(1, 9, 'y')
        m.add(m.sum_eq([x, y], 10))
        result = m.solve()
        assert result.status == CPStatus.OPTIMAL
        assert result.solution['x'] + result.solution['y'] == 10

    def test_infeasible(self):
        m = Model()
        x = m.int_var(1, 5, 'x')
        m.add(x == 10)  # impossible
        result = m.solve()
        assert result.status == CPStatus.INFEASIBLE


class TestBayesian:
    def test_simple_optimization(self):
        # Minimize (x-0.5)^2
        def objective(x):
            return (x[0] - 0.5) ** 2

        result = bayesian_opt(objective, bounds=[(0, 1)], max_iter=20, seed=42)
        assert result.status == BayesStatus.FEASIBLE
        assert abs(result.solution[0] - 0.5) < 0.3

    def test_2d_optimization(self):
        # Minimize (x-0.3)^2 + (y-0.7)^2
        def objective(x):
            return (x[0] - 0.3) ** 2 + (x[1] - 0.7) ** 2

        result = bayesian_opt(objective, bounds=[(0, 1), (0, 1)], max_iter=30, seed=42)
        assert result.status == BayesStatus.FEASIBLE


class TestGenetic:
    def test_simple_evolution(self):
        # Minimize sum of bits (find all zeros)
        def objective(bits):
            return sum(bits)

        def crossover(p1, p2):
            mid = len(p1) // 2
            return p1[:mid] + p2[mid:]

        def mutate(bits):
            from random import randint
            bits = list(bits)
            i = randint(0, len(bits) - 1)
            bits[i] = 1 - bits[i]
            return tuple(bits)

        population = [tuple([1] * 10) for _ in range(20)]
        result = evolve(objective, population, crossover, mutate, max_gen=50, seed=42)
        assert result.status == GAStatus.FEASIBLE
        assert result.objective < 5  # Should find mostly zeros


class TestFlow:
    def test_max_flow_simple(self):
        graph = {
            's': [('t', 10, 0)],
            't': []
        }
        result = max_flow(graph, 's', 't')
        assert result.status == FlowStatus.OPTIMAL
        assert result.objective == 10

    def test_max_flow_network(self):
        graph = {
            's': [('a', 10, 0), ('b', 5, 0)],
            'a': [('t', 10, 0)],
            'b': [('t', 10, 0)],
            't': []
        }
        result = max_flow(graph, 's', 't')
        assert result.status == FlowStatus.OPTIMAL
        assert result.objective == 15  # 10 through a, 5 through b

    def test_assignment(self):
        # 3 workers, 3 tasks
        costs = [
            [10, 5, 13],
            [3, 9, 18],
            [10, 6, 12]
        ]
        result = solve_assignment(costs)
        assert result.status == FlowStatus.OPTIMAL
        # Each worker assigned to exactly one task
        assert len(result.solution) == 3
        assert set(result.solution) == {0, 1, 2}

    def test_min_cost_flow(self):
        graph = {
            's': [('a', 10, 1), ('b', 10, 2)],
            'a': [('t', 10, 1)],
            'b': [('t', 10, 1)],
            't': []
        }
        result = min_cost_flow(graph, 's', 't', 5)
        assert result.status == FlowStatus.OPTIMAL
        # Should prefer path through 'a' (cost 1+1=2 vs 2+1=3)


class TestGradient:
    def test_gradient_descent_quadratic(self):
        # Minimize x^2 + y^2
        def grad(x):
            return [2 * x[0], 2 * x[1]]

        result = gradient_descent(grad, [5.0, 5.0], max_iter=1000)
        assert result.status in (GradStatus.OPTIMAL, GradStatus.MAX_ITER)
        assert abs(result.solution[0]) < 0.1
        assert abs(result.solution[1]) < 0.1

    def test_momentum(self):
        def grad(x):
            return [2 * x[0], 2 * x[1]]

        result = momentum(grad, [5.0, 5.0], max_iter=1000)
        assert abs(result.solution[0]) < 0.1
        assert abs(result.solution[1]) < 0.1

    def test_adam(self):
        def grad(x):
            return [2 * x[0], 2 * x[1]]

        # Adam with default lr=0.001 needs more iterations or higher lr
        result = adam(grad, [5.0, 5.0], lr=0.1, max_iter=1000)
        assert abs(result.solution[0]) < 0.1
        assert abs(result.solution[1]) < 0.1

    def test_maximize(self):
        # Maximize -x^2 (gradient is -2x, want to go uphill)
        def grad(x):
            return [-2 * x[0]]

        result = gradient_descent(grad, [5.0], minimize=False, max_iter=1000)
        assert abs(result.solution[0]) < 0.1


class TestStatusConsistency:
    """Verify all modules export consistent Status enum."""

    def test_status_values(self):
        # All Status enums should have the same values
        statuses = [LPStatus, MILPStatus, AnnealStatus, TabuStatus,
                    SATStatus, CPStatus, BayesStatus, GAStatus, FlowStatus, GradStatus]

        for status in statuses:
            assert hasattr(status, 'OPTIMAL')
            assert hasattr(status, 'FEASIBLE')
            assert hasattr(status, 'INFEASIBLE')
            assert hasattr(status, 'UNBOUNDED')
            assert hasattr(status, 'MAX_ITER')

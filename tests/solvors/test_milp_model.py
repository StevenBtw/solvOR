"""Tests for MilpModel (incremental MILP with warm re-solves) and solve_lexicographic."""

import math
import random

import pytest

from solvor import MilpModel, solve_lexicographic, solve_milp
from solvor.types import Status


def _random_rows(rng, n, m):
    rows = [{j: float(rng.randint(1, 6)) for j in range(n) if rng.random() < 0.4} or {0: 1.0} for _ in range(m)]
    rhs = [float(rng.randint(3, 15)) for _ in range(m)]
    return rows, rhs


def _same(a, b):
    if a.status != b.status:
        return False
    return a.status not in (Status.OPTIMAL, Status.FEASIBLE) or abs(a.objective - b.objective) < 1e-6


def _odd_cycle(n):
    return [{i: 1.0, (i + 1) % n: 1.0} for i in range(n)]


class TestMilpModel:
    def test_single_solve_equals_solve_milp(self):
        rng = random.Random(1)
        for _ in range(60):
            n, m = rng.randint(2, 12), rng.randint(1, 8)
            rows, rhs = _random_rows(rng, n, m)
            c = [float(rng.randint(-3, 9)) for _ in range(n)]
            model = MilpModel(n, binary=range(n))
            model.add_rows(rows, rhs)
            assert _same(model.solve(c, minimize=False), solve_milp(c, rows, rhs, binary=range(n), minimize=False))

    def test_growing_model_equals_fresh_solves(self):
        """Rows added and objectives changed between solves: each solve matches a fresh solve_milp."""
        rng = random.Random(2)
        for _ in range(40):
            n = rng.randint(3, 12)
            integers = sorted(rng.sample(range(n), rng.randint(1, n)))
            ub = [float(rng.randint(1, 4)) for _ in range(n)]
            model = MilpModel(n, integers=integers, ub=ub)
            all_rows, all_rhs = [], []
            for _ in range(5):
                rows, rhs = _random_rows(rng, n, rng.randint(1, 3))
                model.add_rows(rows, rhs)
                all_rows += rows
                all_rhs += rhs
                c = [float(rng.randint(-3, 9)) for _ in range(n)]
                minimize = rng.random() < 0.3
                warm = model.solve(c, minimize=minimize)
                fresh = solve_milp(c, all_rows, all_rhs, integers, ub=ub, minimize=minimize)
                assert _same(warm, fresh)

    def test_lazy_cut_moves_the_optimum(self):
        model = MilpModel(3, binary=range(3))
        model.add_rows([{0: 1, 1: 1}, {1: 1, 2: 1}], [1, 1])
        first = model.solve([3, 2, 3], minimize=False)
        assert first.solution == (1.0, 0.0, 1.0)
        model.add_rows([{0: 1, 2: 1}], [1])  # cut: not both 0 and 2; now every pair conflicts
        second = model.solve([3, 2, 3], minimize=False)
        assert second.objective == 3.0
        assert second.solution in ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        fresh = solve_milp(
            [3, 2, 3], [{0: 1, 1: 1}, {1: 1, 2: 1}, {0: 1, 2: 1}], [1, 1, 1], binary=range(3), minimize=False
        )
        assert fresh.objective == second.objective

    def test_previous_solution_seeds_the_incumbent(self):
        model = MilpModel(5, binary=range(5))
        model.add_rows(_odd_cycle(5), [1.0] * 5)  # LP optimum 2.5, integer optimum 2
        assert model.solve([1.0] * 5, minimize=False).objective == 2.0
        # No nodes, no heuristics: the only possible incumbent is the previous solution
        again = model.solve([1.0] * 5, minimize=False, max_nodes=0, heuristics=False)
        assert again.status == Status.FEASIBLE
        assert again.objective == 2.0

    def test_single_variable_row_after_a_solve_tightens_a_bound(self):
        model = MilpModel(2, integers=[0, 1], ub=[5, 5])
        model.add_rows([{0: 1, 1: 1}], [8])
        assert model.solve([2, 1], minimize=False).solution == (5.0, 3.0)
        model.add_rows([{0: 1}], [2.5])  # becomes x0 <= 2
        assert model.solve([2, 1], minimize=False).solution == (2.0, 5.0)
        assert model.n_rows == 1

    def test_infeasible_rows(self):
        model = MilpModel(2, binary=range(2))
        model.add_rows([{0: 2, 1: 2}], [3], ["="])  # no integer solution
        assert model.solve([1, 1]).status == Status.INFEASIBLE
        empty = MilpModel(1, binary=[0])
        empty.add_rows([[0]], [-1])
        assert empty.solve([1]).status == Status.INFEASIBLE

    def test_validation(self):
        model = MilpModel(2, binary=range(2))
        with pytest.raises(ValueError, match="expected 2 elements in c"):
            model.solve([1, 2, 3])
        with pytest.raises(ValueError, match="c contains NaN"):
            model.solve([1, math.nan])
        with pytest.raises(ValueError, match="column 2 outside"):
            model.add_rows([{2: 1.0}], [1])
        with pytest.raises(ValueError, match="Invalid index in binary"):
            MilpModel(2, binary=[2])

    def test_changing_eps_between_solves(self):
        """The kept LP is tied to eps and max_iter; a different value rebuilds it."""
        model = MilpModel(5, binary=range(5))
        model.add_rows(_odd_cycle(5), [1.0] * 5)
        assert model.solve([1.0] * 5, minimize=False).objective == 2.0
        assert model.solve([1.0] * 5, minimize=False, eps=1e-9).objective == 2.0
        assert model.solve([1.0] * 5, minimize=False, max_iter=50).objective == 2.0

    def test_same_calls_give_identical_results(self):
        """Determinism: the same sequence of calls gives bit-identical results."""

        def run():
            rng = random.Random(9)
            model = MilpModel(15, binary=range(15))
            out = []
            for _ in range(4):
                rows, rhs = _random_rows(rng, 15, 3)
                model.add_rows(rows, rhs)
                c = [float(rng.randint(-3, 9)) for _ in range(15)]
                result = model.solve(c, minimize=False)
                out.append((result.status, result.objective, result.solution, result.iterations, result.evaluations))
            return out

        assert run() == run()


class TestSolveLexicographic:
    def _tiers(self, seed):
        rng = random.Random(seed)
        n = 24
        rows = [{j: 1.0 for j in rng.sample(range(n), rng.randint(2, 4))} for _ in range(10)]
        tiers = [rng.choice((1, 2, 3)) for _ in range(n)]
        objectives = [[1.0 if t == k else 0.0 for t in tiers] for k in (1, 2, 3)]
        return n, rows, objectives

    def test_matches_a_weighted_single_objective(self):
        """Lexicographic tier counts equal the optimum of weights (n+1)^2, n+1, 1."""
        for seed in range(5):
            n, rows, objectives = self._tiers(seed)
            model = MilpModel(n, binary=range(n))
            model.add_rows(rows, [1.0] * len(rows))
            lex = solve_lexicographic(model, objectives, minimize=False)
            w = [(n + 1) ** 2, n + 1, 1]
            weighted = [sum(wk * obj[j] for wk, obj in zip(w, objectives)) for j in range(n)]
            single = solve_milp(weighted, rows, [1.0] * len(rows), binary=range(n), minimize=False)
            assert lex.status == Status.OPTIMAL
            for obj in objectives:
                assert sum(a * x for a, x in zip(obj, lex.solution)) == sum(a * x for a, x in zip(obj, single.solution))

    def test_matches_manual_stages(self):
        n, rows, objectives = self._tiers(11)
        model = MilpModel(n, binary=range(n))
        model.add_rows(rows, [1.0] * len(rows))
        lex = solve_lexicographic(model, objectives, minimize=False)
        stage_rows, stage_rhs, senses = list(rows), [1.0] * len(rows), ["<="] * len(rows)
        for k, obj in enumerate(objectives):
            r = solve_milp(obj, stage_rows, stage_rhs, binary=range(n), minimize=False, senses=senses)
            if k < len(objectives) - 1:
                stage_rows.append(dict(enumerate(obj)))
                stage_rhs.append(r.objective)
                senses.append(">=")
        assert lex.objective == r.objective

    def test_minimize_with_tolerance(self):
        # Stage 1 minimizes x0 + x1 >= 1; with tol=1 stage 2 may use x0 + x1 = 2
        model = MilpModel(2, binary=range(2))
        model.add_rows([{0: 1, 1: 1}], [1], [">="])
        strict = solve_lexicographic(model, [[1, 1], [-1, -1]])
        assert strict.objective == -1.0
        loose_model = MilpModel(2, binary=range(2))
        loose_model.add_rows([{0: 1, 1: 1}], [1], [">="])
        loose = solve_lexicographic(loose_model, [[1, 1], [-1, -1]], tol=1.0)
        assert loose.objective == -2.0

    def test_stops_at_an_infeasible_stage(self):
        model = MilpModel(1, binary=[0])
        model.add_rows([{0: 1}], [2], [">="])
        result = solve_lexicographic(model, [[1], [1]])
        assert result.status == Status.INFEASIBLE


class TestLexicographicStatus:
    def test_a_feasible_stage_is_not_reported_as_optimal(self):
        """A node limit in an earlier stage means the final answer is not proven optimal."""
        rng = random.Random(0)
        n = 35
        rows = [{j: float(rng.randint(5, 40)) for j in range(n)} for _ in range(4)]
        rhs = [sum(r.values()) * 0.35 for r in rows]
        c = [float(rng.randint(10, 60)) for _ in range(n)]
        model = MilpModel(n, binary=range(n))
        model.add_rows(rows, rhs)
        result = solve_lexicographic(model, [c, [0.0] * n], minimize=False, max_nodes=20)
        assert result.status == Status.FEASIBLE

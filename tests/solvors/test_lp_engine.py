"""Tests for the LP engine's warm re-solves (bound changes, added rows, new objectives)."""

import math
import random

import pytest

from solvor.lp_engine import WarmLP, solve_cold
from solvor.types import Status
from solvor.utils.lp_input import LinearProblem


def _cold(lp: WarmLP, c, minimize):
    prob = LinearProblem(lp.n, lp.rows, lp.b, lp.senses, lp.lb, lp.ub)
    return solve_cold(prob, c, lp.lb, lp.ub, minimize=minimize, eps=1e-9, max_iter=10_000)


def _same(warm, cold):
    if warm.status != cold.status:
        return False
    return cold.status != Status.OPTIMAL or abs(warm.objective - cold.objective) <= 1e-6 * (1 + abs(cold.objective))


class TestWarmMatchesCold:
    """Every warm re-solve must agree with a cold solve of the same LP (status and optimum)."""

    @pytest.mark.parametrize("seed", [1, 2])
    def test_branch_and_bound_like_sequences(self, seed):
        rng = random.Random(seed)
        warm_solves = 0
        for _ in range(120):
            n, m = rng.randint(2, 10), rng.randint(1, 8)
            rows = [{j: float(rng.randint(0, 6)) for j in range(n) if rng.random() < 0.5} for _ in range(m)]
            b = [float(rng.randint(3, 25)) for _ in range(m)]
            senses = ["<="] * m
            if rng.random() < 0.4:
                senses[0], b[0] = ">=", float(rng.randint(0, 3))
            lb0 = [0.0] * n
            ub0 = [float(rng.randint(1, 5)) if rng.random() > 0.1 else math.inf for _ in range(n)]
            lp = WarmLP(LinearProblem(n, rows, b, senses, lb0, ub0), lb0, ub0, eps=1e-9, max_iter=10_000)
            c = [float(rng.randint(-6, 6)) for _ in range(n)]
            minimize = rng.random() < 0.5
            lo, hi = list(lb0), list(ub0)
            for step in range(20):
                r = rng.random()
                if step and r < 0.75:
                    j = rng.randrange(n)
                    top = int(ub0[j]) if ub0[j] < math.inf else 6
                    a, z = sorted((float(rng.randint(0, top)), float(rng.randint(0, top))))
                    lo[j], hi[j] = a, z if rng.random() < 0.9 else ub0[j]
                    if rng.random() < 0.3:
                        lo[j], hi[j] = lb0[j], ub0[j]
                    lp.set_bounds(lo, hi)
                elif step and r < 0.85:
                    row = {j: float(rng.randint(0, 4)) for j in range(n) if rng.random() < 0.4}
                    lp.add_row(row, float(rng.randint(4, 30)), "<=")
                elif step:
                    c = [float(rng.randint(-6, 6)) for _ in range(n)]
                before = lp.cold_solves
                warm = lp.solve(c, minimize=minimize)
                warm_solves += lp.cold_solves == before
                assert _same(warm, _cold(lp, c, minimize))
        assert warm_solves > 1500  # most re-solves really are warm

    def test_general_sequences_with_free_variables_and_equalities(self):
        rng = random.Random(7)
        for _ in range(150):
            n, m = rng.randint(1, 6), rng.randint(0, 5)
            rows = [{j: float(rng.randint(-3, 5)) for j in range(n) if rng.random() < 0.6} for _ in range(m)]
            b = [float(rng.randint(-2, 15)) for _ in range(m)]
            senses = [rng.choice(("<=", "<=", ">=", "=")) for _ in range(m)]
            lb = [float(rng.randint(-4, 1)) if rng.random() < 0.85 else -math.inf for _ in range(n)]
            ub = [(lo if lo > -math.inf else -2.0) + rng.randint(0, 8) if rng.random() < 0.8 else math.inf for lo in lb]
            lp = WarmLP(LinearProblem(n, rows, b, senses, lb, ub), lb, ub, eps=1e-9, max_iter=10_000)
            c = [float(rng.randint(-5, 5)) for _ in range(n)]
            minimize = True
            for step in range(8):
                r = rng.random()
                if step and r < 0.4:
                    j = rng.randrange(n)
                    new_lb, new_ub = list(lp.lb), list(lp.ub)
                    new_ub[j] = (new_lb[j] if new_lb[j] > -math.inf else -3.0) + rng.randint(0, 6)
                    lp.set_bounds(new_lb, new_ub)
                elif step and r < 0.7:
                    row = {j: float(rng.randint(-3, 5)) for j in range(n) if rng.random() < 0.6}
                    lp.add_row(row, float(rng.randint(-2, 15)), rng.choice(("<=", ">=", "=")))
                elif step:
                    minimize = not minimize
                assert _same(lp.solve(c, minimize=minimize), _cold(lp, c, minimize))


class TestWarmOperations:
    def _lp(self):
        # maximize x + y subject to x + 2y <= 4, x <= 3 (bound), y <= 2 (bound)
        prob = LinearProblem(2, [{0: 1.0, 1: 2.0}], [4.0], ["<="], [0.0, 0.0], [3.0, 2.0])
        return WarmLP(prob, prob.lb, prob.ub, eps=1e-9, max_iter=1000)

    def test_bound_change_resolves_warm(self):
        lp = self._lp()
        assert lp.solve([1, 1], minimize=False).objective == 3.5
        lp.set_bounds([0.0, 0.0], [1.0, 2.0])
        result = lp.solve([1, 1], minimize=False)
        assert result.objective == 2.5
        assert lp.cold_solves == 1  # only the first solve was cold

    def test_added_row_resolves_warm(self):
        lp = self._lp()
        lp.solve([1, 1], minimize=False)
        lp.add_row({0: 1.0, 1: 1.0}, 2.0, "<=")
        result = lp.solve([1, 1], minimize=False)
        assert abs(result.objective - 2.0) < 1e-9
        assert lp.cold_solves == 1

    def test_added_equality_row(self):
        lp = self._lp()
        lp.solve([1, 1], minimize=False)
        lp.add_row({0: 1.0, 1: -1.0}, 0.0, "=")  # x == y
        result = lp.solve([1, 1], minimize=False)
        x, y = result.solution
        assert abs(x - y) < 1e-9
        assert abs(result.objective - 8 / 3) < 1e-9

    def test_added_row_makes_it_infeasible(self):
        lp = self._lp()
        lp.solve([1, 1], minimize=False)
        lp.add_row({0: 1.0, 1: 1.0}, 6.0, ">=")
        assert lp.solve([1, 1], minimize=False).status == Status.INFEASIBLE

    def test_new_objective_resolves_warm(self):
        lp = self._lp()
        lp.solve([1, 1], minimize=False)
        result = lp.solve([0, 1], minimize=False)
        assert result.objective == 2.0
        assert lp.cold_solves == 1

    def test_inverted_bounds(self):
        lp = self._lp()
        lp.solve([1, 1], minimize=False)
        lp.set_bounds([2.0, 0.0], [1.0, 2.0])
        assert lp.solve([1, 1], minimize=False).status == Status.INFEASIBLE

    def test_loosening_to_infinity_falls_back_to_cold(self):
        """An upper bound that disappears under a column sitting at it cannot be warm-started."""
        lp = self._lp()
        lp.solve([1, 1], minimize=False)
        lp.set_bounds([0.0, 0.0], [math.inf, 2.0])
        result = lp.solve([1, 1], minimize=False)
        assert result.objective == 4.0  # x = 4, y = 0
        assert result.status == Status.OPTIMAL


def _coef(rng, mode):
    if mode == 0:
        return float(rng.randint(-3, 6))
    if mode == 1:
        return round(rng.uniform(-5, 10), 3)
    if mode == 2:
        return float(rng.choice([0, 0, 1, 1, 1, -1, 2]))
    return round(rng.uniform(-1, 1) * 10 ** rng.randint(-1, 3), 4)


class TestWarmMatchesColdAtMilpTolerance:
    """eps = 1e-6 (the MILP default) with float coefficients of mixed magnitude.

    A dual pivot on a tiny element can lose dual feasibility; the warm path must
    still end at a true optimum (it runs the primal simplex after the dual).
    """

    @pytest.mark.parametrize("seed", [0, 1])
    def test_bound_changes_rows_and_objectives(self, seed):
        rng = random.Random(seed)
        for _ in range(260):
            mode = rng.randrange(4)
            n, m = rng.randint(1, 9), rng.randint(0, 8)
            dens = rng.uniform(0.3, 0.9)
            rows = [{j: _coef(rng, mode) for j in range(n) if rng.random() < dens} for _ in range(m)]
            rows = [{j: a for j, a in r.items() if a != 0.0} for r in rows]
            b = [_coef(rng, mode) * rng.randint(1, 4) for _ in range(m)]
            senses = [rng.choice(("<=", "<=", ">=", "=")) for _ in range(m)]
            lb, ub = [], []
            for _ in range(n):
                k, lo = rng.random(), float(rng.randint(-4, 2))
                if k < 0.6:
                    lb.append(lo)
                    ub.append(lo + rng.randint(0, 6))
                elif k < 0.75:
                    lb.append(lo)
                    ub.append(math.inf)
                elif k < 0.9:
                    lb.append(-math.inf)
                    ub.append(lo + rng.randint(0, 6))
                else:
                    lb.append(-math.inf)
                    ub.append(math.inf)
            lp = WarmLP(LinearProblem(n, rows, b, senses, lb, ub), lb, ub, eps=1e-6, max_iter=20_000)
            c = [_coef(rng, mode) for _ in range(n)]
            minimize = rng.random() < 0.5
            lo0, hi0 = list(lb), list(ub)
            for step in range(15):
                r = rng.random()
                if step and r < 0.55:
                    nl, nu = list(lp.lb), list(lp.ub)
                    for _ in range(rng.randint(1, 3)):
                        j, kind = rng.randrange(n), rng.random()
                        if kind < 0.15:
                            nl[j], nu[j] = lo0[j], hi0[j]
                        elif kind < 0.25:
                            if rng.random() < 0.5:
                                nu[j] = math.inf
                            else:
                                nl[j] = -math.inf
                        elif kind < 0.35:
                            nl[j] = nu[j] = float(rng.randint(-3, 4))
                        else:
                            v = rng.uniform(-4, 6)
                            if rng.random() < 0.5:
                                nu[j] = float(math.floor(v))
                            else:
                                nl[j] = float(math.ceil(v))
                    lp.set_bounds(nl, nu)
                elif step and r < 0.75:
                    row = {j: _coef(rng, mode) for j in range(n) if rng.random() < dens}
                    row = {j: a for j, a in row.items() if a != 0.0}
                    lp.add_row(row, _coef(rng, mode) * rng.randint(1, 4), rng.choice(("<=", ">=", "=")))
                elif step and r < 0.9:
                    c = [_coef(rng, mode) for _ in range(n)]
                elif step:
                    minimize = not minimize
                warm = lp.solve(c, minimize=minimize)
                prob = LinearProblem(lp.n, lp.rows, lp.b, lp.senses, lp.lb, lp.ub)
                cold = solve_cold(prob, c, lp.lb, lp.ub, minimize=minimize, eps=1e-6, max_iter=20_000)
                assert _same(warm, cold)


class TestRebuildCounter:
    def test_cold_pivots_do_not_count_towards_a_rebuild(self, monkeypatch):
        """Only pivots made after the last cold build count; otherwise big LPs would never warm-start."""
        monkeypatch.setattr(WarmLP, "REBUILD_PIVOTS", 1)
        rows = [{j: 1.0 for j in range(k, k + 3)} for k in range(8)]
        prob = LinearProblem(10, rows, [2.0] * 8, ["<="] * 8, [0.0] * 10, [1.0] * 10)
        lp = WarmLP(prob, prob.lb, prob.ub, eps=1e-9, max_iter=1000)
        first = lp.solve([1.0] * 10, minimize=False)
        assert first.status == Status.OPTIMAL and first.iterations > 1  # the cold solve pivoted more than the limit
        lp.set_bounds([0.0] * 10, [1.0] * 9 + [0.0])
        assert lp.solve([1.0] * 10, minimize=False).status == Status.OPTIMAL
        assert (lp.cold_solves, lp.warm_solves) == (1, 1)

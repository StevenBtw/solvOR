"""Tests for the LP engine's warm re-solves (bound changes, added rows, new objectives)."""

import math
import random

import pytest

from solvor.lp_engine import BoundedSimplex, Standardized, WarmLP, kernel_class, solve_cold
from solvor.rust import rust_available
from solvor.simplex import solve_lp
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
        assert warm_solves > 1400  # most re-solves really are warm (infeasible verdicts are confirmed cold)

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
        assert lp.cold_solves == 2


def _coef(rng, mode):
    if mode == 0:
        return float(rng.randint(-3, 6))
    if mode == 1:
        return round(rng.uniform(-5, 10), 3)
    if mode == 2:
        return float(rng.choice([0, 0, 1, 1, 1, -1, 2]))
    return round(rng.uniform(-1, 1) * 10 ** rng.randint(-1, 3), 4)


def _tolerance_trials(seed, trials=260):
    """Random LPs with B&B-like edit scripts, float coefficients of mixed magnitude.

    Yields (problem, steps). A step is ("bounds", lb, ub), ("row", row, rhs, sense)
    or ("solve", c, minimize); bound edits start from the current bounds.
    """
    rng = random.Random(seed)
    for _ in range(trials):
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
        prob = LinearProblem(n, rows, b, senses, lb, ub)
        c = [_coef(rng, mode) for _ in range(n)]
        minimize = rng.random() < 0.5
        lo0, hi0 = list(lb), list(ub)
        cur_lb, cur_ub = list(lb), list(ub)
        steps = []
        for step in range(15):
            r = rng.random()
            if step and r < 0.55:
                nl, nu = list(cur_lb), list(cur_ub)
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
                steps.append(("bounds", nl, nu))
                cur_lb, cur_ub = nl, nu
            elif step and r < 0.75:
                row = {j: _coef(rng, mode) for j in range(n) if rng.random() < dens}
                row = {j: a for j, a in row.items() if a != 0.0}
                steps.append(("row", row, _coef(rng, mode) * rng.randint(1, 4), rng.choice(("<=", ">=", "="))))
            elif step and r < 0.9:
                c = [_coef(rng, mode) for _ in range(n)]
            elif step:
                minimize = not minimize
            steps.append(("solve", c, minimize))
        yield prob, steps


def _edit(lp: WarmLP, step) -> None:
    if step[0] == "bounds":
        lp.set_bounds(step[1], step[2])
    else:
        lp.add_row(step[1], step[2], step[3])


class TestWarmMatchesColdAtMilpTolerance:
    """eps = 1e-6 (the MILP default) with float coefficients of mixed magnitude.

    A dual pivot on a tiny element can lose dual feasibility; the warm path must
    still end at a true optimum (it runs the primal simplex after the dual).
    """

    @pytest.mark.parametrize("seed", [0, 1])
    def test_bound_changes_rows_and_objectives(self, seed):
        for prob, steps in _tolerance_trials(seed):
            lp = WarmLP(prob, prob.lb, prob.ub, eps=1e-6, max_iter=20_000)
            for step in steps:
                if step[0] != "solve":
                    _edit(lp, step)
                    continue
                _, c, minimize = step
                warm = lp.solve(c, minimize=minimize)
                now = LinearProblem(lp.n, lp.rows, lp.b, lp.senses, lp.lb, lp.ub)
                cold = solve_cold(now, c, lp.lb, lp.ub, minimize=minimize, eps=1e-6, max_iter=20_000)
                assert _same(warm, cold)


class TestFinalCheckTolerance:
    def test_follows_a_looser_eps(self):
        """The dual simplex accepts violations up to eps; the final check must too, or every warm solve rebuilds."""
        prob = LinearProblem(2, [{0: 1.0, 1: 1.0}], [1.0], ["<="], [0.0, 0.0], [1.0, 1.0])
        loose = WarmLP(prob, prob.lb, prob.ub, eps=1e-4, max_iter=100)
        assert loose._feasible([1.00005, 0.0])
        assert not loose._feasible([1.0005, 0.0])
        tight = WarmLP(prob, prob.lb, prob.ub, eps=1e-9, max_iter=100)
        assert not tight._feasible([1.00005, 0.0])


class TestIterationBudget:
    def test_warm_solves_stay_within_max_iter(self):
        """The dual and primal passes of a warm solve share one max_iter budget."""
        for prob, steps in _tolerance_trials(5, trials=40):  # trial 35 used 4 iterations before
            lp = WarmLP(prob, prob.lb, prob.ub, eps=1e-6, max_iter=3)
            for step in steps:
                if step[0] != "solve":
                    _edit(lp, step)
                    continue
                assert lp.solve(step[1], minimize=step[2]).iterations <= 3


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


def _bits(value):
    """Floats as hex strings, recursively, so equality means bit for bit (-0.0 differs from 0.0)."""
    if isinstance(value, float):
        return value.hex()
    if isinstance(value, (list, tuple)):
        return [_bits(v) for v in value]
    return value


def _result_bits(r):
    return [r.status, _bits(r.objective), r.iterations, r.evaluations, _bits(r.solution)]


def _rebuild_state(lp: WarmLP):
    return lp.cold_solves, lp.warm_solves, lp.lp.pivots if lp.lp else None


needs_rust = pytest.mark.skipif(not rust_available(), reason="Rust extension not built")


@needs_rust
class TestRustKernelMatchesPython:
    """The Rust BoundedSimplex must reproduce the Python one bit for bit, step by step."""

    @pytest.mark.parametrize(("seed", "eps"), [(0, 1e-6), (1, 1e-6), (2, 1e-9)])
    def test_warm_sequences(self, seed, eps):
        for prob, steps in _tolerance_trials(seed):
            python_lp = WarmLP(prob, prob.lb, prob.ub, eps=eps, max_iter=20_000, backend="python")
            rust_lp = WarmLP(prob, prob.lb, prob.ub, eps=eps, max_iter=20_000, backend="rust")
            for step in steps:
                if step[0] != "solve":
                    _edit(python_lp, step)
                    _edit(rust_lp, step)
                    continue
                _, c, minimize = step
                expected = _result_bits(python_lp.solve(c, minimize=minimize))
                assert _result_bits(rust_lp.solve(c, minimize=minimize)) == expected
                assert _rebuild_state(rust_lp) == _rebuild_state(python_lp)

    @pytest.mark.parametrize("seed", range(4))
    def test_kernel_methods(self, seed):
        """Every kernel method, in random order and with small iteration limits, on both kernels."""
        rng = random.Random(seed)
        rust_kernel = kernel_class("rust")
        for prob, _ in _tolerance_trials(seed, trials=150):
            std = Standardized(prob, prob.lb, prob.ub)
            if std.impossible:
                continue
            args = (std.rows, std.rhs, std.is_eq, std.upper, rng.choice((1e-6, 1e-9)))
            kernels = [BoundedSimplex(*args), rust_kernel(*args)]
            cost = std.map_cost([rng.uniform(-5, 5) for _ in range(prob.n)])
            limit = rng.choice((0, 1, 2, 3, 1000))
            outputs = [k.solve(cost, limit) for k in kernels]
            for step in range(13):  # 12 random calls, each one checked
                assert outputs[0] == outputs[1]
                assert _bits(kernels[0].column_values()) == _bits(kernels[1].column_values())
                states = [(k.pivots, k.phase1_done, k.m, k.width) for k in kernels]
                assert states[0] == states[1]
                if step == 12 or not kernels[0].phase1_done:
                    break
                op = rng.randrange(6)
                if op == 0:
                    j = rng.randrange(std.n_cols)
                    lo = float(rng.randint(-2, 2))
                    hi = rng.choice((lo, lo + rng.randint(1, 4), math.inf))
                    outputs = [k.set_bounds(j, lo, hi) for k in kernels]
                elif op == 1:
                    columns = list(range(prob.n))
                    rng.shuffle(columns)  # key order is part of the input, so keep it unsorted
                    row = {j: _coef(rng, rng.randrange(4)) for j in columns if rng.random() < 0.5}
                    coefs, r, eq = std.map_row(row, float(rng.randint(-3, 6)), rng.choice(("<=", ">=", "=")))
                    if math.isinf(r):
                        continue
                    outputs = [k.add_row(coefs, r, eq) for k in kernels]
                elif op == 2:
                    outputs = [k.make_dual_feasible() for k in kernels]
                elif op == 3:
                    limit = rng.choice((0, 1, 2, 1000))
                    outputs = [k.dual(limit) for k in kernels]
                elif op == 4:
                    limit = rng.choice((0, 1, 2, 1000))
                    outputs = [k.primal(limit) for k in kernels]
                else:
                    cost = std.map_cost([rng.uniform(-5, 5) for _ in range(prob.n)])
                    outputs = [k.set_objective(cost) for k in kernels]

    def test_add_row_keeps_the_key_order(self):
        """The new row's right-hand side is summed in dict order, which changes its last bits."""
        slack = {}
        for keys in ((0, 1, 2), (2, 1, 0)):
            for backend in ("python", "rust"):
                lp = kernel_class(backend)([], [], [], [1.0, 1.0, 1.0], 1e-9)
                lp.solve([1.0, 1.0, 1.0], 100)
                for j, lo in enumerate((0.1, 0.2, 0.7)):
                    lp.set_bounds(j, lo, 1.0)
                lp.add_row({j: 1.0 for j in keys}, 1.0, False)
                slack[keys, backend] = lp.column_values()[3].hex()
        assert slack[(0, 1, 2), "python"] != slack[(2, 1, 0), "python"]
        assert slack[(0, 1, 2), "rust"] == slack[(0, 1, 2), "python"]
        assert slack[(2, 1, 0), "rust"] == slack[(2, 1, 0), "python"]

    def test_statuses_are_status_members(self):
        lp = kernel_class("rust")([{0: 1.0, 1: 1.0}], [4.0], [False], [3.0, math.inf], 1e-9)
        status, iterations = lp.solve([-1.0, -1.0], 100)
        assert status is Status.OPTIMAL and iterations == 2

    def test_pivot_counter_can_be_reset(self):
        lp = kernel_class("rust")([{0: 1.0, 1: 1.0}], [4.0], [False], [3.0, math.inf], 1e-9)
        lp.solve([-1.0, -1.0], 100)
        assert lp.pivots > 0
        lp.pivots = 0
        assert lp.pivots == 0

    def test_bad_input_raises(self):
        rust_kernel = kernel_class("rust")
        with pytest.raises(ValueError, match="differ in length"):
            rust_kernel([{0: 1.0}], [1.0, 2.0], [False], [1.0], 1e-9)
        with pytest.raises(IndexError, match="out of range"):
            rust_kernel([{5: 1.0}], [1.0], [False], [1.0], 1e-9)
        lp = rust_kernel([{0: 1.0}], [1.0], [False], [1.0], 1e-9)
        with pytest.raises(IndexError, match="out of range"):
            lp.set_bounds(lp.width, 0.0, 1.0)
        with pytest.raises(IndexError, match="out of range"):
            lp.add_row({lp.width: 1.0}, 1.0, False)


class TestKernelSelection:
    def test_python_backend_never_loads_rust(self, monkeypatch):
        def no_rust():
            raise AssertionError("Rust kernel requested")

        monkeypatch.setattr("solvor.lp_engine.get_rust_module", no_rust)
        assert kernel_class("python") is BoundedSimplex
        prob = LinearProblem(2, [{0: 1.0, 1: 1.0}], [4.0], ["<="], [0.0, 0.0], [3.0, math.inf])
        result = solve_cold(
            prob, [1.0, 1.0], prob.lb, prob.ub, minimize=False, eps=1e-9, max_iter=100, backend="python"
        )
        assert result.objective == 4.0

    def test_rust_backend_without_extension_raises(self, monkeypatch):
        monkeypatch.setattr("solvor.rust._rust_available", False)
        with pytest.raises(ImportError, match="Rust backend explicitly requested"):
            kernel_class("rust")


class TestWarmInfeasibleVerdicts:
    def test_well_scaled_models_trust_the_warm_verdict(self):
        """Well-scaled models keep the warm dual simplex's verdict, so their search does not change."""
        prob = LinearProblem(2, [{0: 1.0, 1: 1.0}], [3.0], [">="], [0.0, 0.0], [2.0, 2.0])
        lp = WarmLP(prob, prob.lb, prob.ub, eps=1e-9, max_iter=100)
        assert lp.solve([1.0, 1.0], minimize=True).status == Status.OPTIMAL
        lp.set_bounds([0.0, 0.0], [1.0, 1.0])
        assert lp.solve([1.0, 1.0], minimize=True).status == Status.INFEASIBLE
        assert (lp.cold_solves, lp.warm_solves) == (1, 1)

    def test_badly_scaled_models_confirm_it_with_a_cold_solve(self):
        """A drifted tableau of a badly scaled model (here 1000:1 in one row) can show a false infeasibility."""
        prob = LinearProblem(2, [{0: 1000.0, 1: 1.0}], [1500.0], [">="], [0.0, 0.0], [2.0, 2.0])
        lp = WarmLP(prob, prob.lb, prob.ub, eps=1e-9, max_iter=100)
        assert lp.solve([1.0, 1.0], minimize=True).status == Status.OPTIMAL
        lp.set_bounds([0.0, 0.0], [1.0, 1.0])
        assert lp.solve([1.0, 1.0], minimize=True).status == Status.INFEASIBLE
        assert lp.cold_solves == 2


class TestScaling:
    def test_well_scaled_rows_and_columns_keep_factor_one(self):
        """Only a spread of 2**8 or more within a row or column is scaled; uniformly large rows are left alone."""
        rows = [{0: 5.0, 1: 40.0, 2: 0.1}, {0: 1.0, 2: 3.3}, {3: 300.0, 4: 900.0}]
        prob = LinearProblem(5, rows, [10.0, 4.0, 1000.0], ["<=", ">=", "<="], [0.0] * 5, [1.0] * 5)
        std = Standardized(prob, prob.lb, prob.ub)
        assert std.scale == [1.0] * 5
        assert std.rows == [{0: 5.0, 1: 40.0, 2: 0.1}, {0: -1.0, 2: -3.3}, {3: 300.0, 4: 900.0}]
        assert not std.badly_scaled

    def test_bound_ranges_alone_do_not_scale(self):
        ub = [0.25, 3000.0, 1e6, math.inf]
        prob = LinearProblem(4, [{0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}], [10.0], ["<="], [0.0] * 4, ub)
        std = Standardized(prob, prob.lb, prob.ub)
        assert std.scale == [1.0] * 4
        assert std.upper == ub

    def test_wide_spread_gets_power_of_two_factors(self):
        prob = LinearProblem(2, [{0: 1e6, 1: 1.0}, {0: 1.0, 1: 1.0}], [2e6, 1.0], ["<=", "<="], [0.0, 0.0], [1.0, 1.0])
        std = Standardized(prob, prob.lb, prob.ub)
        assert std.badly_scaled
        assert std.scale == [2.0**-9, 1.0]
        for row, original in zip(std.rows, prob.rows):
            for j, v in row.items():
                ratio = v / (original[j] * std.scale[j])
                assert math.frexp(ratio)[0] == 0.5  # an exact power of two
        assert std.to_x([2.0**9, 0.0]) == [1.0, 0.0]

    def test_column_factor_keeps_the_scaled_range_between_1_and_2_11(self):
        """A smaller range would read as a fixed column; a factor far below 1 would shrink the cost under eps."""
        rows = [{0: 1e9, 1: 1e-6}, {0: 1.0, 1: 1e-9}]
        prob = LinearProblem(2, rows, [1.0, 1.0], ["<=", "<="], [0.0, 0.0], [1.0, 1.0])
        assert Standardized(prob, prob.lb, prob.ub).scale == [2.0**-10, 1.0]

    def test_small_costs_survive_scaling(self):
        c = [-2.87449063397315, -5.27017580582406e-05, 0.5313982799582897, 0.00036697885825528884]
        c += [-1.6404823292206272, -0.000365963880148854, -0.35002494328736944, -2.195421747396047e-05]
        row = {1: -0.4, 4: -4.0, 5: -0.2, 6: 1.0, 7: -700000.0}
        result = solve_lp(c, [row], [0.5843833895089641], ub=[1] * 8)
        assert abs(result.objective - -4.865438526336828) <= 1e-12

    @pytest.mark.parametrize(("coefficient", "eps"), [(1e-6, 1e-6), (1e-11, 1e-10)])
    def test_columns_with_tiny_coefficients_still_enter(self, coefficient, eps):
        assert solve_lp([-1.0], [[coefficient]], [1.0], ub=[1.0], eps=eps).objective == -1.0

    def test_subnormal_coefficients_do_not_crash(self):
        assert solve_lp([-1, -1], [{0: 1e-310, 1: 1}], [1], ub=[1, 1]).objective == -2.0

    def test_free_and_upper_bounded_columns_map_back_exactly(self):
        """x0 is free (two scaled columns, the negative one in use) and x1 has only an upper bound."""
        from solvor.simplex import solve_lp

        inf = math.inf
        result = solve_lp([1.0, -1.0], [{0: 1e6, 1: 1.0}], [-2e6 + 3], senses=[">="], lb=[-inf, -inf], ub=[inf, 5.0])
        assert result.solution == (-2.000002, 5.0)
        assert result.objective == -7.000002

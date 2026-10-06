r"""
MILP Solver, linear optimization with integer constraints.

Linear programming with integer constraints. The workhorse of discrete
optimization: diet problems, scheduling, set covering, facility location.

    from solvor.milp import solve_milp

    # minimize c @ x, subject to A @ x <= b, x >= 0, some x integer
    result = solve_milp(c, A, b, integers=[0, 2])
    result = solve_milp(c, A, b, integers=[0, 1], minimize=False)  # maximize

    # sparse rows, 0/1 variables, other senses
    result = solve_milp(c, [{0: 1, 1: 1}, {1: 1, 2: 1}], [1, 1], binary=range(3), minimize=False)
    result = solve_milp(c, rows, b, integers=[0], senses=[">=", "<="], lb=[-5, 0], ub=[5, 10])

    # warm start from previous solution (prunes search tree)
    result = solve_milp(c, A, b, integers=[0, 2], warm_start=previous.solution)

    # find multiple solutions (result.solutions contains all found)
    result = solve_milp(c, A, b, integers=[0, 2], solution_limit=5)

    # incremental: add rows (lazy cuts, objective locks) and re-solve warm
    model = MilpModel(n_vars=3, binary=range(3))
    model.add_rows([{0: 1, 1: 1}, {1: 1, 2: 1}], [1, 1])
    first = model.solve([3, 2, 3], minimize=False)
    model.add_rows([{0: 1, 2: 1}], [1])        # a cut
    second = model.solve([3, 2, 3], minimize=False)

    # lexicographic objectives: best for c1, then best for c2 among those, ...
    result = solve_lexicographic(model, [c1, c2, c3], minimize=False)

How it works: presolve turns single-variable rows into bounds and tightens
rows whose variables and coefficients are all integers (divide by the gcd,
round the right-hand side). Branch and bound then solves LP relaxations with
a bounded simplex that is kept alive between nodes: a child differs from its
parent by one bound, so the dual simplex re-solves it in a few pivots. After
branching it dives into one child (the rounding direction) and queues the
other; when a dive ends it continues from the queued node with the best bound.
MilpModel keeps the same LP between solve() calls, so added rows and new
objectives also re-solve warm.

Determinism: results depend only on the inputs and, for MilpModel, on the
sequence of calls. No wall-clock limits, no randomness unless lns_iterations
is set (then pass seed), integer variables are visited in index order.

Use this for:

- Linear objectives with integer constraints
- Diet/blending, scheduling, set covering
- Facility location, power grid design
- When you need proven optimal values

Parameters:

    c: objective coefficients (minimize c @ x)
    A: constraint rows, dense lists or sparse {column: coefficient} dicts
    b: right-hand sides
    integers: indices of integer-constrained variables (default: none)
    binary: indices of 0/1 variables (integer with bounds [0, 1])
    senses: per row "<=", ">=" or "=" (default: all "<=")
    lb, ub: variable bounds (default: 0 and +inf)
    minimize: True for min, False for max (default: True)
    warm_start: initial solution to prune search tree
    solution_limit: find multiple solutions (default: 1)
    heuristics: rounding + local search heuristics (default: True)
    lns_iterations: LNS improvement passes, 0 = off (default: 0)
    lns_destroy_frac: fraction of variables to unfix per LNS iteration (default: 0.3)
    seed: random seed for LNS reproducibility
    max_nodes: branch-and-bound node limit (default: 100000)
    gap_tol: optimality gap tolerance (default: 1e-6)
    backend: "auto", "rust", or "python" (default: "auto")

Backend: the LP relaxations run on an optional Rust simplex kernel (3-100x
faster); branch and bound, presolve and heuristics stay in Python, and results
are the same, bit for bit. Use backend="python" for the pure Python implementation.

CP is more expressive for logical constraints. SAT handles pure boolean.
For continuous-only problems, use simplex directly.
"""

from collections.abc import Mapping, Sequence
from dataclasses import replace
from heapq import heappop, heappush
from math import ceil, floor, inf, isnan
from random import Random
from typing import Literal

from solvor.lp_engine import WarmLP
from solvor.milp_heuristics import is_feasible, lns_improve, round_binary
from solvor.milp_presolve import presolve_row, round_integer_bounds
from solvor.types import Result, Status
from solvor.utils import check_integers_valid
from solvor.utils.lp_input import LinearProblem, normalize_bounds, normalize_lp, normalize_rows
from solvor.utils.validate import check_non_negative

__all__ = ["MilpModel", "solve_lexicographic", "solve_milp"]

# Presolve runs as rows arrive, before solve() knows its eps; same value as solve_milp's default eps
PRESOLVE_TOL = 1e-6
# LP relaxations get their own, tighter tolerance; eps is the integrality and feasibility tolerance
LP_EPS = 1e-9


class MilpModel:
    """A MILP that grows by rows and is re-solved warm.

    Variables, their integrality and bounds are fixed at construction (bounds
    can only tighten, through single-variable rows). Rows are added with
    add_rows and pass through the same presolve as solve_milp. Each solve()
    takes its own objective. The LP tableau, and the last solution as a
    starting incumbent, carry over from one solve to the next. backend picks
    the LP kernel, as in solve_milp; results are identical.
    """

    def __init__(
        self,
        n_vars: int,
        *,
        integers: Sequence[int] = (),
        binary: Sequence[int] | None = None,
        lb: Sequence[float] | None = None,
        ub: Sequence[float] | None = None,
        backend: Literal["auto", "rust", "python"] | None = None,
    ):
        self.n = n_vars
        self._backend = backend
        integers = list(integers)
        check_integers_valid(integers, n_vars)
        self._int_set = set(integers)
        self._lower, self._upper = normalize_bounds(n_vars, lb, ub)
        if binary is not None:
            binary = list(binary)
            check_integers_valid(binary, n_vars, name="binary")
            for j in binary:
                self._int_set.add(j)
                self._lower[j] = max(self._lower[j], 0.0)
                self._upper[j] = min(self._upper[j], 1.0)
        self._infeasible = not round_integer_bounds(
            sorted(self._int_set), self._lower, self._upper, PRESOLVE_TOL
        ) or any(self._lower[j] > self._upper[j] + PRESOLVE_TOL for j in range(n_vars))
        self._rows: list[dict[int, float]] = []
        self._b: list[float] = []
        self._senses: list[str] = []
        self._lp: WarmLP | None = None
        self._lp_max_iter: int | None = None
        self._last: tuple[float, ...] | None = None

    @property
    def n_rows(self) -> int:
        """Rows kept after presolve (single-variable rows become bounds and are not counted)."""
        return len(self._rows)

    def add_rows(
        self,
        A: Sequence[Sequence[float] | Mapping[int, float]],
        b: Sequence[float],
        senses: Sequence[str] | None = None,
    ) -> None:
        """Add constraint rows (dense or sparse, as in solve_milp)."""
        rows, rhs, sense_list = normalize_rows(self.n, A, b, senses, stacklevel=3)
        self._add_normalized(rows, rhs, sense_list)

    def _add_normalized(self, rows: list[dict[int, float]], rhs: list[float], senses: list[str]) -> None:
        for row, bi, sense in zip(rows, rhs, senses):
            feasible, kept = presolve_row(row, bi, sense, self._int_set, self._lower, self._upper, PRESOLVE_TOL)
            if not feasible:
                self._infeasible = True
            if kept is None:
                continue
            coefs, r = kept
            self._rows.append(coefs)
            self._b.append(r)
            self._senses.append(sense)
            if self._lp is not None:
                self._lp.add_row(coefs, r, sense)

    def solve(
        self,
        c: Sequence[float],
        *,
        minimize: bool = True,
        eps: float = 1e-6,
        max_iter: int = 10_000,
        max_nodes: int = 100_000,
        gap_tol: float = 1e-6,
        warm_start: Sequence[float] | None = None,
        solution_limit: int = 1,
        heuristics: bool = True,
        lns_iterations: int = 0,
        lns_destroy_frac: float = 0.3,
        seed: int | None = None,
    ) -> Result:
        """Optimize c over the current rows and bounds. Same keywords as solve_milp."""
        n = self.n
        if len(c) != n:
            raise ValueError(f"Length mismatch: expected {n} elements in c, got {len(c)}")
        check_non_negative(eps, name="eps")
        cost = [float(v) for v in c]
        if any(isnan(v) for v in cost):
            raise ValueError("c contains NaN")
        no_solution = inf if minimize else -inf
        if self._infeasible:
            return Result(None, no_solution, 0, 0, Status.INFEASIBLE)

        prob = LinearProblem(n, self._rows, self._b, self._senses, self._lower, self._upper)
        int_set = self._int_set
        int_list = sorted(int_set)
        lower, upper = tuple(self._lower), tuple(self._upper)
        lp = self._warm_lp(prob, max_iter)
        lp.set_bounds(lower, upper)

        root = lp.solve(cost, minimize=minimize)
        total_iters = root.iterations

        if root.status == Status.INFEASIBLE:
            return Result(None, no_solution, 0, total_iters, Status.INFEASIBLE)

        if root.status == Status.UNBOUNDED:
            return Result(None, -no_solution, 0, total_iters, Status.UNBOUNDED)

        if root.status != Status.OPTIMAL:  # iteration limit: no trustworthy LP bound or point
            return Result(None, no_solution, 0, total_iters, root.status)

        best_solution, best_obj = None, no_solution
        sign = 1 if minimize else -1
        all_solutions: list[tuple[float, ...]] = []

        # Incumbent from warm_start, else from the previous solve, if feasible now
        for candidate in (warm_start, self._last):
            if best_solution is None and candidate is not None:
                ws = tuple(float(v) for v in candidate)
                if len(ws) == n and is_feasible(ws, prob, lower, upper, int_set, eps):
                    best_solution = _accepted(ws, int_list, prob, eps)
                    if best_solution is not None:
                        best_obj = sum(cost[j] * best_solution[j] for j in range(n))
                        all_solutions.append(best_solution)

        root_point = _clamp(root.solution, int_list, lower, upper)
        root_inaccurate = _outside_bounds(root.solution, int_list, lower, upper, eps)
        if not root_inaccurate and _most_fractional(root_point, int_list, eps) is None:
            sol = _accepted(root_point, int_list, prob, eps)
            if sol is not None:
                self._last = sol
                return Result(sol, sum(cost[j] * sol[j] for j in range(n)), 1, total_iters)

        # Rounding heuristics flip integer variables between 0 and 1
        looks_binary = all(-eps <= root.solution[j] <= 1 + eps for j in int_list)

        if heuristics and looks_binary and best_solution is None:
            cols = prob.columns()
            rounded = round_binary(root.solution, int_list, cost, prob, cols, lower, upper, minimize, eps)
            if rounded is not None:
                best_solution = _accepted(rounded, int_list, prob, eps)
                if best_solution is not None:
                    best_obj = sum(cost[j] * best_solution[j] for j in range(n))
                    all_solutions.append(best_solution)

        # LNS improvement for binary problems
        if heuristics and looks_binary and lns_iterations > 0 and best_solution is not None:
            rng = Random(seed)
            improved, iters = lns_improve(
                best_solution,
                cost,
                prob,
                lower,
                upper,
                int_set,
                minimize,
                eps,
                max_iter,
                lns_iterations,
                lns_destroy_frac,
                rng,
                self._backend,
            )
            total_iters += iters
            if improved is not None:
                improved = _accepted(improved, int_list, prob, eps)
            if improved is not None:
                improved_obj = sum(cost[j] * improved[j] for j in range(n))
                if (minimize and improved_obj < best_obj) or (not minimize and improved_obj > best_obj):
                    best_solution, best_obj = improved, improved_obj
                    if improved not in all_solutions:
                        all_solutions.append(improved)

        # Branch and bound. Heap entries: (bound, counter, lower, upper); next_node is the dive child.
        tree: list[tuple[float, int, tuple[float, ...], tuple[float, ...]]] = []
        counter = 0
        nodes_explored = 0
        unreliable = 0  # nodes whose LP point was visibly inaccurate: nothing that depends on them is proven
        next_node: tuple[float, tuple[float, ...], tuple[float, ...], Result | None] | None = (
            sign * root.objective,
            lower,
            upper,
            root,
        )

        while True:
            if next_node is None:
                while tree:
                    bound, _, nl, nu = heappop(tree)
                    if best_solution is None or bound < sign * best_obj - eps:
                        next_node = (bound, nl, nu, None)
                        break
                if next_node is None:
                    break
            if nodes_explored >= max_nodes:
                heappush(tree, (next_node[0], counter, next_node[1], next_node[2]))
                next_node = None
                break

            node_bound, nl, nu, res = next_node
            next_node = None
            if res is None:
                lp.set_bounds(nl, nu)
                res = lp.solve(cost, minimize=minimize)
                total_iters += res.iterations
            nodes_explored += 1

            if res.status != Status.OPTIMAL:
                continue
            if best_solution is not None and sign * res.objective >= sign * best_obj - eps:
                continue

            if _outside_bounds(res.solution, int_list, nl, nu, eps):
                unreliable += 1
            # Clamped: an LP value just past its bound would otherwise branch into the same node again
            point = _clamp(res.solution, int_list, nl, nu)
            frac_var = _most_fractional(point, int_list, eps)
            sol = None
            if frac_var is None:
                sol = _accepted(point, int_list, prob, eps)
                if sol is None:  # integral within eps, but snapping breaks a row: branch on the largest fraction
                    frac_var = _largest_fraction(point, int_list)
                    if frac_var is None:  # exactly integral and still violating a row: drop the node
                        unreliable += 1
                        continue

            if sol is not None:
                # Found an integer-feasible solution
                sol_obj = sum(cost[j] * sol[j] for j in range(n))

                if solution_limit > 1 and sol not in all_solutions:
                    all_solutions.append(sol)
                    if len(all_solutions) >= solution_limit:
                        self._last = best_solution or sol
                        objective = best_obj if best_solution else sol_obj
                        solutions = tuple(all_solutions)
                        return _final(
                            best_solution or sol,
                            objective,
                            nodes_explored,
                            total_iters,
                            Status.FEASIBLE,
                            unreliable,
                            solutions,
                        )

                if sign * sol_obj < sign * best_obj:
                    best_solution, best_obj = sol, sol_obj
                    global_bound = min(tree[0][0], node_bound) if tree else node_bound
                    gap = _compute_gap(best_obj, global_bound / sign if global_bound != 0 else 0)
                    if gap < gap_tol and solution_limit == 1:
                        self._last = best_solution
                        return _final(best_solution, best_obj, nodes_explored, total_iters, Status.OPTIMAL, unreliable)
                continue

            # Branch on the fractional variable: dive into the rounding direction, queue the other child
            val = point[frac_var]
            child_bound = sign * res.objective
            down_upper = list(nu)
            down_upper[frac_var] = floor(val)
            up_lower = list(nl)
            up_lower[frac_var] = ceil(val)
            down = (nl, tuple(down_upper))
            up = (tuple(up_lower), nu)
            dive, other = (up, down) if val - floor(val) >= 0.5 else (down, up)
            heappush(tree, (child_bound, counter, other[0], other[1]))
            counter += 1
            next_node = (child_bound, dive[0], dive[1], None)

        if best_solution is None:
            return _final(None, no_solution, nodes_explored, total_iters, Status.INFEASIBLE, unreliable)

        self._last = best_solution
        status = Status.OPTIMAL if not tree else Status.FEASIBLE
        solutions = tuple(all_solutions) if solution_limit > 1 and all_solutions else None
        return _final(best_solution, best_obj, nodes_explored, total_iters, status, unreliable, solutions)

    def _warm_lp(self, prob: LinearProblem, max_iter: int) -> WarmLP:
        if self._lp is None or self._lp_max_iter != max_iter:
            self._lp = WarmLP(prob, prob.lb, prob.ub, eps=LP_EPS, max_iter=max_iter, backend=self._backend)
            self._lp_max_iter = max_iter
        return self._lp


def solve_milp(
    c: Sequence[float],
    A: Sequence[Sequence[float] | Mapping[int, float]],
    b: Sequence[float],
    integers: Sequence[int] = (),
    *,
    binary: Sequence[int] | None = None,
    senses: Sequence[str] | None = None,
    lb: Sequence[float] | None = None,
    ub: Sequence[float] | None = None,
    minimize: bool = True,
    eps: float = 1e-6,
    max_iter: int = 10_000,
    max_nodes: int = 100_000,
    gap_tol: float = 1e-6,
    warm_start: Sequence[float] | None = None,
    solution_limit: int = 1,
    heuristics: bool = True,
    lns_iterations: int = 0,
    lns_destroy_frac: float = 0.3,
    seed: int | None = None,
    backend: Literal["auto", "rust", "python"] | None = None,
) -> Result:
    prob = normalize_lp(c, A, b, lb=lb, ub=ub, senses=senses)
    model = MilpModel(prob.n, integers=integers, binary=binary, lb=prob.lb, ub=prob.ub, backend=backend)
    model._add_normalized(prob.rows, prob.b, prob.senses)
    return model.solve(
        c,
        minimize=minimize,
        eps=eps,
        max_iter=max_iter,
        max_nodes=max_nodes,
        gap_tol=gap_tol,
        warm_start=warm_start,
        solution_limit=solution_limit,
        heuristics=heuristics,
        lns_iterations=lns_iterations,
        lns_destroy_frac=lns_destroy_frac,
        seed=seed,
    )


def solve_lexicographic(
    model: MilpModel,
    objectives: Sequence[Sequence[float]],
    *,
    minimize: bool = True,
    tol: float = 0.0,
    **solve_kwargs,
) -> Result:
    """Optimize objectives in priority order on `model`.

    After each objective but the last, a row keeps its value: c . x >= best - tol
    when maximizing, c . x <= best + tol when minimizing. Rows stay in the model.
    Returns the last stage's result, or the first stage that is not
    OPTIMAL or FEASIBLE. The result is OPTIMAL only if every stage was: a stage
    that stopped early (FEASIBLE) may have fixed a worse value for the later ones.
    """
    if len(objectives) == 0:  # not `not objectives`: numpy arrays have no truth value
        raise ValueError("solve_lexicographic needs at least one objective")
    proven = True
    for k, c in enumerate(objectives):
        result = model.solve(c, minimize=minimize, **solve_kwargs)
        if result.status not in (Status.OPTIMAL, Status.FEASIBLE):
            return result
        proven = proven and result.status == Status.OPTIMAL
        if k < len(objectives) - 1:
            row = {j: float(v) for j, v in enumerate(c) if v != 0}
            if minimize:
                model.add_rows([row], [result.objective + tol], ["<="])
            else:
                model.add_rows([row], [result.objective - tol], [">="])
    if not proven and result.status == Status.OPTIMAL:
        return replace(result, status=Status.FEASIBLE)
    return result


# Every row holds up to eps * max(1, |activity|, |rhs|) (SCIP's rule)
def _rows_hold(x, prob, eps):
    for row, bi, sense in zip(prob.rows, prob.b, prob.senses):
        activity = sum(a * x[j] for j, a in row.items())
        tol = eps * max(1.0, abs(activity), abs(bi))
        if (sense != ">=" and activity > bi + tol) or (sense != "<=" and activity < bi - tol):
            return False
    return True


# Snapped point, or None if a row fails: a 1e-6 snap through a 1e6 coefficient moves the row by 1
def _accepted(x, int_list, prob, eps):
    sol = _snap(x, int_list)
    return sol if _rows_hold(sol, prob, eps) else None


def _clamp(solution, int_list, lower, upper):
    point = list(solution)
    for j in int_list:
        point[j] = min(max(point[j], lower[j]), upper[j])
    return point


def _outside_bounds(solution, int_list, lower, upper, eps):
    for j in int_list:
        if solution[j] < lower[j] - eps * max(1.0, abs(lower[j])):
            return True
        if solution[j] > upper[j] + eps * max(1.0, abs(upper[j])):
            return True
    return False


def _largest_fraction(solution, int_list):
    best_var, best_frac = None, 0.0
    for j in int_list:
        frac = abs(solution[j] - round(solution[j]))
        if frac > best_frac:
            best_var, best_frac = j, frac
    return best_var


# An unreliable node may hide better or feasible points, so nothing is proven: no OPTIMAL, and the error says why
def _final(solution, objective, nodes, iters, status, unreliable, solutions=None):
    if not unreliable:
        return Result(solution, objective, nodes, iters, status, solutions=solutions)
    error = (
        f"{unreliable} branch-and-bound node(s) had an inaccurate LP solution (outside its bounds, or integral "
        "but violating a row): numerical trouble, often from coefficients of very different magnitude"
    )
    if status == Status.OPTIMAL:
        status = Status.FEASIBLE
    return Result(solution, objective, nodes, iters, status, error=error, solutions=solutions)


def _snap(solution: Sequence[float], int_list: list[int]) -> tuple[float, ...]:
    """Integer variables exactly integral: removes pivot-path noise like 0.9999999999999998 from results."""
    out = list(solution)
    for j in int_list:
        out[j] = float(round(out[j]))
    return tuple(out)


def _most_fractional(solution, int_list, eps):
    best_var, best_frac = None, 0.0
    for j in int_list:
        val = solution[j]
        frac = abs(val - round(val))
        if frac > eps and frac > best_frac:
            best_var, best_frac = j, frac
    return best_var


def _compute_gap(best_obj, bound):
    if abs(best_obj) < 1e-10:
        return abs(best_obj - bound)
    return abs(best_obj - bound) / abs(best_obj)

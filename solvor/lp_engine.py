"""
Dense-tableau LP engine shared by solve_lp, solve_milp and MilpModel.

Internal module. Three layers:

    Standardized   maps x (lb <= x <= ub, rows with senses) onto columns
                   y >= lo with optional upper bounds and rows "<=" or "="
    BoundedSimplex the tableau: primal simplex (phase 1 and 2), dual simplex,
                   in-place bound changes and row additions
    WarmLP         keeps one BoundedSimplex alive across solves whose bounds,
                   rows or objective change, and falls back to a cold build
                   whenever a warm start cannot be trusted

Every nonbasic column sits at one of its bounds. In the tableau a column is
written as t = y - lo (at its lower bound) or t = hi - y (at its upper bound,
"flipped"), so every nonbasic t is 0 and the right-hand side holds the basic
t values. Bland's rule (smallest index on ties) keeps both simplex variants
deterministic and cycle-free.
"""

from array import array
from collections.abc import Sequence
from math import inf

from solvor.types import Result, Status
from solvor.utils.lp_input import LinearProblem

__all__ = ["BoundedSimplex", "Standardized", "WarmLP", "solve_cold"]


class Standardized:
    """Maps x with lb <= x <= ub onto columns y and rows onto '<=' or '='.

    x = lb + y when lb is finite, x = ub - y when only ub is finite, and
    x = y_plus - y_minus when x is free. '>=' rows are negated. The mapping is
    fixed when it is built; later bound changes become bounds on y (see y_bounds).
    """

    def __init__(self, prob: LinearProblem, lb: Sequence[float], ub: Sequence[float]):
        n = prob.n
        self.n = n
        self.first = [0] * n
        self.second = [-1] * n
        self.offset = [0.0] * n
        self.sign = [1.0] * n
        self.upper: list[float] = []
        for j in range(n):
            lo, hi = lb[j], ub[j]
            self.first[j] = len(self.upper)
            if lo > -inf:
                self.offset[j] = lo
                self.upper.append(hi - lo)
            elif hi < inf:
                self.offset[j], self.sign[j] = hi, -1.0
                self.upper.append(inf)
            else:
                self.second[j] = len(self.upper) + 1
                self.upper.extend((inf, inf))
        self.n_cols = len(self.upper)

        self.rows: list[dict[int, float]] = []
        self.rhs: list[float] = []
        self.is_eq: list[bool] = []
        self.impossible = False  # a row with an infinite rhs that can never hold
        for row, bi, sense in zip(prob.rows, prob.b, prob.senses):
            coefs, r, eq = self.map_row(row, bi, sense)
            if r == inf and not eq:
                continue  # can never bind
            if r == -inf or (eq and r == inf):
                self.impossible = True
                continue
            self.rows.append(coefs)
            self.rhs.append(r)
            self.is_eq.append(eq)

    def map_row(self, row: dict[int, float], rhs: float, sense: str) -> tuple[dict[int, float], float, bool]:
        new: dict[int, float] = {}
        r = rhs
        for j, a in row.items():
            r -= a * self.offset[j]
            p = self.first[j]
            new[p] = new.get(p, 0.0) + a * self.sign[j]
            q = self.second[j]
            if q >= 0:
                new[q] = new.get(q, 0.0) - a
        if sense == ">=":
            return {k: -v for k, v in new.items()}, -r, False
        return new, r, sense == "="

    def map_cost(self, cost: Sequence[float]) -> list[float]:
        out = [0.0] * self.n_cols
        for j in range(self.n):
            out[self.first[j]] += cost[j] * self.sign[j]
            if self.second[j] >= 0:
                out[self.second[j]] -= cost[j]
        return out

    def y_bounds(self, j: int, lb: float, ub: float) -> tuple[float, float] | None:
        """Bounds on x_j's column, or None if this mapping cannot express them (rebuild needed)."""
        if self.second[j] >= 0:
            return None
        if self.sign[j] > 0:
            lo, hi = lb - self.offset[j], ub - self.offset[j]
        else:
            lo, hi = self.offset[j] - ub, self.offset[j] - lb
        if lo == -inf:
            return None
        return lo, hi

    def to_x(self, y: Sequence[float]) -> list[float]:
        x = []
        for j in range(self.n):
            v = self.offset[j] + self.sign[j] * y[self.first[j]]
            if self.second[j] >= 0:
                v -= y[self.second[j]]
            x.append(v)
        return x


class BoundedSimplex:
    """Dense tableau for: rows . y + s = rhs, lo <= y <= hi, slack s >= 0 (s = 0 on '=' rows).

    Columns are [structural | slack | artificial | slacks of added rows], the
    right-hand side sits at index `width`, and the objective row is T[m].
    """

    def __init__(
        self, rows: list[dict[int, float]], rhs: list[float], is_eq: list[bool], upper: list[float], eps: float
    ):
        n, m = len(upper), len(rows)
        needs_art = [r < -eps or (eq and r > eps) for r, eq in zip(rhs, is_eq)]
        width = n + m + sum(needs_art)
        self.n, self.m, self.width, self.eps = n, m, width, eps
        self.art_start, self.art_end = n + m, width
        self.pivots = 0
        # Phase 1 is feasible when every artificial is zero up to a tolerance taken from its own row
        self.art_tol = [1e-9 * (1.0 + abs(r)) for r, needs in zip(rhs, needs_art) if needs]
        self.phase1_done = width == n + m

        self.lo = array("d", [0.0]) * width
        self.hi = array("d", upper)
        self.hi.extend(0.0 if eq else inf for eq in is_eq)
        self.hi.extend([inf] * (width - n - m))
        self.span = array("d", self.hi)
        self.flipped = bytearray(width)
        self.is_basic = bytearray(width)
        self.basis = array("i", [0] * m)
        self.row_of = array("i", [-1]) * width

        zeros = array("d", [0.0]) * (width + 1)
        self.T: list[array] = []
        art = self.art_start
        for i in range(m):
            row = array("d", zeros)
            for j, a in rows[i].items():
                row[j] = a
            row[n + i] = 1.0
            row[width] = rhs[i]
            if needs_art[i]:
                if rhs[i] < 0:
                    for j in rows[i]:
                        row[j] = -row[j]
                    row[n + i] = -1.0
                    row[width] = -rhs[i]
                row[art] = 1.0
                basic = art
                art += 1
            else:
                basic = n + i
            self._set_basic(i, basic)
            self.T.append(row)
        self.T.append(array("d", zeros))

    # ----- cold solve -------------------------------------------------------

    def solve(self, cost: Sequence[float], max_iter: int) -> tuple[Status, int]:
        """Phase 1 (only if artificials exist), then phase 2 on `cost` (one entry per structural column)."""
        iters = 0
        if self.art_end > self.art_start:
            phase1 = [0.0] * self.width
            for a in range(self.art_start, self.art_end):
                phase1[a] = 1.0
            self.set_objective(phase1)
            status, iters = self.primal(max_iter)
            if status == Status.MAX_ITER:
                return status, iters
            w = self.width
            for i in range(self.m):
                b = self.basis[i]
                if self.art_start <= b < self.art_end and self.T[i][w] > self.art_tol[b - self.art_start]:
                    return Status.INFEASIBLE, iters
            self._drive_out_artificials()
            self.phase1_done = True

        self.set_objective(cost)
        status, more = self.primal(max_iter - iters)
        return status, iters + more

    def column_values(self) -> list[float]:
        """Value of every column in y (not in the flipped representation)."""
        w = self.width
        t = [0.0] * w
        for i in range(self.m):
            t[self.basis[i]] = self.T[i][w]
        return [self.hi[j] - t[j] if self.flipped[j] else self.lo[j] + t[j] for j in range(w)]

    def set_objective(self, cost: Sequence[float]) -> None:
        """Objective row = reduced costs of `cost` for the current basis (missing entries are 0)."""
        T, m, w = self.T, self.m, self.width
        obj = array("d", [0.0]) * (w + 1)
        for j in range(min(len(cost), w)):
            cj = cost[j]
            if cj != 0.0:
                obj[j] = -cj if self.flipped[j] else cj
        for i in range(m):
            cb = obj[self.basis[i]]
            if cb != 0.0:
                row = T[i]
                for j in range(w + 1):
                    a = row[j]
                    if a != 0.0:
                        obj[j] -= cb * a
        T[m] = obj

    def primal(self, max_iter: int) -> tuple[Status, int]:
        """Primal simplex from a primal feasible basis."""
        T, m, w, eps = self.T, self.m, self.width, self.eps
        span, basis, is_basic = self.span, self.basis, self.is_basic
        for iteration in range(max(max_iter, 0)):
            obj = T[m]
            # Bland's rule for entering: smallest index with negative reduced cost (fixed columns never enter)
            enter = -1
            for j in range(w):
                if obj[j] < -eps and not is_basic[j] and span[j] > eps:
                    enter = j
                    break
            if enter < 0:
                return Status.OPTIMAL, iteration

            # Bounded ratio test: a basic column hits 0 or its span, or the entering column
            # reaches its own other bound first (bound flip). Ties go to the smallest basis index.
            best, leave, to_upper = span[enter], -1, False
            for i in range(m):
                a = T[i][enter]
                if a > eps:
                    t, up = T[i][w] / a, False
                else:
                    span_basic = span[basis[i]]
                    if a < -eps and span_basic < inf:
                        t, up = (span_basic - T[i][w]) / -a, True
                    else:
                        continue
                if t < 0.0:
                    t = 0.0
                if t < best - eps or (leave >= 0 and abs(t - best) <= eps and basis[i] < basis[leave]):
                    best, leave, to_upper = t, i, up

            if leave < 0:
                if best == inf:
                    return Status.UNBOUNDED, iteration
                self._complement(enter)  # bound flip, no basis change
                continue

            left = basis[leave]
            self._pivot(leave, enter)
            if to_upper:
                self._complement(left)
        return Status.MAX_ITER, max(max_iter, 0)

    # ----- warm operations --------------------------------------------------

    def make_dual_feasible(self) -> bool:
        """Move boxed nonbasic columns to the bound their reduced cost prefers.

        Returns False if a column without a finite span has the wrong sign
        (dual simplex cannot start; the caller rebuilds instead).
        """
        obj = self.T[self.m]
        for j in range(self.width):
            if self.is_basic[j] or self.span[j] <= self.eps or obj[j] >= -self.eps:
                continue
            if self.span[j] == inf:
                return False
            self._complement(j)
        return True

    def dual(self, max_iter: int) -> tuple[Status, int]:
        """Dual simplex: restore primal feasibility while keeping the objective row dual feasible."""
        T, eps, w = self.T, self.eps, self.width
        for iteration in range(max(max_iter, 0)):
            m = self.m
            # Leaving row: the most violated basic column (ties: smallest row)
            leave, worst, above = -1, eps, False
            for i in range(m):
                v = T[i][w]
                if -v > worst:
                    leave, worst, above = i, -v, False
                else:
                    s = self.span[self.basis[i]]
                    if s < inf and v - s > worst:
                        leave, worst, above = i, v - s, True
            if leave < 0:
                return Status.OPTIMAL, iteration
            if above:
                self._flip_basic(leave)  # now below its (other) bound: t' = span - t < 0

            # Entering column: smallest ratio obj[j] / -T[r][j] over columns that raise the row
            row, obj = T[leave], T[self.m]
            enter, best = -1, inf
            for j in range(w):
                a = row[j]
                if a < -eps and not self.is_basic[j] and self.span[j] > eps:
                    ratio = max(obj[j], 0.0) / -a
                    if ratio < best - eps:
                        enter, best = j, ratio
            if enter < 0:
                return Status.INFEASIBLE, iteration
            self._pivot(leave, enter)
        return Status.MAX_ITER, max(max_iter, 0)

    def set_bounds(self, j: int, lo: float, hi: float) -> None:
        """Change column j's bounds in place (lo must be finite). The basis may become primal infeasible."""
        old_lo, old_hi, w = self.lo[j], self.hi[j], self.width
        if self.is_basic[j]:
            r = self.row_of[j]
            if not self.flipped[j]:  # t = y - lo
                self.T[r][w] -= lo - old_lo
            elif hi < inf:  # t = hi - y
                self.T[r][w] += hi - old_hi
            else:  # t = old_hi - y can no longer be used: switch to t' = y - lo = (old_hi - lo) - t
                self._negate_row_except(r, j)
                self.T[r][w] = (old_hi - lo) - self.T[r][w]
                self.flipped[j] = 0
        elif not self.flipped[j]:  # sits at lo
            if lo != old_lo:
                self._shift(j, lo - old_lo)
        elif hi < inf:  # sits at hi, t = hi - y
            if hi != old_hi:
                self._shift(j, old_hi - hi)
        else:  # sits at hi, which disappears: move to lo
            self._shift(j, old_hi - lo)
            self._negate_column(j)
            self.flipped[j] = 0
        self.lo[j], self.hi[j], self.span[j] = lo, hi, hi - lo

    def add_row(self, coefs: dict[int, float], rhs: float, is_eq: bool) -> None:
        """Append `coefs . y <= rhs` (or `=`) with a new basic slack, expressed in the current basis."""
        w = self.width
        for row in self.T:
            row.insert(w, 0.0)  # new slack column, just before the rhs
        new = array("d", [0.0]) * (w + 2)
        r = rhs
        for j, a in coefs.items():
            if self.flipped[j]:
                new[j] -= a
                r -= a * self.hi[j]
            else:
                new[j] += a
                r -= a * self.lo[j]
        new[w + 1] = r
        for i in range(self.m):  # eliminate basic columns
            e = new[self.basis[i]]
            if e != 0.0:
                src = self.T[i]
                for k in range(w + 2):
                    s = src[k]
                    if s != 0.0:
                        new[k] -= e * s
        new[w] = 1.0
        self.T.insert(self.m, new)

        self.lo.append(0.0)
        self.hi.append(0.0 if is_eq else inf)
        self.span.append(0.0 if is_eq else inf)
        self.flipped.append(0)
        self.is_basic.append(0)
        self.row_of.append(-1)
        self.basis.append(0)
        self.width = w + 1
        self.m += 1
        self._set_basic(self.m - 1, w)

    # ----- tableau primitives -----------------------------------------------

    def _set_basic(self, r: int, q: int) -> None:
        self.basis[r] = q
        self.is_basic[q] = 1
        self.row_of[q] = r

    def _pivot(self, r: int, q: int) -> None:
        T, m = self.T, self.m
        prow = T[r]
        inv = 1.0 / prow[q]
        nz = [j for j, a in enumerate(prow) if a != 0.0]  # only the pivot row's nonzeros change other rows
        for j in nz:
            prow[j] *= inv
        for i in range(m + 1):
            if i == r:
                continue
            row = T[i]
            f = row[q]
            if f != 0.0:
                for j in nz:
                    row[j] -= f * prow[j]
                row[q] = 0.0
        old = self.basis[r]
        self.is_basic[old] = 0
        self.row_of[old] = -1
        self._set_basic(r, q)
        self.pivots += 1

    def _complement(self, j: int) -> None:
        """Move nonbasic column j to its other bound (t' = span - t), which sits at 0."""
        u, w = self.span[j], self.width
        for row in self.T:
            a = row[j]
            if a != 0.0:
                row[w] -= a * u
                row[j] = -a
        self.flipped[j] ^= 1

    def _flip_basic(self, r: int) -> None:
        """Rewrite row r for its basic column measured from the other bound: t' = span - t."""
        b = self.basis[r]
        self._negate_row_except(r, b)
        self.T[r][self.width] = self.span[b] - self.T[r][self.width]
        self.flipped[b] ^= 1

    def _negate_row_except(self, r: int, keep: int) -> None:
        row, w = self.T[r], self.width
        for k in range(w):
            if k != keep and row[k] != 0.0:
                row[k] = -row[k]

    def _shift(self, j: int, dt: float) -> None:
        """Nonbasic column j's t changes by dt: update every row's right-hand side."""
        w = self.width
        for row in self.T:
            a = row[j]
            if a != 0.0:
                row[w] -= a * dt

    def _negate_column(self, j: int) -> None:
        for row in self.T:
            if row[j] != 0.0:
                row[j] = -row[j]

    def _drive_out_artificials(self) -> None:
        """Pivot basic artificials (all at 0 now) out where possible, then pin every artificial to 0."""
        for i in range(self.m):
            if not self.art_start <= self.basis[i] < self.art_end:
                continue
            row = self.T[i]
            for j in range(self.art_start):
                if not self.is_basic[j] and abs(row[j]) > self.eps:
                    self._pivot(i, j)
                    break
        for a in range(self.art_start, self.art_end):
            self.hi[a] = 0.0
            self.span[a] = 0.0


def solve_cold(
    prob: LinearProblem,
    c: Sequence[float],
    lb: Sequence[float],
    ub: Sequence[float],
    *,
    minimize: bool,
    eps: float,
    max_iter: int,
) -> Result:
    """One-shot solve of a normalized LP with the given bounds (no state kept)."""
    lp = WarmLP(prob, lb, ub, eps=eps, max_iter=max_iter)
    return lp.solve(c, minimize=minimize)


class WarmLP:
    """An LP whose rows, variable bounds and objective may change between solves.

    Re-solves start from the previous tableau: dual simplex to restore primal
    feasibility after bound changes or added rows, then primal simplex to restore
    optimality (after an objective change, or when a dual pivot on a tiny element
    left a reduced cost slightly wrong). It rebuilds from scratch when a warm start
    is not possible or not trustworthy: a column mapping that cannot express the
    new bounds, a dual-infeasible unbounded column, an iteration limit, a warm
    UNBOUNDED, too many pivots since the last build (numerical drift), or a
    solution that fails the final feasibility check.
    """

    REBUILD_PIVOTS = 2000

    def __init__(self, prob: LinearProblem, lb: Sequence[float], ub: Sequence[float], *, eps: float, max_iter: int):
        self.n = prob.n
        self.rows = list(prob.rows)
        self.b = list(prob.b)
        self.senses = list(prob.senses)
        self.lb, self.ub = list(lb), list(ub)
        self.eps, self.max_iter = eps, max_iter
        self.std: Standardized | None = None
        self.lp: BoundedSimplex | None = None
        self.cost_y: list[float] | None = None
        self.cold_solves = 0
        self.warm_solves = 0

    def add_row(self, row: dict[int, float], rhs: float, sense: str) -> None:
        self.rows.append(row)
        self.b.append(rhs)
        self.senses.append(sense)
        if self.lp is not None and self.std is not None:
            coefs, r, eq = self.std.map_row(row, rhs, sense)
            if r == inf and not eq:
                return  # can never bind
            if r == -inf or (eq and r == inf):
                self.lp = None  # can never hold: the cold rebuild reports INFEASIBLE
                return
            self.lp.add_row(coefs, r, eq)

    def set_bounds(self, lb: Sequence[float], ub: Sequence[float]) -> None:
        for j in range(self.n):
            if lb[j] != self.lb[j] or ub[j] != self.ub[j]:
                self.lb[j], self.ub[j] = lb[j], ub[j]
                if self.lp is not None and self.std is not None:
                    y = self.std.y_bounds(j, lb[j], ub[j])
                    if y is None:
                        self.lp = None  # mapping cannot express it: rebuild on the next solve
                    else:
                        self.lp.set_bounds(self.std.first[j], y[0], y[1])

    def solve(self, c: Sequence[float], *, minimize: bool) -> Result:
        n = self.n
        if any(self.lb[j] > self.ub[j] + self.eps for j in range(n)):
            return Result(tuple([0.0] * n), inf if minimize else -inf, 0, 0, Status.INFEASIBLE)

        cost = list(c) if minimize else [-v for v in c]
        result = None
        if self.lp is not None and self.lp.pivots < self.REBUILD_PIVOTS:
            result = self._solve_warm(cost, c, minimize)
        if result is None:
            result = self._solve_cold(cost, c, minimize)
        return result

    def _prob(self) -> LinearProblem:
        return LinearProblem(self.n, self.rows, self.b, self.senses, self.lb, self.ub)

    def _solve_cold(self, cost: list[float], c: Sequence[float], minimize: bool) -> Result:
        self.cold_solves += 1
        self.std = Standardized(self._prob(), self.lb, self.ub)
        if self.std.impossible:
            self.lp = None
            return self._result(Status.INFEASIBLE, 0, c, minimize)
        self.lp = BoundedSimplex(self.std.rows, self.std.rhs, self.std.is_eq, self.std.upper, self.eps)
        self.cost_y = self.std.map_cost(cost)
        status, iters = self.lp.solve(self.cost_y, self.max_iter)
        if status != Status.OPTIMAL:
            result = self._result(status, iters, c, minimize)
            self.lp = None  # never warm-start from an unfinished tableau
            return result
        result = self._result(status, iters, c, minimize)
        self.lp.pivots = 0  # drift is counted from this build on
        return result

    def _solve_warm(self, cost: list[float], c: Sequence[float], minimize: bool) -> Result | None:
        lp, std = self.lp, self.std
        assert lp is not None and std is not None
        cost_y = std.map_cost(cost)
        iters = 0
        if not lp.make_dual_feasible():
            return None
        status, it = lp.dual(self.max_iter)
        iters += it
        if status == Status.MAX_ITER:
            return None
        if status == Status.INFEASIBLE:
            self.warm_solves += 1
            return Result(tuple([0.0] * self.n), inf if minimize else -inf, iters, iters, Status.INFEASIBLE)
        if cost_y != self.cost_y:
            lp.set_objective(cost_y)
            self.cost_y = cost_y
        # Always: a dual pivot on a tiny element can leave a reduced cost slightly
        # negative, and the primal pass is a no-op when the basis is already optimal
        status, it = lp.primal(self.max_iter)
        iters += it
        if status in (Status.MAX_ITER, Status.UNBOUNDED):
            return None  # an unbounded ray from a drifted tableau is not trusted: rebuild
        result = self._result(status, iters, c, minimize)
        if status == Status.OPTIMAL and not self._feasible(result.solution):
            return None
        self.warm_solves += 1
        return result

    def _result(self, status: Status, iters: int, c: Sequence[float], minimize: bool) -> Result:
        n = self.n
        if status == Status.INFEASIBLE:
            return Result(tuple([0.0] * n), inf if minimize else -inf, iters, iters, Status.INFEASIBLE)
        assert self.lp is not None and self.std is not None
        if status == Status.MAX_ITER and not self.lp.phase1_done:
            # Stopped before a feasible point was found: there is no solution to report
            return Result(tuple([0.0] * n), inf if minimize else -inf, iters, iters, Status.MAX_ITER)
        x = self.std.to_x(self.lp.column_values())
        if status == Status.UNBOUNDED:
            return Result(tuple(x), -inf if minimize else inf, iters, iters, Status.UNBOUNDED)
        return Result(tuple(x), sum(cj * xj for cj, xj in zip(c, x)), iters, iters, status)

    def _feasible(self, x: Sequence[float]) -> bool:
        tol = 1e-6
        for j in range(self.n):
            if x[j] < self.lb[j] - tol * (1 + abs(self.lb[j])) or x[j] > self.ub[j] + tol * (1 + abs(self.ub[j])):
                return False
        for row, bi, sense in zip(self.rows, self.b, self.senses):
            act = sum(a * x[j] for j, a in row.items())
            slack = tol * (1 + abs(bi))
            if (sense != ">=" and act > bi + slack) or (sense != "<=" and act < bi - slack):
                return False
        return True

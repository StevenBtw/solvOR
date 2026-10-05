"""
Primal heuristics for MILP: rounding with local search, and LNS.

Internal module used by solve_milp and MilpModel. The rounding heuristic keeps
row activities up to date incrementally: a move only touches the rows its
variable appears in, so checking a move costs O(nonzeros in that column)
instead of a pass over the whole matrix.
"""

from math import ceil, floor, inf

from solvor.lns import lns as _lns
from solvor.lp_engine import solve_cold
from solvor.types import Status

__all__ = ["is_feasible", "lns_improve", "round_binary", "row_ok"]


def row_ok(activity: float, rhs: float, sense: str, eps: float) -> bool:
    if sense == "<=":
        return activity <= rhs + eps
    if sense == ">=":
        return activity >= rhs - eps
    return abs(activity - rhs) <= eps


def is_feasible(x, prob, lower, upper, int_set, eps):
    for j in range(prob.n):
        if not lower[j] - eps <= x[j] <= upper[j] + eps:
            return False
    for j in int_set:
        if abs(x[j] - round(x[j])) > eps:
            return False
    for row, bi, sense in zip(prob.rows, prob.b, prob.senses):
        if not row_ok(sum(a * x[j] for j, a in row.items()), bi, sense, eps):
            return False
    return True


def round_binary(lp_solution, int_set, c, prob, cols, lower, upper, minimize, eps):
    """Greedy rounding with flip and swap local search.

    Row activities are kept up to date incrementally: a move only touches the
    rows its variable appears in (cols[j]), so checking a move costs
    O(nonzeros in that column) instead of a pass over the whole matrix.
    """
    sol = list(lp_solution)
    sign = 1 if minimize else -1
    b, senses = prob.b, prob.senses
    act = [sum(a * sol[j] for j, a in row.items()) for row in prob.rows]

    def move(j, value):
        delta = value - sol[j]
        sol[j] = value
        if delta:
            for i, a in cols[j]:
                act[i] += a * delta

    def ok(*js):
        for j in js:
            if not lower[j] - eps <= sol[j] <= upper[j] + eps:
                return False
            for i, _ in cols[j]:
                if not row_ok(act[i], b[i], senses[i], eps):
                    return False
        return True

    # Round fractional vars, preferring low-impact first
    candidates = [
        (sign * c[j], lp_solution[j], j) for j in int_set if abs(lp_solution[j] - round(lp_solution[j])) > eps
    ]
    candidates.sort()

    for _, val, j in candidates:
        rounded = round(val)
        move(j, float(rounded))
        if not ok(j):
            move(j, 1.0 - rounded)
            if not ok(j):
                return None

    if not is_feasible(sol, prob, lower, upper, int_set, eps):
        return None

    # Phase 1: flip improvement
    improved = True
    while improved:
        improved = False
        flip_candidates = [
            (sign * c[j], j) for j in int_set if (minimize and sol[j] > 0.5) or (not minimize and sol[j] < 0.5)
        ]
        flip_candidates.sort()

        for _, j in flip_candidates:
            old_val = sol[j]
            move(j, 1.0 - old_val)
            if ok(j):
                improved = True
            else:
                move(j, old_val)

    # Phase 2: swap improvement
    improved = True
    while improved:
        improved = False
        zeros = [j for j in int_set if sol[j] < 0.5]
        ones = [j for j in int_set if sol[j] > 0.5]

        best_gain, best_swap = 0, None
        for j_on in zeros:
            gain_on = -sign * c[j_on]
            for j_off in ones:
                net_gain = gain_on + sign * c[j_off]
                if net_gain > best_gain:
                    move(j_on, 1.0)
                    move(j_off, 0.0)
                    if ok(j_on, j_off):
                        best_gain, best_swap = net_gain, (j_on, j_off)
                    move(j_on, 0.0)
                    move(j_off, 1.0)

        if best_swap:
            j_on, j_off = best_swap
            move(j_on, 1.0)
            move(j_off, 0.0)
            improved = True

    return tuple(sol)


def lns_improve(solution, c, prob, lower, upper, int_set, minimize, eps, max_iter, iterations, destroy_frac, rng):
    n = len(solution)
    int_list = list(int_set)
    k = max(1, int(len(int_list) * destroy_frac))

    def objective_fn(sol):
        return sum(c[j] * sol[j] for j in range(n))

    def destroy(sol, rng):
        unfixed = set(rng.sample(int_list, min(k, len(int_list))))
        return (sol, unfixed)

    def repair(partial, _):
        sol, unfixed = partial
        candidate = solve_sub_mip(sol, c, prob, lower, upper, int_set, unfixed, minimize, eps, max_iter)
        return candidate if candidate else sol

    result = _lns(
        solution,
        objective_fn,
        destroy,
        repair,
        minimize=minimize,
        max_iter=iterations,
        max_no_improve=iterations,
        seed=rng.randint(0, 2**31),
    )
    return result.solution, result.evaluations


def solve_sub_mip(current_sol, c, prob, base_lower, base_upper, int_set, free_vars, minimize, eps, max_iter):
    n = len(c)
    sign = 1 if minimize else -1

    lower, upper = list(base_lower), list(base_upper)
    for j in int_set:
        if j in free_vars:
            lower[j], upper[j] = max(lower[j], 0.0), min(upper[j], 1.0)
        else:
            lower[j] = upper[j] = current_sol[j]

    result = solve_cold(prob, c, lower, upper, minimize=minimize, eps=eps, max_iter=max_iter)
    if result.status != Status.OPTIMAL:
        return None

    sol = list(result.solution)
    frac_vars = [j for j in free_vars if abs(sol[j] - round(sol[j])) > eps]
    if not frac_vars:
        return tuple(sol)

    # Small B&B for sub-problem
    best_sol, best_obj = None, inf if minimize else -inf

    rounded = list(sol)
    for j in free_vars:
        rounded[j] = round(rounded[j])
    if is_feasible(rounded, prob, base_lower, base_upper, int_set, eps):
        best_sol = tuple(rounded)
        best_obj = sum(c[j] * rounded[j] for j in range(n))

    stack = [(list(lower), list(upper))]
    nodes = 0

    while stack and nodes < 100:
        lo, hi = stack.pop()
        nodes += 1

        res = solve_cold(prob, c, lo, hi, minimize=minimize, eps=eps, max_iter=max_iter)
        if res.status != Status.OPTIMAL:
            continue
        if best_sol is not None and sign * res.objective >= sign * best_obj - eps:
            continue

        branch_var, best_frac = None, 0
        for j in free_vars:
            frac = abs(res.solution[j] - round(res.solution[j]))
            if frac > eps and frac > best_frac:
                branch_var, best_frac = j, frac

        if branch_var is None:
            obj = res.objective
            if sign * obj < sign * best_obj:
                best_sol, best_obj = tuple(res.solution), obj
            continue

        val = res.solution[branch_var]
        lo_down, hi_down = list(lo), list(hi)
        hi_down[branch_var] = floor(val)
        stack.append((lo_down, hi_down))

        lo_up, hi_up = list(lo), list(hi)
        lo_up[branch_var] = ceil(val)
        stack.append((lo_up, hi_up))

    return best_sol

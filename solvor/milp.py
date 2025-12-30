r"""
MILP Solver, linear optimization with integer constraints.

Linear programming with integer constraints. The workhorse of discrete
optimization: diet problems, scheduling, set covering, facility location.

    from solvor.milp import solve_milp

    # minimize c @ x, subject to A @ x <= b, x >= 0, some x integer
    result = solve_milp(c, A, b, integers=[0, 2])
    result = solve_milp(c, A, b, integers=[0, 1], minimize=False)  # maximize

    # warm start from previous solution (prunes search tree)
    result = solve_milp(c, A, b, integers=[0, 2], warm_start=previous.solution)

    # find multiple solutions (result.solutions contains all found)
    result = solve_milp(c, A, b, integers=[0, 2], solution_limit=5)

How it works: branch and bound using simplex as subroutine. Solves LP relaxations
(ignoring integer constraints), branches on fractional values, prunes subtrees
that can't beat current best. Best-first search prioritizes promising branches.

Use this for:

- Linear objectives with integer constraints
- Diet/blending, scheduling, set covering
- Facility location, power grid design
- When you need proven optimal values

Parameters:

    c: objective coefficients
    A: constraint matrix
    b: constraint bounds
    integers: indices of integer-constrained variables
    warm_start: initial solution to prune search tree
    solution_limit: find multiple solutions

CP is more expressive for logical constraints. SAT handles pure boolean.
For continuous-only problems, use simplex directly.
"""

from collections.abc import Sequence
from heapq import heappop, heappush
from math import ceil, floor
from typing import NamedTuple

from solvor.simplex import Status as LPStatus
from solvor.simplex import solve_lp
from solvor.types import Result, Status
from solvor.utils import check_integers_valid, check_matrix_dims, warn_large_coefficients

__all__ = ["solve_milp"]


# I deliberately picked NamedTuple over dataclass for performance
class Node(NamedTuple):
    bound: float
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    depth: int


def solve_milp(
    c: Sequence[float],
    A: Sequence[Sequence[float]],
    b: Sequence[float],
    integers: Sequence[int],
    *,
    minimize: bool = True,
    eps: float = 1e-6,
    max_iter: int = 10_000,
    max_nodes: int = 100_000,
    gap_tol: float = 1e-6,
    warm_start: Sequence[float] | None = None,
    solution_limit: int = 1,
) -> Result:
    n = len(c)
    check_matrix_dims(c, A, b)
    check_integers_valid(integers, n)
    warn_large_coefficients(A)

    int_set = set(integers)
    total_iters = 0

    lower = [0.0] * n
    upper = [float("inf")] * n

    root_result = _solve_node(c, A, b, lower, upper, minimize, eps, max_iter)
    total_iters += root_result.iterations

    if root_result.status == LPStatus.INFEASIBLE:
        return Result(None, float("inf") if minimize else float("-inf"),
                     0, total_iters, Status.INFEASIBLE)

    if root_result.status == LPStatus.UNBOUNDED:
        return Result(None, float("-inf") if minimize else float("inf"),
                     0, total_iters, Status.UNBOUNDED)

    best_solution, best_obj = None, float("inf") if minimize else float("-inf")
    sign = 1 if minimize else -1
    all_solutions: list[tuple] = []

    # Use warm start as initial incumbent if provided and feasible
    if warm_start is not None:
        ws = tuple(warm_start)
        if len(ws) == n and _is_feasible(ws, A, b, int_set, eps):
            best_obj = sum(c[j] * ws[j] for j in range(n))
            best_solution = ws
            all_solutions.append(ws)

    frac_var = _most_fractional(root_result.solution, int_set, eps)

    if frac_var is None:
        return Result(root_result.solution, root_result.objective, 1, total_iters)

    # Detect binary vars from LP solution range
    is_binary = all(-eps <= root_result.solution[j] <= 1 + eps for j in int_set)
    if is_binary:
        for j in int_set:
            lower[j] = max(lower[j], 0.0)
            upper[j] = min(upper[j], 1.0)

        # Rounding heuristic finds good incumbent fast
        if best_solution is None:
            rounded = _round_binary(root_result.solution, int_set, c, A, b, minimize, eps)
            if rounded is not None:
                best_obj = sum(c[j] * rounded[j] for j in range(n))
                best_solution = rounded
                all_solutions.append(rounded)

    tree: list[tuple[float, int, Node]] = []
    counter = 0
    root_bound = sign * root_result.objective
    heappush(tree, (root_bound, counter, Node(root_bound, tuple(lower), tuple(upper), 0)))
    counter += 1
    nodes_explored = 0

    while tree and nodes_explored < max_nodes:
        node_bound, _, node = heappop(tree)

        # Prune if can't improve
        if best_solution is not None and node_bound >= sign * best_obj - eps:
            continue

        result = _solve_node(c, A, b, node.lower, node.upper, minimize, eps, max_iter)
        total_iters += result.iterations
        nodes_explored += 1

        if result.status != LPStatus.OPTIMAL:
            continue

        if best_solution is not None and sign * result.objective >= sign * best_obj - eps:
            continue

        frac_var = _most_fractional(result.solution, int_set, eps)

        if frac_var is None:
            # Found an integer-feasible solution
            sol = tuple(result.solution)
            sol_obj = result.objective

            # Collect solution if within limit
            if solution_limit > 1 and sol not in all_solutions:
                all_solutions.append(sol)
                if len(all_solutions) >= solution_limit:
                    return Result(best_solution or sol,
                                 best_obj if best_solution else sol_obj,
                                 nodes_explored, total_iters, Status.FEASIBLE,
                                 solutions=tuple(all_solutions))

            if sign * sol_obj < sign * best_obj:
                best_solution, best_obj = sol, sol_obj
                gap = _compute_gap(best_obj, node_bound / sign if node_bound != 0 else 0, minimize)
                if gap < gap_tol and solution_limit == 1:
                    return Result(best_solution, best_obj, nodes_explored, total_iters)

            continue

        val = result.solution[frac_var]
        child_bound = sign * result.objective

        lower_left, upper_left = list(node.lower), list(node.upper)
        upper_left[frac_var] = floor(val)
        heappush(tree, (child_bound, counter,
                       Node(child_bound, tuple(lower_left), tuple(upper_left), node.depth + 1)))
        counter += 1

        lower_right, upper_right = list(node.lower), list(node.upper)
        lower_right[frac_var] = ceil(val)
        heappush(tree, (child_bound, counter,
                       Node(child_bound, tuple(lower_right), tuple(upper_right), node.depth + 1)))
        counter += 1

    if best_solution is None:
        return Result(None, float("inf") if minimize else float("-inf"),
                     nodes_explored, total_iters, Status.INFEASIBLE)

    status = Status.OPTIMAL if not tree else Status.FEASIBLE
    if solution_limit > 1 and all_solutions:
        return Result(best_solution, best_obj, nodes_explored, total_iters, status,
                     solutions=tuple(all_solutions))
    return Result(best_solution, best_obj, nodes_explored, total_iters, status)


def _solve_node(c, A, b, lower, upper, minimize, eps, max_iter):
    # Substitute fixed variables to reduce problem size
    n = len(c)
    fixed = {}
    free_vars = []

    for j in range(n):
        lo, hi = lower[j], upper[j]
        if hi < lo - eps:
            return Result(None, float("inf") if minimize else float("-inf"),
                         0, 0, LPStatus.INFEASIBLE)
        if hi - lo < eps:
            fixed[j] = lo
        else:
            free_vars.append(j)

    if not free_vars:
        obj = sum(c[j] * fixed[j] for j in fixed)
        sol = [fixed.get(j, 0.0) for j in range(n)]
        for i, row in enumerate(A):
            lhs = sum(row[j] * sol[j] for j in range(n))
            if lhs > b[i] + eps:
                return Result(None, float("inf") if minimize else float("-inf"),
                             0, 0, LPStatus.INFEASIBLE)
        return Result(tuple(sol), obj, 0, 0, LPStatus.OPTIMAL)

    # Build reduced problem
    n_free = len(free_vars)
    A_red, b_red = [], []

    for i, row in enumerate(A):
        fixed_contrib = sum(row[j] * fixed[j] for j in fixed)
        new_rhs = b[i] - fixed_contrib
        if new_rhs < -eps:
            return Result(None, float("inf") if minimize else float("-inf"),
                         0, 0, LPStatus.INFEASIBLE)
        A_red.append([row[j] for j in free_vars])
        b_red.append(new_rhs)

    # Only add non-trivial bounds
    for j_new, j_old in enumerate(free_vars):
        lo, hi = lower[j_old], upper[j_old]
        if lo > eps:
            row = [0.0] * n_free
            row[j_new] = -1.0
            A_red.append(row)
            b_red.append(-lo)
        if hi < float("inf"):
            row = [0.0] * n_free
            row[j_new] = 1.0
            A_red.append(row)
            b_red.append(hi)

    c_red = [c[j] for j in free_vars]
    fixed_obj = sum(c[j] * fixed[j] for j in fixed)

    result = solve_lp(c_red, A_red, b_red, minimize=minimize, eps=eps, max_iter=max_iter)

    if result.status != LPStatus.OPTIMAL:
        return result

    # Reconstruct full solution
    full_sol = [0.0] * n
    for j in fixed:
        full_sol[j] = fixed[j]
    for j_new, j_old in enumerate(free_vars):
        full_sol[j_old] = result.solution[j_new]

    return Result(tuple(full_sol), result.objective + fixed_obj,
                 result.iterations, result.iterations, result.status)


def _most_fractional(solution, int_set, eps):
    best_var, best_frac = None, 0.0
    for j in int_set:
        val = solution[j]
        frac = abs(val - round(val))
        if frac > eps and frac > best_frac:
            best_var, best_frac = j, frac
    return best_var


def _compute_gap(best_obj, bound, minimize):
    if abs(best_obj) < 1e-10:
        return abs(best_obj - bound)
    return abs(best_obj - bound) / abs(best_obj)


def _is_feasible(x, A, b, int_set, eps):
    n = len(x)
    if any(x[j] < -eps for j in range(n)):
        return False
    for j in int_set:
        if abs(x[j] - round(x[j])) > eps:
            return False
    for i, row in enumerate(A):
        lhs = sum(row[j] * x[j] for j in range(n))
        if lhs > b[i] + eps:
            return False
    return True


def _round_binary(lp_solution, int_set, c, A, b, minimize, eps):
    # Greedy rounding with local search improvement
    n = len(lp_solution)
    sol = list(lp_solution)
    sign = 1 if minimize else -1

    # Round fractional vars, preferring low-impact first
    candidates = [(sign * c[j], lp_solution[j], j) for j in int_set
                  if abs(lp_solution[j] - round(lp_solution[j])) > eps]
    candidates.sort()

    for _, val, j in candidates:
        rounded = round(val)
        sol[j] = float(rounded)

        feasible = True
        for i, row in enumerate(A):
            if sum(row[k] * sol[k] for k in range(n)) > b[i] + eps:
                feasible = False
                break

        if not feasible:
            sol[j] = 1.0 - rounded
            feasible = True
            for i, row in enumerate(A):
                if sum(row[k] * sol[k] for k in range(n)) > b[i] + eps:
                    feasible = False
                    break
            if not feasible:
                return None

    if not _is_feasible(sol, A, b, int_set, eps):
        return None

    # Phase 1: flip improvement
    improved = True
    while improved:
        improved = False
        flip_candidates = [(sign * c[j], j) for j in int_set
                          if (minimize and sol[j] > 0.5) or (not minimize and sol[j] < 0.5)]
        flip_candidates.sort()

        for _, j in flip_candidates:
            old_val = sol[j]
            sol[j] = 1.0 - old_val
            if _is_feasible(sol, A, b, int_set, eps):
                improved = True
            else:
                sol[j] = old_val

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
                    sol[j_on], sol[j_off] = 1.0, 0.0
                    if _is_feasible(sol, A, b, int_set, eps):
                        best_gain, best_swap = net_gain, (j_on, j_off)
                    sol[j_on], sol[j_off] = 0.0, 1.0

        if best_swap:
            j_on, j_off = best_swap
            sol[j_on], sol[j_off] = 1.0, 0.0
            improved = True

    return tuple(sol)

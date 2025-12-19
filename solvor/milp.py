"""
MILP Solver, linear optimization with integer constraints.

Use this when you need to minimize or maximize a linear objective with linear
constraints, and some variables must be integers. Classic problems: diet/blending
(minimize cost meeting nutritional requirements), scheduling with discrete slots,
set covering, power grid design, energy management (optimizing consumption across
timeslots based on production forecasts).

Under the hood, this is branch and bound using simplex as a subroutine. It solves
LP relaxations (ignoring integer constraints), then branches on fractional values,
pruning subtrees that can't beat the current best. Best-first search prioritizes
the most promising branches.

    from solvor.milp import solve_milp

    # minimize c @ x, subject to A @ x <= b, x >= 0, some x integer
    result = solve_milp(c, A, b, integers=[0, 2])
    result = solve_milp(c, A, b, integers=[0, 1], minimize=False)  # maximize

CP is more expressive for logical constraints like "all different" but
doesn't optimize. SAT handles boolean satisfiability only. MILP is for
when you have a clear linear objective and need the optimal value.

If your problem is purely continuous (no integers), just use simplex (directly).
"""

from collections.abc import Sequence
from heapq import heappush, heappop
from math import floor, ceil
from typing import NamedTuple
from solvor.simplex import solve_lp, Status as LPStatus
from solvor.types import Status, Result

__all__ = ["solve_milp"]

# I deliberatly picked NamedTuple over dataclass for performance
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
) -> Result:
   
    n = len(c)
    int_set = set(integers)
    total_iters = 0

    lower = [0.0] * n
    upper = [float('inf')] * n

    root_result = _solve_node(c, A, b, lower, upper, minimize, eps, max_iter)
    total_iters += root_result.iterations

    if root_result.status == LPStatus.INFEASIBLE:
        return Result(None, float('inf') if minimize else float('-inf'), 0, total_iters, Status.INFEASIBLE)

    if root_result.status == LPStatus.UNBOUNDED:
        return Result(None, float('-inf') if minimize else float('inf'), 0, total_iters, Status.UNBOUNDED)

    best_solution, best_obj = None, float('inf') if minimize else float('-inf')
    sign = 1 if minimize else -1

    frac_var = _most_fractional(root_result.solution, int_set, eps)

    if frac_var is None:
        return Result(root_result.solution, root_result.objective, 1, total_iters)

    tree = []
    counter = 0
    root_bound = sign * root_result.objective
    heappush(tree, (root_bound, counter, Node(root_bound, list(lower), list(upper), 0)))
    counter += 1
    nodes_explored = 0

    while tree and nodes_explored < max_nodes:
        node_bound, _, node = heappop(tree)

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
            if sign * result.objective < sign * best_obj:
                best_solution, best_obj = result.solution, result.objective

                if best_solution is not None:
                    gap = _compute_gap(best_obj, node_bound / sign if node_bound != 0 else 0, minimize)

                    if gap < gap_tol:
                        return Result(best_solution, best_obj, nodes_explored, total_iters)

            continue

        val = result.solution[frac_var]
        child_bound = sign * result.objective

        lower_left, upper_left = list(node.lower), list(node.upper)
        upper_left[frac_var] = floor(val)
        heappush(tree, (child_bound, counter, Node(child_bound, lower_left, upper_left, node.depth + 1)))
        counter += 1

        lower_right, upper_right = list(node.lower), list(node.upper)
        lower_right[frac_var] = ceil(val)
        heappush(tree, (child_bound, counter, Node(child_bound, lower_right, upper_right, node.depth + 1)))
        counter += 1

    if best_solution is None:
        return Result(None, float('inf') if minimize else float('-inf'), nodes_explored, total_iters, Status.INFEASIBLE)

    status = Status.OPTIMAL if not tree else Status.FEASIBLE
    return Result(best_solution, best_obj, nodes_explored, total_iters, status)

def _solve_node(c, A, b, lower, upper, minimize, eps, max_iter):
    n = len(c)

    bound_rows = []
    bound_rhs = []

    for j in range(n):
        if lower[j] > eps:
            row = [0.0] * n
            row[j] = -1.0
            bound_rows.append(row)
            bound_rhs.append(-lower[j])

        if upper[j] < float('inf'):
            row = [0.0] * n
            row[j] = 1.0
            bound_rows.append(row)
            bound_rhs.append(upper[j])

    if bound_rows:
        A_ext = [list(row) for row in A] + bound_rows
        b_ext = list(b) + bound_rhs
    else:
        A_ext, b_ext = A, b

    return solve_lp(c, A_ext, b_ext, minimize=minimize, eps=eps, max_iter=max_iter)

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
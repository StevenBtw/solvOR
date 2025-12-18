"""
MILP Solver - Branch and Bound for Mixed Integer Linear Programming

Solves: minimize c @ x  subject to  A @ x <= b, x >= 0, x[i] ∈ Z for i in integers

Branch and Bound explores a tree of LP relaxations. Each node adds bounds on
integer variables. Nodes are pruned when their bound exceeds the best_solution.
Best-first selection prioritizes nodes with the tightest bound.

Usage:
    from solvor.milp import solve_milp, Status
    result = solve_milp(c, A, b, integers=[0, 2])
    result = solve_milp(c, A, b, integers=[0, 1], minimize=False)

Parameters:
    c         : Objective weights (n,)
    A         : Constraint matrix (m x n)
    b         : Right-hand side (m,)
    integers  : Indices of integer-constrained variables
    minimize  : True for min, False for max (default: True)
    eps       : Numerical tolerance (default: 1e-6)
    max_iter  : Maximum simplex iterations per node (default: 10,000)
    max_nodes : Maximum nodes explored in tree (default: 100,000)
    gap_tol   : Stop when relative gap < this (default: 1e-6)

Returns Result(solution, objective, iterations, evaluations, status)
    iterations = nodes explored, evaluations = total simplex iterations

Parameter impact:
    max_nodes too low  -> may miss optimal, returns best feasible
    gap_tol too tight  -> more nodes explored for marginal improvement
    eps too tight      -> false integrality detection
"""

from collections import namedtuple
from collections.abc import Sequence
from heapq import heappush, heappop
from math import floor, ceil
from solvor.simplex import solve_lp, Status as LPStatus
from solvor.types import Status, Result

__all__ = ["solve_milp", "Status", "Result"]

Node = namedtuple('Node', ['bound', 'lower', 'upper', 'depth'])

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
    """(c, A, b, integers, opts) -> Result with optimal integer solution or status."""
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
        return Result(root_result.solution, root_result.objective, 1, total_iters, Status.OPTIMAL)

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
                        return Result(best_solution, best_obj, nodes_explored, total_iters, Status.OPTIMAL)

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
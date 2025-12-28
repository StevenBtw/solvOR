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
        return Result(None, float("inf") if minimize else float("-inf"), 0, total_iters, Status.INFEASIBLE)

    if root_result.status == LPStatus.UNBOUNDED:
        return Result(None, float("-inf") if minimize else float("inf"), 0, total_iters, Status.UNBOUNDED)

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

    tree = []
    counter = 0
    root_bound = sign * root_result.objective
    heappush(tree, (root_bound, counter, Node(root_bound, tuple(lower), tuple(upper), 0)))
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
            # Found an integer-feasible solution
            sol = tuple(result.solution)
            sol_obj = result.objective

            # Collect solution if within limit
            if solution_limit > 1 and sol not in all_solutions:
                all_solutions.append(sol)
                if len(all_solutions) >= solution_limit:
                    return Result(
                        best_solution or sol,
                        best_obj if best_solution else sol_obj,
                        nodes_explored,
                        total_iters,
                        Status.FEASIBLE,
                        solutions=tuple(all_solutions),
                    )

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
        left_node = Node(child_bound, tuple(lower_left), tuple(upper_left), node.depth + 1)
        heappush(tree, (child_bound, counter, left_node))
        counter += 1

        lower_right, upper_right = list(node.lower), list(node.upper)
        lower_right[frac_var] = ceil(val)
        right_node = Node(child_bound, tuple(lower_right), tuple(upper_right), node.depth + 1)
        heappush(tree, (child_bound, counter, right_node))
        counter += 1

    if best_solution is None:
        return Result(None, float("inf") if minimize else float("-inf"), nodes_explored, total_iters, Status.INFEASIBLE)

    status = Status.OPTIMAL if not tree else Status.FEASIBLE
    if solution_limit > 1 and all_solutions:
        return Result(best_solution, best_obj, nodes_explored, total_iters, status, solutions=tuple(all_solutions))
    return Result(best_solution, best_obj, nodes_explored, total_iters, status)


def _solve_node(c, A, b, lower, upper, minimize, eps, max_iter):
    n = len(c)

    bound_rows = []
    bound_rhs = []

    for j in range(n):
        if lower[j] > -float("inf"):
            row = [0.0] * n
            row[j] = -1.0
            bound_rows.append(row)
            bound_rhs.append(-lower[j])

        if upper[j] < float("inf"):
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


def _is_feasible(x, A, b, int_set, eps):
    """Check if solution satisfies constraints and integrality."""
    n = len(x)
    # Non-negativity
    if any(x[j] < -eps for j in range(n)):
        return False
    # Integrality
    for j in int_set:
        if abs(x[j] - round(x[j])) > eps:
            return False
    # Constraints A @ x <= b
    for i, row in enumerate(A):
        lhs = sum(row[j] * x[j] for j in range(n))
        if lhs > b[i] + eps:
            return False
    return True

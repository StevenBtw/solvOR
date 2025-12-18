"""
Simplex Solver - Two-Phase Linear Programming

Solves: minimize c @ x  subject to  A @ x <= b, x >= 0

The simplex algorithm pivots through basic feasible solutions along edges
of the feasible polytope. Phase I finds an initial feasible solution using
artificial variables. Phase II optimizes the objective using Bland's rule
to prevent cycling in degenerate cases.

Usage:
    from src.simplex import solve_lp, Status
    result = solve_lp(c, A, b)
    result = solve_lp(c, A, b, minimize=False)  # maximize

Parameters:
    c         : Objective weights (n,)
    A         : Constraint matrix (m x n)
    b         : Right-hand side (m,)
    minimize  : True for min, False for max (default: True)
    eps       : Numerical tolerance (default: 1e-10)
    max_iter  : Maximum pivot iterations (default: 100,000)

Returns Result(solution, objective, iterations, evaluations, status)

Status values:
    OPTIMAL    - Proven optimal solution found
    INFEASIBLE - No feasible solution exists
    UNBOUNDED  - Objective can improve infinitely
    MAX_ITER   - Iteration limit reached
"""

from array import array
from collections import namedtuple
from enum import IntEnum, auto

__all__ = ["solve_lp", "Status", "Result"]

EPS = 1e-10

class Status(IntEnum):
    OPTIMAL = auto()
    FEASIBLE = auto()
    INFEASIBLE = auto()
    UNBOUNDED = auto()
    MAX_ITER = auto()

Result = namedtuple('Result', ['solution', 'objective', 'iterations', 'evaluations', 'status'])

def solve_lp(c, A, b, minimize=True, eps=EPS, max_iter=100_000):
    """(c, A, b, opts) -> Result with optimal solution or status."""
    m, n = len(b), len(c)
    weights = list(c) if minimize else [-ci for ci in c]

    matrix = []
    for i in range(m):
        row = array('d', A[i])
        row.extend([0.0] * m)
        row[n + i] = 1.0
        row.append(b[i])
        matrix.append(row)

    obj = array('d', weights)
    obj.extend([0.0] * (m + 1))
    matrix.append(obj)

    basis = array('i', range(n, n + m))
    basis_set = set(basis)

    if any(matrix[i][-1] < -eps for i in range(m)):
        status, iters, matrix, basis, basis_set = _phase1(matrix, basis, basis_set, m, n, eps, max_iter)
        if status != Status.OPTIMAL:
            return Result(tuple([0.0] * n), float('inf'), iters, iters, Status.INFEASIBLE)
        max_iter -= iters
    else:
        iters = 0

    status, iters2, matrix, basis, basis_set = _phase2(matrix, basis, basis_set, m, eps, max_iter)
    return _extract(matrix, basis, m, n, status, iters + iters2, minimize)

def _phase1(matrix, basis, basis_set, m, n, eps, max_iter):
    n_cols = len(matrix[0])
    n_total = n + m
    art_cols = []

    for i in range(m):
        if matrix[i][-1] < -eps:
            for j in range(n_cols):
                matrix[i][j] *= -1

            art_col = n_total + len(art_cols)

            for row in matrix:
                row.insert(-1, 0.0)

            matrix[i][-2] = 1.0
            basis_set.discard(basis[i])
            basis[i] = art_col
            basis_set.add(art_col)
            art_cols.append(art_col)

    if not art_cols:
        return Status.OPTIMAL, 0, matrix, basis, basis_set

    n_cols = len(matrix[0])
    orig_obj = array('d', matrix[-1])
    matrix[-1] = array('d', [0.0] * n_cols)

    for col in art_cols:
        matrix[-1][col] = 1.0

    for i in range(m):
        if basis[i] in art_cols:
            for j in range(n_cols):
                matrix[-1][j] -= matrix[i][j]

    status, iters, matrix, basis, basis_set = _phase2(matrix, basis, basis_set, m, eps, max_iter)

    if matrix[-1][-1] < -eps:
        return Status.INFEASIBLE, iters, matrix, basis, basis_set

    for _ in art_cols:
        for row in matrix:
            del row[-2]

    matrix[-1] = orig_obj
    n_cols = len(matrix[0])

    for i in range(m):
        var = basis[i]
        if var < n_cols - 1:
            cost = matrix[-1][var]
            if abs(cost) > eps:
                for j in range(n_cols):
                    matrix[-1][j] -= cost * matrix[i][j]

    return Status.OPTIMAL, iters, matrix, basis, basis_set

def _phase2(matrix, basis, basis_set, m, eps, max_iter):
    n_cols = len(matrix[0])

    for iteration in range(max_iter):
        enter = -1

        for j in range(n_cols - 1):
            if j not in basis_set and matrix[-1][j] < -eps:
                enter = j
                break

        if enter == -1:
            return Status.OPTIMAL, iteration, matrix, basis, basis_set

        leave, min_ratio = -1, float('inf')

        for i in range(m):
            if matrix[i][enter] > eps:
                ratio = matrix[i][-1] / matrix[i][enter]
                if ratio < min_ratio - eps:
                    min_ratio, leave = ratio, i
                elif abs(ratio - min_ratio) <= eps:
                    if leave == -1 or basis[i] < basis[leave]:
                        leave = i

        if leave == -1:
            return Status.UNBOUNDED, iteration, matrix, basis, basis_set

        matrix = _pivot(matrix, m, leave, enter, eps)
        basis_set.discard(basis[leave])
        basis[leave] = enter
        basis_set.add(enter)

    return Status.MAX_ITER, max_iter, matrix, basis, basis_set

def _pivot(matrix, m, row, col, eps):
    n_cols = len(matrix[0])
    inv = 1.0 / matrix[row][col]

    for j in range(n_cols):
        matrix[row][j] *= inv

    for i in range(m + 1):
        if i != row:
            f = matrix[i][col]
            if abs(f) > eps:
                for j in range(n_cols):
                    matrix[i][j] -= f * matrix[row][j]

    return matrix

def _extract(matrix, basis, m, n, status, iters, minimize):
    solution = [0.0] * n

    for i in range(m):
        if basis[i] < n:
            solution[basis[i]] = matrix[i][-1]

    obj = -matrix[-1][-1]
    if not minimize:
        obj = -obj

    return Result(tuple(solution), obj, iters, iters, status)

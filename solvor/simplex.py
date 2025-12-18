"""
Simplex Solver, linear programming that aged like wine.

You're walking along edges of a giant crystal always uphill, until you hit a corner
that's optimal. You can visualize it. Most algorithms are abstract symbol
manipulation. Simplex is a journey through space.

Use this for linear objectives with linear constraints and continuous variables.
Resource allocation, blending, production planning, transportation. Unlike
gradient descent which approximates, simplex finds the exact optimum for LP.

    from solvor.simplex import solve_lp

    # minimize c @ x, subject to A @ x <= b, x >= 0
    result = solve_lp(c, A, b)
    result = solve_lp(c, A, b, minimize=False)  # maximize

This is also doing the grunt work inside MILP, which is branch and bound that calls simplex
repeatedly to solve LP relaxations.

Don't use this for: integer constraints (use MILP), non-linear objectives
(use gradient or anneal), or problems with poor numerical scaling (simplex
can struggle with badly scaled coefficients).
"""

from array import array
from collections.abc import Sequence
from solvor.types import Status, Result

__all__ = ["solve_lp", "Status", "Result"]

def solve_lp(
    c: Sequence[float],
    A: Sequence[Sequence[float]],
    b: Sequence[float],
    *,
    minimize: bool = True,
    eps: float = 1e-10,
    max_iter: int = 100_000,
) -> Result:
    
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

    orig_obj = array('d', matrix[-1])

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

"""
Dancing Links (DLX), a Knuth's Algorithm X implementation.

Use this for exact cover problems: Sudoku, N-Queens, pentomino tiling, scheduling
where every constraint must be satisfied exactly once. If your problem has "place
these pieces without overlap" or "fill this grid with exactly one of each" vibes,
DLX is probably a good fit.

Don't use this for: optimization problems (use MIP/SAT), approximate solutions,
or problems where constraints can be partially satisfied.

This is a pure implementation without numpy or other C extensions, just linked lists.
Fast enough for most puzzle-sized problems, but if you're solving industrial-scale
exact cover, consider: sparse matrix representations, iterative deepening, or
constraint propagation before feeding it to DLX.

Example: tiling a 2x3 board with 3 dominoes:

    The board has 6 cells (A-F), each domino placement covers 2 cells.
    Columns = cells to cover, Rows = possible domino placements.

    # Board:  [A][B][C]    Possible placements:
    #         [D][E][F]    0: AB, 1: BC, 2: DE, 3: EF, 4: AD, 5: BE, 6: CF

    from solvor.dlx import solve_exact_cover

    matrix = [
        [1, 1, 0, 0, 0, 0],  # row 0: domino covers A, B
        [0, 1, 1, 0, 0, 0],  # row 1: domino covers B, C
        [0, 0, 0, 1, 1, 0],  # row 2: domino covers D, E
        [0, 0, 0, 0, 1, 1],  # row 3: domino covers E, F
        [1, 0, 0, 1, 0, 0],  # row 4: domino covers A, D
        [0, 1, 0, 0, 1, 0],  # row 5: domino covers B, E
        [0, 0, 1, 0, 0, 1],  # row 6: domino covers C, F
    ]
    result = solve_exact_cover(matrix)
    # result.solution = (0, 3, 6) -> placements AB + EF + CF, or
    #                   (1, 2, 4) -> placements BC + DE + AD, etc.

    result = solve_exact_cover(matrix, find_all=True)
    # result.solution = [(0, 3, 6), (1, 2, 4), ...] all valid tilings
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from solvor.types import Status, Result

__all__ = ["solve_exact_cover", "Status", "Result"]

@dataclass(slots=True)
class _Node:
    column: '_Column | None' = None
    row: int = -1
    left: '_Node | None' = field(default=None, init=False, repr=False)
    right: '_Node | None' = field(default=None, init=False, repr=False)
    up: '_Node | None' = field(default=None, init=False, repr=False)
    down: '_Node | None' = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.left = self
        self.right = self
        self.up = self
        self.down = self

@dataclass(slots=True)
class _Column:
    name: object
    size: int = field(default=0, init=False)
    left: '_Column | None' = field(default=None, init=False, repr=False)
    right: '_Column | None' = field(default=None, init=False, repr=False)
    up: '_Node | None' = field(default=None, init=False, repr=False)
    down: '_Node | None' = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.left = self
        self.right = self
        self.up = self
        self.down = self

def _build_links(matrix, columns=None):
    if not matrix or not matrix[0]:
        return None, []

    n_cols = len(matrix[0])
    col_names = columns if columns else list(range(n_cols))
    root = _Node()
    col_headers = []
    
    prev = root
    for name in col_names:
        col = _Column(name)
        col_headers.append(col)
        col.left = prev
        prev.right = col
        prev = col
    prev.right = root
    root.left = prev

    for row_idx, row in enumerate(matrix):
        first = None
        prev_node = None

        for col_idx, val in enumerate(row):
            if val:
                col = col_headers[col_idx]
                node = _Node(column=col, row=row_idx)

                node.up = col.up
                node.down = col
                col.up.down = node
                col.up = node
                col.size += 1

                if first is None:
                    first = node
                    prev_node = node
                else:
                    node.left = prev_node
                    prev_node.right = node
                    prev_node = node
                    
        if first is not None:
            first.left = prev_node
            prev_node.right = first

    return root, col_headers

def _cover(col):
    col.right.left = col.left
    col.left.right = col.right

    node = col.down
    while node is not col:
        row_node = node.right
        while row_node is not node:
            row_node.down.up = row_node.up
            row_node.up.down = row_node.down
            row_node.column.size -= 1
            row_node = row_node.right
        node = node.down

def _uncover(col):
    node = col.up
    while node is not col:
        row_node = node.left
        while row_node is not node:
            row_node.column.size += 1
            row_node.down.up = row_node
            row_node.up.down = row_node
            row_node = row_node.left
        node = node.up

    col.right.left = col
    col.left.right = col

def solve_exact_cover(
    matrix: Sequence[Sequence[int]],
    *,
    columns: Sequence | None = None,
    find_all: bool = False,
    max_solutions: int | None = None,
    max_iter: int = 10_000_000,
) -> Result:
    if not matrix:
        return Result((), 0, 0, 0, Status.OPTIMAL)

    root, col_headers = _build_links(matrix, columns)
    if root is None:
        return Result((), 0, 0, 0, Status.OPTIMAL)

    solutions = []
    current = []
    iterations = 0
    covers = 0

    def search():
        nonlocal iterations, covers
        iterations += 1
        if iterations > max_iter:
            return False
        
        if root.right is root:
            solutions.append(tuple(current))
            if not find_all:
                return True
            if max_solutions and len(solutions) >= max_solutions:
                return True
            return False

        min_col = None
        min_size = float('inf')
        col = root.right
        while col is not root:
            if col.size < min_size:
                min_size = col.size
                min_col = col
                if min_size == 0:
                    break
            col = col.right

        if min_size == 0:
            return False

        _cover(min_col)
        covers += 1

        row_node = min_col.down
        while row_node is not min_col:
            current.append(row_node.row)

            node = row_node.right
            while node is not row_node:
                _cover(node.column)
                covers += 1
                node = node.right

            if search():
                if not find_all:
                    return True
                if max_solutions and len(solutions) >= max_solutions:
                    return True
                
            current.pop()
            node = row_node.left
            while node is not row_node:
                _uncover(node.column)
                node = node.left

            row_node = row_node.down

        _uncover(min_col)
        return False

    search()

    if iterations > max_iter:
        if solutions:
            sol = solutions if find_all else solutions[0]
            return Result(sol, len(solutions) if find_all else len(sol), iterations, covers, Status.MAX_ITER)
        return Result(None, 0, iterations, covers, Status.MAX_ITER)

    if not solutions:
        return Result(None, 0, iterations, covers, Status.INFEASIBLE)

    if find_all:
        status = Status.FEASIBLE if max_solutions and len(solutions) >= max_solutions else Status.OPTIMAL
        return Result(solutions, len(solutions), iterations, covers, status)

    return Result(solutions[0], len(solutions[0]), iterations, covers, Status.OPTIMAL)
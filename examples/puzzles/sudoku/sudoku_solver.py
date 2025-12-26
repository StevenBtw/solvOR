"""
Sudoku Solver - Constraint Programming

Solve 9x9 Sudoku puzzles using CP-SAT with all_different constraints.
Each cell is an integer variable (1-9), and the classic rules are expressed
as all_different constraints on rows, columns, and 3x3 boxes.

Formulation:
    Variables: cells[r][c] in {1..9}
    Constraints:
    - all_different(row) for each row
    - all_different(column) for each column
    - all_different(box) for each 3x3 box
    - cells[r][c] == puzzle[r][c] for given clues

Why this solver:
    Sudoku is a pure constraint satisfaction problem. The CP-SAT solver
    encodes integer variables as boolean vectors and uses CDCL (Conflict-
    Driven Clause Learning) for efficient search with backjumping.

Expected result:
    The puzzle below (Wikipedia "typical" puzzle) has a unique solution:
    534 | 678 | 912
    672 | 195 | 348
    198 | 342 | 567
    ---------------------
    859 | 761 | 423
    426 | 853 | 791
    713 | 924 | 856
    ---------------------
    961 | 537 | 284
    287 | 419 | 635
    345 | 286 | 179

Reference:
    https://en.wikipedia.org/wiki/Sudoku
    This is the "typical puzzle" shown on the Wikipedia Sudoku page.
"""

from solvor.cp import Model


def solve_sudoku(puzzle):
    """Solve a 9x9 Sudoku puzzle using constraint programming.

    Args:
        puzzle: 9x9 list of lists, 0 for empty cells, 1-9 for clues

    Returns:
        9x9 list of lists with solution, or None if unsolvable
    """
    model = Model()
    n = 9
    box_size = 3

    # Create 9x9 grid of variables (1-9)
    cells = [[model.int_var(1, n, f"c{r}{c}") for c in range(n)] for r in range(n)]

    # Fix known values (clues)
    for r in range(n):
        for c in range(n):
            if puzzle[r][c] != 0:
                model.add(cells[r][c] == puzzle[r][c])

    # Row constraints: each row has all different values
    for r in range(n):
        model.add(model.all_different([cells[r][c] for c in range(n)]))

    # Column constraints: each column has all different values
    for c in range(n):
        model.add(model.all_different([cells[r][c] for r in range(n)]))

    # 3x3 box constraints: each box has all different values
    for box_r in range(box_size):
        for box_c in range(box_size):
            box = [cells[box_r * 3 + r][box_c * 3 + c] for r in range(box_size) for c in range(box_size)]
            model.add(model.all_different(box))

    result = model.solve()

    if result.ok:
        return [[result.solution[f"c{r}{c}"] for c in range(n)] for r in range(n)]
    return None


def print_board(board):
    """Print a 9x9 Sudoku board with box separators."""
    for r, row in enumerate(board):
        if r % 3 == 0 and r > 0:
            print("-" * 21)
        line = ""
        for c, v in enumerate(row):
            if c % 3 == 0 and c > 0:
                line += "| "
            line += f"{v} "
        print(line)


def main():
    # Wikipedia "typical" Sudoku puzzle (0 = empty)
    puzzle = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ]

    print("Sudoku Solver - Constraint Programming")
    print("=" * 40)
    print("\nPuzzle:")
    print_board(puzzle)

    solution = solve_sudoku(puzzle)

    print("\n" + "=" * 40)
    if solution:
        print("SOLUTION")
        print("=" * 40)
        print()
        print_board(solution)
    else:
        print("No solution found")


if __name__ == "__main__":
    main()

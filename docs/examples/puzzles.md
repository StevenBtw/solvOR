# Puzzles

Constraint programming puzzles with complete solutions.

## Sudoku

Fill 9x9 grid so each row, column, and 3x3 box contains 1-9.

[sudoku_solver.py](https://github.com/StevenBtw/solvOR/blob/main/examples/puzzles/sudoku/sudoku_solver.py)

```python
from solvor import Model

def solve_sudoku(puzzle):
    m = Model()
    grid = [[m.int_var(1, 9, f'c{i}{j}') for j in range(9)] for i in range(9)]

    # Row, column, box constraints
    for i in range(9):
        m.add(m.all_different(grid[i]))
        m.add(m.all_different([grid[j][i] for j in range(9)]))

    for br in range(3):
        for bc in range(3):
            cells = [grid[br*3+i][bc*3+j] for i in range(3) for j in range(3)]
            m.add(m.all_different(cells))

    # Given clues
    for i in range(9):
        for j in range(9):
            if puzzle[i][j] != 0:
                m.add(grid[i][j] == puzzle[i][j])

    return m.solve()
```

## N-Queens

Place N queens on NxN board with no conflicts.

[n_queens.py](https://github.com/StevenBtw/solvOR/blob/main/examples/puzzles/n_queens/n_queens.py)

```python
from solvor import Model

def solve_nqueens(n):
    m = Model()
    queens = [m.int_var(0, n-1, f'q{i}') for i in range(n)]

    m.add(m.all_different(queens))  # No same column
    m.add(m.all_different([queens[i] + i for i in range(n)]))  # Diagonals
    m.add(m.all_different([queens[i] - i for i in range(n)]))

    return m.solve()
```

## Magic Square

Fill NxN grid where all rows, columns, diagonals sum to the same value.

[magic_square.py](https://github.com/StevenBtw/solvOR/blob/main/examples/puzzles/magic_square/magic_square.py)

```python
from solvor import Model

def solve_magic_square(n):
    magic_sum = n * (n*n + 1) // 2
    m = Model()
    grid = [[m.int_var(1, n*n, f'c{i}{j}') for j in range(n)] for i in range(n)]

    m.add(m.all_different([grid[i][j] for i in range(n) for j in range(n)]))

    for i in range(n):
        m.add(m.sum_eq(grid[i], magic_sum))  # Rows
        m.add(m.sum_eq([grid[j][i] for j in range(n)], magic_sum))  # Columns

    m.add(m.sum_eq([grid[i][i] for i in range(n)], magic_sum))  # Main diagonal
    m.add(m.sum_eq([grid[i][n-1-i] for i in range(n)], magic_sum))  # Anti-diagonal

    return m.solve()
```

## Einstein's Riddle (Zebra Puzzle)

Classic logic puzzle with 5 houses, 5 nationalities, 5 colors, etc.

[zebra_puzzle.py](https://github.com/StevenBtw/solvOR/blob/main/examples/puzzles/einstein_riddle/zebra_puzzle.py)

"Who owns the zebra?"

## Pentomino Tiling

Tile a 6x10 rectangle with the 12 pentomino pieces.

[pentomino_tiling.py](https://github.com/StevenBtw/solvOR/blob/main/examples/puzzles/pentomino/pentomino_tiling.py)

Uses exact cover (Dancing Links). The 6x10 rectangle has 2,339 solutions.

```python
from solvor import solve_exact_cover

# Build matrix: rows = placements, cols = piece + cells
result = solve_exact_cover(matrix)
```

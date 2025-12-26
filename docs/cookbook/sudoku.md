# Sudoku Solver

Solve 9x9 Sudoku puzzles using constraint programming.

## The Problem

Fill a 9x9 grid so that each row, each column, and each 3x3 box contains the digits 1-9 exactly once. Some cells are pre-filled as clues.

## Why It's Interesting

Sudoku is exact cover in disguise. Each constraint ("row 1 contains a 5") must be satisfied exactly once. Perfect for CP-SAT.

## Example

```python
from solvor import Model

def solve_sudoku(puzzle):
    """
    Solve 9x9 Sudoku.
    puzzle: 9x9 list, use 0 for empty cells
    """
    m = Model()

    # Create variables: grid[i][j] in {1,2,...,9}
    grid = [[m.int_var(1, 9, f'cell_{i}_{j}') for j in range(9)] for i in range(9)]

    # Row constraints: each row has all different
    for i in range(9):
        m.add(m.all_different(grid[i]))

    # Column constraints: each column has all different
    for j in range(9):
        m.add(m.all_different([grid[i][j] for i in range(9)]))

    # Box constraints: each 3x3 box has all different
    for box_r in range(3):
        for box_c in range(3):
            cells = [grid[box_r*3 + i][box_c*3 + j] for i in range(3) for j in range(3)]
            m.add(m.all_different(cells))

    # Given clues
    for i in range(9):
        for j in range(9):
            if puzzle[i][j] != 0:
                m.add(grid[i][j] == puzzle[i][j])

    result = m.solve()

    if result.solution:
        return [[result.solution[f'cell_{i}_{j}'] for j in range(9)] for i in range(9)]
    return None

# Example puzzle (0 = empty)
puzzle = [
    [5,3,0, 0,7,0, 0,0,0],
    [6,0,0, 1,9,5, 0,0,0],
    [0,9,8, 0,0,0, 0,6,0],

    [8,0,0, 0,6,0, 0,0,3],
    [4,0,0, 8,0,3, 0,0,1],
    [7,0,0, 0,2,0, 0,0,6],

    [0,6,0, 0,0,0, 2,8,0],
    [0,0,0, 4,1,9, 0,0,5],
    [0,0,0, 0,8,0, 0,7,9]
]

solution = solve_sudoku(puzzle)
if solution:
    for row in solution:
        print(row)
```

## How It Works

1. **Variables:** Each cell is an integer variable in {1,2,...,9}
2. **Constraints:**
   - `all_different` for each row (9 constraints)
   - `all_different` for each column (9 constraints)
   - `all_different` for each 3x3 box (9 constraints)
   - Equality for given clues

## Variations

### X-Sudoku (Diagonals)

```python
# Add diagonal constraints
m.add(m.all_different([grid[i][i] for i in range(9)]))      # Main diagonal
m.add(m.all_different([grid[i][8-i] for i in range(9)]))    # Anti-diagonal
```

### Killer Sudoku (Cages with Sums)

```python
# Add sum constraints for cages
cage = [grid[0][0], grid[0][1], grid[1][0]]
m.add(m.sum_eq(cage, 12))  # Sum to 12
```

## See Also

- [CP-SAT Model](../algorithms/constraint-programming/cp-sat.md)
- [N-Queens](n-queens.md)

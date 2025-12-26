# N-Queens

Place N queens on an NxN chessboard with no conflicts.

## The Problem

Place N queens so no two queens attack each other. Queens attack along rows, columns, and diagonals.

## Example

```python
from solvor import Model

def solve_n_queens(n):
    m = Model()

    # queens[i] = column position of queen in row i
    queens = [m.int_var(0, n-1, f'q{i}') for i in range(n)]

    # All queens in different columns
    m.add(m.all_different(queens))

    # No two queens on same diagonal
    for i in range(n):
        for j in range(i+1, n):
            m.add(queens[i] + i != queens[j] + j)  # Forward diagonal
            m.add(queens[i] - i != queens[j] - j)  # Backward diagonal

    result = m.solve()
    return [result.solution[f'q{i}'] for i in range(n)] if result.solution else None

solution = solve_n_queens(8)
print(f"8-Queens solution: {solution}")
# Output: [0, 4, 7, 5, 2, 6, 1, 3] (or similar)
```

## How It Works

- One queen per row (implicit in encoding)
- One queen per column (`all_different`)
- Diagonal constraints prevent attacks

## Counting Solutions

To count all solutions:

```python
def count_n_queens(n):
    count = 0
    # Use backtracking or CP-SAT enumeration
    # ...
    return count

# Known counts: 4-queens=2, 8-queens=92, 12-queens=14200
```

## See Also

- [CP-SAT Model](../algorithms/constraint-programming/cp-sat.md)
- [Sudoku](sudoku.md)

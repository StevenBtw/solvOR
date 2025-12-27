# N-Queens

Place N queens on an NxN chessboard with no conflicts. The classic [constraint satisfaction](https://en.wikipedia.org/wiki/Eight_queens_puzzle) demo.

## The Problem

Place N queens so no two attack each other. Queens attack along rows, columns, and diagonals. For 8 queens there are 92 solutions. For 27 queens there are over 234 billion. The search space explodes, but constraints cut it down beautifully.

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
    # Use backtracking or constraint programming enumeration
    # ...
    return count

# Known counts: 4-queens=2, 8-queens=92, 12-queens=14200
```

## See Also

- [Model (CP)](../algorithms/constraint-programming/cp.md)
- [Sudoku](sudoku.md)

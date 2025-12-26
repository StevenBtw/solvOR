# Magic Square

Arrange numbers 1 to N² in an NxN grid so all rows, columns, and diagonals sum to the same "magic constant".

## The Problem

Fill an NxN grid with numbers 1 to N² such that:

- Each number appears exactly once
- All rows sum to the magic constant M = N x (N² + 1) / 2
- All columns sum to M
- Both diagonals sum to M

## Example

```python
from solvor import Model

def solve_magic_square(n: int = 3):
    magic_constant = n * (n * n + 1) // 2

    model = Model()

    # Create NxN grid of variables
    grid = [[model.int_var(1, n * n, f"x_{i}_{j}") for j in range(n)] for i in range(n)]

    # All different: each number 1..N² appears exactly once
    all_vars = [grid[i][j] for i in range(n) for j in range(n)]
    model.add(model.all_different(all_vars))

    # Row sums = magic constant
    for i in range(n):
        model.add(sum(grid[i][j] for j in range(n)) == magic_constant)

    # Column sums = magic constant
    for j in range(n):
        model.add(sum(grid[i][j] for i in range(n)) == magic_constant)

    # Diagonal sums = magic constant
    model.add(sum(grid[i][i] for i in range(n)) == magic_constant)
    model.add(sum(grid[i][n - 1 - i] for i in range(n)) == magic_constant)

    result = model.solve()

    if result.status.is_success:
        for i in range(n):
            row = [result.solution[f"x_{i}_{j}"] for j in range(n)]
            print(" ".join(f"{v:3d}" for v in row))

    return result

solve_magic_square(3)  # Magic constant = 15
solve_magic_square(4)  # Magic constant = 34
```

**Output (3x3):**
```
  2   7   6
  9   5   1
  4   3   8
```

## Magic Constants

| N | Magic Constant | Numbers |
|---|----------------|---------|
| 3 | 15 | 1-9 |
| 4 | 34 | 1-16 |
| 5 | 65 | 1-25 |

## See Also

- [Sudoku](sudoku.md)
- [N-Queens](n-queens.md)

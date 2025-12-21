# solve_exact_cover

**Dancing Links (DLX), Knuth's Algorithm X for exact cover.**

## Summary 

- **Category:** [Exact Cover](Exact-Cover)
- **Problem Type:** Exact cover satisfaction
- **Complexity:** Exponential (fast in practice)
- **Guarantees:** Finds all solutions or proves none exist

## Quick Example

```python
from solvor import solve_exact_cover

# Cover columns using rows (each column exactly once)
matrix = [
    [1, 0, 1, 0],
    [0, 1, 1, 0],
    [1, 1, 0, 1],
    [0, 0, 0, 1]
]

result = solve_exact_cover(matrix)
print(result.solution)  # Tuple of row indices

# Find all solutions
result = solve_exact_cover(matrix, find_all=True)
print(result.solution)  # List of all solutions
```

## See Also
- [Cookbook: Sudoku](Cookbook-Sudoku) - Sudoku as exact cover
- [Exact Cover](Exact-Cover)

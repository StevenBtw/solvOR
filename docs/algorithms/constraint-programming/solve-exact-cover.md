# solve_exact_cover

Dancing Links (Algorithm X). Solves exact cover using linked list "dancing". Nodes remove themselves from the matrix and restore themselves during backtracking. Fast for puzzle-sized problems.

## When to Use

- Sudoku and variants
- N-Queens placement
- Pentomino/polyomino tiling puzzles
- Scheduling where every constraint must be satisfied exactly once
- Set covering where overlaps are forbidden

## Signature

```python
def solve_exact_cover(
    matrix: Sequence[Sequence[int]],
    *,
    columns: Sequence | None = None,
    secondary: Sequence | None = None,
    find_all: bool = False,
    max_solutions: int | None = None,
    max_iter: int = 10_000_000,
) -> Result[tuple[int, ...] | list[tuple[int, ...]]]
```

## Parameters

| Parameter | Description |
|-----------|-------------|
| `matrix` | Binary matrix (0s and 1s) |
| `columns` | Optional names for columns (default: 0, 1, 2, ...) |
| `secondary` | Column names that can be covered 0 or 1 times (optional) |
| `find_all` | If True, find all solutions |
| `max_solutions` | Limit number of solutions |

## Example

```python
from solvor import solve_exact_cover

# Tiling a 2x3 board with dominoes
matrix = [
    [1, 1, 0, 0, 0, 0],  # covers A, B
    [0, 1, 1, 0, 0, 0],  # covers B, C
    [0, 0, 0, 1, 1, 0],  # covers D, E
    [0, 0, 0, 0, 1, 1],  # covers E, F
    [1, 0, 0, 1, 0, 0],  # covers A, D
    [0, 1, 0, 0, 1, 0],  # covers B, E
    [0, 0, 1, 0, 0, 1],  # covers C, F
]

result = solve_exact_cover(matrix)
print(result.solution)  # (4, 5, 6) - rows that cover all columns exactly once
```

## Understanding Exact Cover

Given a matrix of 0s and 1s, find rows such that each column has exactly one 1.

**Modeling problems:**

1. **Identify what needs to be covered** - These become columns
2. **Identify possible choices** - These become rows
3. **Mark which constraints each choice satisfies** - These become 1s

## Complexity

- **Time:** Exponential worst case, very fast in practice for puzzles
- **Guarantees:** Finds all solutions or proves none exist

## Tips

1. **Column ordering matters.** DLX chooses columns with fewest 1s first (automatically).
2. **Start small.** Test with tiny examples (4-Queens, 3x3 grids).
3. **Secondary columns** for optional constraints that can be covered 0 or 1 times.

## See Also

- [CP-SAT Model](cp-sat.md) - More general constraint programming
- [Cookbook: Pentomino](../../cookbook/pentomino.md) - Tiling example
- [Cookbook: Sudoku](../../cookbook/sudoku.md) - Can also use exact cover

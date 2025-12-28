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

## How It Works

**Algorithm X:** Knuth's recursive backtracking for exact cover:

1. If matrix is empty, we've found a solution
2. Pick a column c (one that must be covered)
3. For each row r that covers c:
   - Include r in the partial solution
   - Remove r and all conflicting rows from the matrix
   - Remove all columns that r covers
   - Recurse
   - Restore everything and try the next row
4. If no rows cover c, backtrack (dead end)

**The MRV heuristic:** Always pick the column with fewest 1s. If a column has only one row covering it, there's no choice—take it. If a column has zero rows, fail fast. This prunes the search tree dramatically.

**Dancing Links (DLX):** The matrix is stored as a sparse doubly-linked structure. Each node links to its neighbors in four directions: left, right, up, down.

```text
     c1    c2    c3
      ↓     ↓     ↓
     [1]←→[1]←→[0]  row 0
      ↕     ↕
     [0]←→[1]←→[1]  row 1
            ↕     ↕
     [1]←→[0]←→[1]  row 2
```

**The dancing:** When we "remove" a row, we unlink its nodes but don't delete them:

```text
node.left.right = node.right
node.right.left = node.left
```

The node still remembers its neighbors. To restore during backtracking:

```text
node.left.right = node
node.right.left = node
```

The nodes "dance" in and out of the structure. This makes backtracking O(1)—no copying or rebuilding.

**Why it's fast:** For puzzles like Sudoku, the matrix is sparse and the MRV heuristic guides us to forced moves first. Combined with O(1) backtracking, DLX solves most Sudokus in microseconds.

## Complexity

- **Time:** Exponential worst case, very fast in practice for puzzles
- **Guarantees:** Finds all solutions or proves none exist

## Tips

1. **Column ordering matters.** DLX chooses columns with fewest 1s first (automatically).
2. **Start small.** Test with tiny examples (4-Queens, 3x3 grids).
3. **Secondary columns** for optional constraints that can be covered 0 or 1 times.

## See Also

- [Model (CP)](cp.md) - More general constraint programming
- [Cookbook: Pentomino](../../cookbook/pentomino.md) - Tiling example
- [Cookbook: Sudoku](../../cookbook/sudoku.md) - Can also use exact cover
- [Wikipedia: Dancing Links](https://en.wikipedia.org/wiki/Dancing_Links) - Knuth's Algorithm X

# Exact Cover

When your problem is "pick a subset of sets that covers everything exactly once," you've got an exact cover problem. Sudoku, pentomino tiling, N-Queens - these are all exact cover in disguise. Dancing Links (DLX) is Knuth's beautiful algorithm for solving them.

## Solvers in this Category

### [solve_exact_cover](solve_exact_cover)
Dancing Links (Algorithm X). Solves exact cover using linked list "dancing" - nodes remove themselves from the matrix and restore themselves during backtracking. Pure Python, no numpy required.

**Complexity:** Exponential worst case, but very fast in practice for puzzle-sized problems
**Guarantees:** Finds all solutions or proves none exist

## When to Use This Category

**Perfect for:**
- Sudoku and variants (Killer Sudoku, X-Sudoku, etc.)
- N-Queens placement
- Pentomino/polyomino tiling puzzles
- Logic grid puzzles
- Scheduling where every constraint must be satisfied exactly once
- Set covering where overlaps are forbidden

**The magic:** DLX is astonishingly fast for its exponential complexity. The "dancing" technique (covering/uncovering columns) provides efficient backtracking with minimal overhead.

## When NOT to Use This Category

- **Optimization required** → Use MILP, CP-SAT, or SAT
- **Approximate solutions acceptable** → Use metaheuristics
- **Constraints can be partially satisfied** → Use CP-SAT or constraint programming
- **Weighted set cover** (minimize cost) → Use MILP or greedy approximation

## Quick Example

```python
from solvor import solve_exact_cover

# Tiling a 2x3 board with dominoes
# Board: [A][B][C]
#        [D][E][F]
# Columns = cells (A-F), Rows = domino placements

matrix = [
    [1, 1, 0, 0, 0, 0],  # Row 0: domino covers A, B
    [0, 1, 1, 0, 0, 0],  # Row 1: domino covers B, C
    [0, 0, 0, 1, 1, 0],  # Row 2: domino covers D, E
    [0, 0, 0, 0, 1, 1],  # Row 3: domino covers E, F
    [1, 0, 0, 1, 0, 0],  # Row 4: domino covers A, D
    [0, 1, 0, 0, 1, 0],  # Row 5: domino covers B, E
    [0, 0, 1, 0, 0, 1],  # Row 6: domino covers C, F
]

result = solve_exact_cover(matrix)
print(result.solution)  # (4, 5, 6) or (0, 3, 6) or (1, 2, 4)
# Meaning: Use rows 4, 5, 6 -> dominoes at AD, BE, CF

# Find all solutions
result = solve_exact_cover(matrix, find_all=True)
print(result.solution)  # List of all valid tilings
print(result.objective)  # Number of solutions found
```

## Understanding Exact Cover

**Exact Cover Problem:** Given a matrix of 0s and 1s, find a subset of rows such that each column has exactly one 1.

**Example:**
```
    Col0 Col1 Col2 Col3
Row0  1    0    1    0
Row1  0    1    1    0
Row2  1    1    0    1
Row3  0    0    0    1

Solution: Rows 0 and 1
  Row0: 1, 0, 1, 0
  Row1: 0, 1, 1, 0
  Sum:  1, 1, 2, 0  <- Column 2 covered twice! Not exact cover.

Solution: Rows 0 and 3
  Row0: 1, 0, 1, 0
  Row3: 0, 0, 0, 1
  Sum:  1, 0, 1, 1  <- Columns 0,2,3 covered, column 1 not covered!

No exact cover exists for this matrix.
```

## How to Model Problems as Exact Cover

### General Template
1. **Identify what needs to be covered** - These become columns
2. **Identify possible choices** - These become rows
3. **Mark which constraints each choice satisfies** - These become 1s in the matrix

### Example: Sudoku

**Columns (constraints):**
- Row-number constraints: "Row i contains number n"
- Column-number constraints: "Column j contains number n"
- Box-number constraints: "Box k contains number n"
- Cell constraints: "Cell (i,j) is filled"

**Rows (choices):**
- Each possible placement: "Place number n at position (i,j)"

**Matrix:**
- Row for placement (i,j,n) has 1s in:
  - Column for "row i contains n"
  - Column for "column j contains n"
  - Column for "box k contains n" (where (i,j) is in box k)
  - Column for "cell (i,j) is filled"

### Example: N-Queens

**Columns (constraints):**
- One queen per row: "Row i has a queen"
- One queen per column: "Column j has a queen"
- At most one queen per diagonal (optional, for stricter constraint)

**Rows (choices):**
- Each possible placement: "Place queen at (i,j)"

**Matrix:**
- Row for placement (i,j) has 1s in:
  - Column for "row i has queen"
  - Column for "column j has queen"

## Tips & Tricks

### Modeling
- **Start with constraints** - What must be satisfied exactly once?
- **Each choice becomes a row** - What are the possible actions?
- **1s show which constraints a choice satisfies** - Connect choices to constraints

### Performance
- **Column ordering matters** - Choose columns with fewest 1s first (DLX does this automatically)
- **Reduce matrix size** - Eliminate dominated rows/columns before solving
- **Use find_all judiciously** - Enumerating all solutions can take exponential time

### Optional Constraints
DLX supports "secondary columns" that can be covered zero or one times (not required). This library's `solve_exact_cover` doesn't expose this directly, but you can:
- Omit optional constraints from the matrix
- Check them after finding solutions

### Debugging
- **Start small** - Test with tiny examples (3x3 Sudoku, 4-Queens)
- **Visualize the matrix** - Print it out to check if modeling is correct
- **Check infeasibility** - If no solution, verify constraints aren't contradictory

## How Dancing Links Works (Intuition)

**The "Dancing" Metaphor:**

Imagine a doubly-linked list where each node knows its neighbors. To "remove" a node:
```python
node.left.right = node.right
node.right.left = node.left
```

To "restore" (undo the removal):
```python
node.left.right = node
node.right.left = node
```

The node still knows its neighbors, so restoration is O(1)!

**Algorithm X with DLX:**
1. **Choose a column** (pick the one with fewest 1s)
2. **Try each row with a 1 in that column:**
   - "Cover" that row and all columns it satisfies
   - Recursively solve the reduced problem
   - If solution found, we're done!
   - Otherwise, "uncover" and try next row
3. **Backtrack** if no row works

**Why it's fast:** The linked list structure makes covering/uncovering O(number of 1s), and the heuristic (choose column with fewest 1s) prunes the search tree aggressively.

## Real-World Applications

### Puzzles
- **Sudoku:** The classic example
- **N-Queens:** Place N queens on NxN board, no conflicts
- **Pentomino Tiling:** Fit 12 pentomino pieces into a 6x10 rectangle
- **Polyomino Puzzles:** Tetris-style piece placement

### Scheduling
- **Shift Assignment:** Every shift covered exactly once
- **Course Scheduling:** Every course scheduled exactly once, no conflicts

### Combinatorics
- **Set Packing:** Choose disjoint sets that cover everything
- **Graph Coloring:** (Can be modeled as exact cover with some tricks)

## Exact Cover vs Other Approaches

**Why not just use SAT or CP?**

You can! SAT and CP are more general. DLX is specialized for exact cover and exploits its structure:
- **DLX is faster** for pure exact cover problems
- **SAT/CP are more flexible** for additional constraints, optimization, etc.

**When to use each:**
- **Pure exact cover** (Sudoku, tiling) → DLX
- **Exact cover + optimization** → SAT with optimization or MILP
- **Exact cover + complex constraints** → CP-SAT

## Common Gotchas

1. **Modeling errors** - Easiest mistake is getting the matrix wrong. Test on tiny examples first.
2. **Infeasibility** - If `result.status == Status.INFEASIBLE`, your constraints may be contradictory.
3. **Explosion of solutions** - Some problems have exponentially many solutions. Use `max_solutions` to limit.
4. **Performance on large matrices** - DLX is fast, but exponential worst case. Past ~1000 columns, it slows down.

## Finding All Solutions vs One Solution

```python
# Find one solution (fast)
result = solve_exact_cover(matrix)
if result.solution:
    print(f"Found solution: {result.solution}")

# Find all solutions (can be slow!)
result = solve_exact_cover(matrix, find_all=True)
print(f"Found {result.objective} solutions")
for sol in result.solution:
    print(sol)

# Find up to 100 solutions (bounded)
result = solve_exact_cover(matrix, find_all=True, max_solutions=100)
```

## See Also

- [Constraint Programming](Constraint-Programming) - More general constraint satisfaction
- [Linear & Integer Programming](Linear-&-Integer-Programming) - Optimization instead of satisfaction
- [Cookbook: Sudoku](Cookbook-Sudoku) - Full Sudoku solver using DLX
- [Cookbook: N-Queens](Cookbook-N-Queens) - N-Queens using CP (alternative approach)

# Model (Constraint Programming)

Constraint programming with integer variables. Write natural constraints like `all_different([x, y, z])` or `x + y == 10`, and the model encodes them to SAT clauses internally. You get the expressiveness of constraint programming with the power of SAT-based solving.

For more background, see [Constraint Programming](https://en.wikipedia.org/wiki/Constraint_programming) on Wikipedia.

## When to Use

- Logic puzzles (Sudoku, N-Queens, Kakuro)
- Scheduling with "all different" or complex rules
- Configuration (assembling compatible components)
- Nurse rostering, timetabling
- Anything with implication chains

## Example

```python
from solvor import Model

# Solve: x, y, z in {1..9}, all different, sum = 15
m = Model()
x = m.int_var(1, 9, 'x')
y = m.int_var(1, 9, 'y')
z = m.int_var(1, 9, 'z')

m.add(m.all_different([x, y, z]))
m.add(x + y + z == 15)

result = m.solve()
print(result.solution)  # {'x': 3, 'y': 5, 'z': 7} or similar
```

## API

### Creating Variables

```python
x = m.int_var(1, 9, 'x')      # Integer in [1, 9]
b = m.int_var(0, 1, 'b')      # Boolean (0 or 1)
```

### Adding Constraints

```python
m.add(x + y == 10)             # Arithmetic equality
m.add(x != y)                  # Inequality
m.add(m.all_different([x, y, z]))  # Global constraint
m.add(m.sum_eq([x, y, z], 15)) # Sum constraint
m.add(m.sum_le([x, y], 10))    # Sum upper bound
m.add(m.sum_ge([x, y], 5))     # Sum lower bound
```

### Global Constraints

For scheduling and routing problems:

```python
# Hamiltonian circuit (TSP successor formulation)
m.add(m.circuit([next_0, next_1, next_2]))

# Non-overlapping intervals
starts = [m.int_var(0, 10, f's{i}') for i in range(3)]
durations = [2, 3, 1]
m.add(m.no_overlap(starts, durations))

# Cumulative resource constraint
demands = [1, 2, 1]
capacity = 2
m.add(m.cumulative(starts, durations, demands, capacity))
```

### Arithmetic Expressions

```python
m.add(x + y == 10)            # Addition
m.add(x - y != 0)             # Subtraction
m.add(x * 3 + y == 15)        # Multiplication by constant
m.add(slot * n_rooms + room == combined)  # Combined index pattern
```

### Solving

```python
result = m.solve()            # Auto-selects best solver
if result.solution:
    print(result.solution['x'])
```

### Solver Selection

The model automatically picks the best solver for your constraints:

- **DFS (default for simple constraints):** Fast backtracking with constraint propagation. Used for `all_different`, `==`, `!=`, and arithmetic expressions.
- **SAT (for global constraints):** SAT encoding with DPLL solving. Used for `circuit`, `no_overlap`, `cumulative`, and `sum_*` constraints.

You can force a specific solver:

```python
result = m.solve(solver="auto")  # Default: picks best solver
result = m.solve(solver="dfs")   # Force DFS backtracking
result = m.solve(solver="sat")   # Force SAT encoding
```

### Hints and Multiple Solutions

```python
# Guide the solver toward specific values
result = m.solve(hints={'x': 5})

# Find up to 10 solutions
result = m.solve(solution_limit=10)
for sol in result.solutions:
    print(sol)
```

## Sudoku Example

```python
from solvor import Model

def solve_sudoku(puzzle):
    m = Model()
    grid = [[m.int_var(1, 9, f'c{i}{j}') for j in range(9)] for i in range(9)]

    # Row constraints
    for row in grid:
        m.add(m.all_different(row))

    # Column constraints
    for j in range(9):
        m.add(m.all_different([grid[i][j] for i in range(9)]))

    # Box constraints
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

## Complexity

- **Time:** NP-hard (constraint satisfaction is NP-complete in general)
- **Guarantees:** Finds a solution or proves none exists

## Tips

1. **Model naturally first.** Don't prematurely optimize constraints. Get it working, then refine.
2. **All-different is powerful.** Use it rather than pairwise inequalities. See [All-different constraint](https://en.wikipedia.org/wiki/All_different_constraint).
3. **Symmetry breaking.** Add constraints to eliminate symmetric solutions.

## See Also

- [solve_sat](solve-sat.md) - Raw SAT solving
- [solve_exact_cover](solve-exact-cover.md) - Dancing Links for exact cover problems
- [Cookbook: Sudoku](../../cookbook/sudoku.md) - Full Sudoku solver
- [Cookbook: N-Queens](../../cookbook/n-queens.md) - Classic constraint programming problem

# Model (CP-SAT)

Constraint programming with integer variables. Write natural constraints like `all_different([x, y, z])` or `x + y == 10`, and the model encodes them to SAT clauses. You get the expressiveness of CP with the power of modern SAT solvers.

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
b = m.bool_var('b')            # Boolean
```

### Adding Constraints

```python
m.add(x + y == 10)             # Arithmetic
m.add(x < y)                   # Comparison
m.add(m.all_different([x, y, z]))  # Global constraint
m.add(m.sum_eq([x, y, z], 15)) # Sum constraint
```

### Solving

```python
result = m.solve()
if result.solution:
    print(result.solution['x'])
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

- **Time:** NP-hard
- **Guarantees:** Finds a solution or proves none exists

## Tips

1. **Model naturally first.** Don't prematurely optimize constraints. Get it working, then refine.
2. **All-different is powerful.** Use it rather than pairwise inequalities.
3. **Symmetry breaking.** Add constraints to eliminate symmetric solutions.

## See Also

- [solve_sat](solve-sat.md) - Raw SAT solving
- [Cookbook: Sudoku](../../cookbook/sudoku.md) - Full Sudoku solver
- [Cookbook: N-Queens](../../cookbook/n-queens.md) - Classic CP problem

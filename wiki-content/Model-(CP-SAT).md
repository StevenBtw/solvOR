# Model (CP-SAT)

**Constraint programming with integer variables, natural modeling compiled to SAT.**

## Summary

- **Category:** [Constraint Programming](Constraint-Programming)
- **Problem Type:** Constraint satisfaction with integer variables
- **Complexity:** NP-hard
- **Guarantees:** Finds satisfying assignment or proves unsatisfiability
- **Status:** Returns solution, `INFEASIBLE`, or `MAX_ITER`

## When to Use This

Perfect for problems where:
- You have integer variables (not just booleans)
- Constraints are complex logical rules (`all_different`, implications, sums)
- You want natural problem modeling without manual encoding
- You need satisfaction, not optimization (or optimization is secondary)

**Real examples:**
- **Sudoku:** 9×9 grid, each row/column/box has all_different(1-9)
- **N-Queens:** Place N queens on a board with no conflicts
- **Scheduling:** Assign tasks to time slots with complex rules
- **Configuration:** Build a valid system from compatible components

## When NOT to Use This

- **Need to optimize a clear linear objective** → Use [`solve_milp`](solve_milp) for better performance
- **Pure boolean (true/false) only** → Use [`solve_sat`](solve_sat) directly
- **All variables are continuous** → Use [`solve_lp`](solve_lp) or gradient methods
- **Exact cover structure** → Use [`solve_exact_cover`](solve_exact_cover)

## Quick Example

```python
from solvor import Model

# Simple constraint satisfaction
m = Model()
x = m.int_var(1, 9, 'x')
y = m.int_var(1, 9, 'y')
z = m.int_var(1, 9, 'z')

m.add(m.all_different([x, y, z]))
m.add(x + y + z == 15)

result = m.solve()
print(result.solution)  # {'x': 3, 'y': 5, 'z': 7} or similar
```

## How It Works

**CP-SAT compiles to SAT:**
1. **Integer variables** are encoded as collections of boolean variables
   - `x ∈ {1,2,3}` becomes three booleans: `x_is_1`, `x_is_2`, `x_is_3`
2. **Constraints** are encoded as SAT clauses
   - `all_different([x,y])` becomes "x and y can't both be 1, can't both be 2, ..."
   - `x + y == 5` becomes specific combinations allowed
3. **SAT solver** finds a satisfying assignment to the boolean variables
4. **Decode** back to integer values

**Why it works:** SAT solvers are incredibly fast. By compiling high-level constraints to SAT, you get natural modeling with industrial-strength solving.

**Metaphor:** You write in Python, it compiles to machine code. You model in integers, it compiles to booleans.

## API Guide

### Creating Variables

```python
m = Model()

# Integer variable with bounds
x = m.int_var(lb=0, ub=10, name='x')  # x ∈ {0,1,2,...,10}

# Name is optional (auto-generated if omitted)
y = m.int_var(0, 5)
```

### Adding Constraints

```python
# Equality
m.add(x == 5)
m.add(x == y)

# Inequality
m.add(x != 3)
m.add(x != y)

# Arithmetic (creates intermediate sum expressions)
m.add(x + y == 10)
m.add(x + y + z == 15)

# All different
m.add(m.all_different([x, y, z]))

# Sum constraints
m.add(m.sum_eq([x, y, z], 10))   # x + y + z == 10
m.add(m.sum_le([x, y], 5))       # x + y <= 5
m.add(m.sum_ge([x, y], 3))       # x + y >= 3
```

### Solving

```python
# Solve the model
result = m.solve()

# With SAT solver parameters
result = m.solve(max_conflicts=500, max_restarts=200)
```

## Common Gotchas

### 1. Variables Need Bounds
```python
# Wrong: Unbounded variable
x = m.int_var(name='x')  # Error! Need lb and ub

# Right: Always specify bounds
x = m.int_var(0, 100, 'x')
```

### 2. Solution Contains Only Named Variables
```python
m = Model()
x = m.int_var(0, 10, 'x')
y = m.int_var(0, 10)  # No name!

result = m.solve()
print(result.solution)  # {'x': 5} - y is internal, not returned!

# To get all variables, always name them
```

### 3. Constraints Must Be Added
```python
# Wrong: Creating constraint but not adding
x == y  # This does nothing!

# Right: Add to the model
m.add(x == y)
```

### 4. No Optimization Objective
```python
# CP-SAT finds *a* solution, not the best one
# If you need to minimize/maximize, use MILP instead

# Wrong: Expecting optimization
m.add(x + y == min_value)  # No "minimize x+y" feature

# For optimization, use solve_milp with integer constraints
```

## Result Format

Returns a `Result` namedtuple:
```python
Result(
    solution={'x': 3, 'y': 5, 'z': 7},  # Named variable assignments
    objective=0,                         # Always 0 (no optimization)
    iterations=50,                       # SAT decisions
    evaluations=200,                     # SAT propagations
    status=Status.OPTIMAL                # Success or INFEASIBLE/MAX_ITER
)
```

**solution:** Dictionary mapping variable names to integer values
- Only named variables appear (auto-generated names starting with `_` are excluded)

## Example: Sudoku (Simplified 4×4)

```python
from solvor import Model

# 4×4 Sudoku: Each row, column, and 2×2 box contains 1,2,3,4
m = Model()

# Create variables: grid[i][j] ∈ {1,2,3,4}
grid = [[m.int_var(1, 4, f'cell_{i}_{j}') for j in range(4)] for i in range(4)]

# Row constraints: each row has all different
for i in range(4):
    m.add(m.all_different(grid[i]))

# Column constraints: each column has all different
for j in range(4):
    m.add(m.all_different([grid[i][j] for i in range(4)]))

# Box constraints: each 2×2 box has all different
for box_r in range(2):
    for box_c in range(2):
        cells = [grid[box_r*2 + i][box_c*2 + j] for i in range(2) for j in range(2)]
        m.add(m.all_different(cells))

# Given clues (example)
m.add(grid[0][0] == 1)
m.add(grid[1][1] == 2)

result = m.solve()
if result.solution:
    for i in range(4):
        print([result.solution[f'cell_{i}_{j}'] for j in range(4)])
```

## Supported Constraints

### Equality/Inequality
- `x == value` - Variable equals constant
- `x == y` - Variables equal
- `x != value` - Variable not equal to constant
- `x != y` - Variables not equal

### Arithmetic
- `x + y` - Creates sum expression (use with `==`)
- `x + y == value` - Sum equals constant
- `x + y + z == value` - Multi-variable sum

### All Different
- `m.all_different([x, y, z])` - All variables have different values
- Encodes to pairwise != constraints

### Sum Constraints
- `m.sum_eq(vars, target)` - Sum of vars equals target
- `m.sum_le(vars, target)` - Sum ≤ target
- `m.sum_ge(vars, target)` - Sum ≥ target

## Tips for Better Encoding

### 1. Tight Bounds
```python
# Tight bounds = fewer boolean variables = faster solving
x = m.int_var(0, 100)     # 101 boolean variables
x = m.int_var(10, 20)     # Only 11 boolean variables (if you know x ∈ [10,20])
```

### 2. Use all_different
```python
# Efficient: all_different
m.add(m.all_different([x, y, z]))

# Inefficient: Manual pairwise constraints
m.add(x != y)
m.add(x != z)
m.add(y != z)

# all_different compiles to the same thing but is clearer
```

### 3. Symmetry Breaking
```python
# If solution (x=1,y=2,z=3) is equivalent to (x=3,y=2,z=1),
# break symmetry by forcing an order:
m.add(x <= y)  # or x < y depending on problem
```

### 4. Redundant Constraints
```python
# Adding implied constraints can help the solver
# Example: if x + y == 10 and both ∈ [0,10], adding x <= 10 is redundant
# but might tighten the SAT encoding
```

## Real-World Applications

### Puzzles
- **Sudoku, Kakuro, KenKen:** Classic constraint satisfaction
- **N-Queens:** No two queens attack each other
- **Logic Grid Puzzles:** "Einstein's riddle" style

### Scheduling
- **Timetabling:** Assign classes to rooms/times with no conflicts
- **Nurse Rostering:** Assign shifts satisfying regulations
- **Exam Scheduling:** No student has two exams at once

### Configuration
- **Component Selection:** Build a system from compatible parts
- **Resource Allocation:** Assign resources with compatibility constraints

## Comparison with MILP

| Feature | CP-SAT (Model) | MILP |
|---------|----------------|------|
| **Optimization** | No (satisfaction only) | Yes (minimize/maximize) |
| **Constraint types** | Arbitrary logical | Linear inequalities |
| **Natural modeling** | `all_different`, implications | Matrix constraints |
| **Performance** | Fast for satisfaction | Fast for optimization |
| **Best for** | Puzzles, complex logic | Resource allocation, scheduling with objectives |

**Rule of thumb:** Use CP-SAT for "find any valid solution." Use MILP for "find the best solution."

## Advanced: Incremental Solving

```python
# Solve once
m = Model()
x = m.int_var(1, 10, 'x')
m.add(x > 5)
result1 = m.solve()

# Add more constraints and re-solve
m.add(x < 8)
result2 = m.solve()

# Note: Internally re-encodes and re-solves (doesn't reuse work)
```

## Related Solvers

- [`solve_sat`](solve_sat) - Raw SAT solver (boolean variables only)
- [`solve_milp`](solve_milp) - For optimization with linear objective
- [`solve_exact_cover`](solve_exact_cover) - Specialized for exact cover problems

## See Also

- [Constraint Programming](Constraint-Programming) - Category overview
- [Cookbook: Sudoku](Cookbook-Sudoku) - Full Sudoku solver
- [Cookbook: N-Queens](Cookbook-N-Queens) - N-Queens with CP

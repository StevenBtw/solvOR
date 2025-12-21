# Constraint Programming

When your problem is less "optimize this objective" and more "find me anything that satisfies these rules," you want constraint programming. Logic puzzles, scheduling with complex rules, configuration problems where the constraints themselves are the challenge.

## Solvers in this Category

### [solve_sat](solve_sat)
Boolean satisfiability. Feed it clauses in CNF (conjunctive normal form), get back a satisfying assignment. This is the engine under the hood of CP-SAT and many other solvers.

**Complexity:** NP-complete
**Guarantees:** Finds a solution or proves none exists

### [Model (CP-SAT)](Model-(CP-SAT))
Constraint programming with integer variables. Write natural constraints like `all_different([x, y, z])` or `x + y == 10`, the model encodes them to SAT clauses. You get the expressiveness of CP with the raw power of modern SAT solvers.

**Complexity:** NP-hard
**Guarantees:** Satisfaction only, no optimization guarantees

## When to Use This Category

**Perfect for:**
- Logic puzzles (Sudoku, N-Queens, Kakuro)
- Scheduling with "all different" or complex rules
- Configuration (assembling components that must be compatible)
- Nurse rostering, timetabling
- Anything with implication chains: "if A then B, if B then not C"

**The magic:** You model the problem naturally, the solver handles the search. No need to encode everything yourself.

## When NOT to Use This Category

- **Have a clear objective to optimize** → Use MILP (linear) or metaheuristics (non-linear)
- **Continuous variables dominate** → Use gradient descent or simplex
- **Exact cover structure** → Use DLX directly (it's faster than encoding to SAT)
- **Trivially small problems** → Encoding overhead might not be worth it

## Comparing Solvers

| Solver | Input | Expressiveness | Use When |
|--------|-------|----------------|----------|
| `solve_sat` | CNF clauses | Boolean logic only | You have clauses ready, or want full control |
| `Model` | High-level constraints | Integer variables + logic | Natural problem modeling, you want convenience |

`Model` is `solve_sat` with a friendly interface. Under the hood, it encodes your integer constraints into boolean SAT clauses.

## Quick Example

```python
from solvor import solve_sat, Model

# SAT: (x1 OR x2) AND (NOT x1 OR x3) AND (NOT x2 OR NOT x3)
result = solve_sat([[1, 2], [-1, 3], [-2, -3]])
print(result.solution)  # {1: True, 2: False, 3: True} or similar

# CP-SAT: Sudoku cell constraints
m = Model()
x = m.int_var(1, 9, 'x')
y = m.int_var(1, 9, 'y')
z = m.int_var(1, 9, 'z')
m.add(m.all_different([x, y, z]))
m.add(x + y + z == 15)
result = m.solve()
print(result.solution)  # {'x': 3, 'y': 5, 'z': 7} or similar
```

## Tips & Tricks

1. **Model naturally first** - Don't prematurely optimize your constraints. Get it working, then refine.

2. **All-different is powerful** - Many problems have "no two things the same" constraints. Use `all_different` rather than pairwise inequalities.

3. **Symmetry breaking** - Add constraints to eliminate symmetric solutions (e.g., queens on the same diagonal).

4. **Constraint propagation** - The solver does this automatically, but understanding it helps: each decision propagates implications, pruning the search space.

## Real-World Applications

- **Sudoku solvers:** Classic constraint satisfaction
- **N-Queens:** Place N queens on a chessboard with no conflicts
- **Course scheduling:** Assign classes to rooms/times with no conflicts
- **Nurse rostering:** Assign shifts satisfying regulations and preferences
- **Configuration problems:** Build a valid system from compatible components

## SAT vs CP-SAT Decision Tree

**Use SAT when:**
- Your problem is already boolean
- You're comfortable with CNF encoding
- You need maximum control and performance

**Use CP-SAT when:**
- You have integer variables (1-9, not just true/false)
- You want readable constraint modeling
- You value development speed over squeezing out last 10% performance

## See Also

- [Linear & Integer Programming](Linear-&-Integer-Programming) - When you need optimization, not just satisfaction
- [Exact Cover](Exact-Cover) - Specialized solver for a subset of constraint problems
- [Cookbook: Sudoku](Cookbook-Sudoku) - Full Sudoku solver
- [Cookbook: N-Queens](Cookbook-N-Queens) - Classic CP problem

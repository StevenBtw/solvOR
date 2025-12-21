# solve_sat

**Boolean satisfiability with clause learning, the engine behind modern constraint solvers.**

## Summary 

- **Category:** [Constraint Programming](Constraint-Programming)
- **Problem Type:** Boolean satisfiability (SAT)
- **Complexity:** NP-complete
- **Guarantees:** Finds satisfying assignment or proves unsatisfiability
- **Status:** Returns solution, `INFEASIBLE`, or `MAX_ITER`

## When to Use This

Perfect for problems where:
- Everything is boolean (true/false, on/off, yes/no)
- You have logical implications, exclusions, requirements
- Constraints are expressed as "OR" clauses
- You need to find *any* satisfying assignment (not optimize)

**Real examples:**
- **Logic puzzles:** Constraints expressed as boolean rules
- **Configuration:** Component compatibility (if A then not B, etc.)
- **Scheduling:** Boolean constraints on assignments
- **Circuit verification:** Does the circuit satisfy the specification?

## When NOT to Use This

- **Need to optimize an objective** → Use [`solve_milp`](solve_milp)
- **Integer variables are more natural** → Use [`Model`](Model-(CP-SAT)) which encodes integers to SAT
- **Exact cover structure** → Use [`solve_exact_cover`](solve_exact_cover) (faster than encoding to SAT)
- **Continuous variables** → Use [`solve_lp`](solve_lp) or [`gradient_descent`](gradient_descent)

## Quick Example

```python
from solvor import solve_sat

# (x1 OR x2) AND (NOT x1 OR x3) AND (NOT x2 OR NOT x3)
clauses = [
    [1, 2],      # x1 OR x2
    [-1, 3],     # NOT x1 OR x3
    [-2, -3]     # NOT x2 OR NOT x3
]

result = solve_sat(clauses)
print(result.solution)  # {1: True, 2: False, 3: True} or similar
```

**Encoding:**
- Positive integer `k` means "variable k is true"
- Negative integer `-k` means "variable k is false"
- Each clause is a list of literals (OR'd together)
- All clauses must be satisfied (AND'd together)

## How It Works

**DPLL with Clause Learning (CDCL):**
1. **Unit Propagation:** If a clause has only one unassigned literal, assign it
2. **Decision:** Pick an unassigned variable, try assigning it true
3. **Propagate:** Apply unit propagation recursively
4. **Conflict?**
   - Yes → Analyze conflict, learn a new clause, backtrack
   - No → Continue with next decision
5. **Repeat** until all variables assigned (SAT) or conflict at level 0 (UNSAT)

**Why it works:** Clause learning prevents re-exploring the same dead-ends. Modern SAT solvers are astonishingly fast, solving million-variable instances routinely.

**Metaphor:** Exploring a maze, but marking dead-ends with "don't go this way again" signs that apply globally, not just locally.

## Parameters Guide

```python
solve_sat(
    clauses,                    # List of clauses in CNF
    *,
    assumptions=None,           # Additional literals to assume true
    max_conflicts=100,          # Conflicts before restart
    max_restarts=100            # Number of restarts
)
```

**clauses:** List of lists, each inner list is a clause
- `[[1, 2], [-1, 3]]` means `(x1 OR x2) AND (NOT x1 OR x3)`

**assumptions:** List of literals to assume true (useful for incremental solving)
- `assumptions=[1, -3]` forces `x1=True, x3=False`

**max_conflicts / max_restarts:** Tuning parameters. Higher values = more thorough search.

## Common Gotchas

### 1. Variable Numbering Starts at 1
```python
# Wrong: Using variable 0
clauses = [[0, 1]]  # Variable 0 doesn't exist!

# Right: Variables numbered 1, 2, 3, ...
clauses = [[1, 2]]
```

### 2. Empty Clause Means Unsatisfiable
```python
# If you have an empty clause [], the formula is unsatisfiable
clauses = [[], [1, 2]]  # Unsatisfiable (empty clause is always false)
```

### 3. Solution is a Dictionary
```python
result = solve_sat([[1, 2]])
# result.solution is {1: True, 2: False} (or similar)
# NOT a list like [True, False]

# Access specific variable:
if result.solution[1]:
    print("Variable 1 is true")
```

### 4. CNF is Required
```python
# SAT solvers need Conjunctive Normal Form (AND of ORs)
# If you have:  (x1 AND x2) OR x3
# Convert to CNF:  (x1 OR x3) AND (x2 OR x3)

# Wrong formula type:
formula = "(x1 AND x2) OR x3"  # Not CNF!

# Right: Convert to CNF first
clauses = [[1, 3], [2, 3]]
```

## Result Format

Returns a `Result` namedtuple:
```python
Result(
    solution={1: True, 2: False, ...},  # Variable assignments
    objective=len(solution),             # Number of variables assigned
    iterations=50,                       # Decisions made
    evaluations=200,                     # Propagations performed
    status=Status.OPTIMAL                # Success or INFEASIBLE/MAX_ITER
)
```

**solution:** Dictionary mapping variable number to boolean value
- `{1: True, 2: False, 3: True}` means x₁=T, x₂=F, x₃=T

**status:**
- Success (no explicit `OPTIMAL` status, just a solution exists)
- `INFEASIBLE`: No satisfying assignment exists (proven unsatisfiable)
- `MAX_ITER`: Hit search limits (increase `max_conflicts` or `max_restarts`)

## Converting Problems to CNF

### Implications
- `x → y` ("if x then y") is equivalent to `NOT x OR y`
- Clause: `[-x, y]`

### Equivalence
- `x ↔ y` ("x if and only if y") is `(x → y) AND (y → x)`
- Clauses: `[-x, y], [x, -y]`

### AND to CNF
- `x = (y AND z)` means:
  - If x then y: `[-x, y]`
  - If x then z: `[-x, z]`
  - If y and z then x: `[-y, -z, x]`

### OR is Already CNF
- `x = (y OR z)` means: `[-x, y, z]` (if x then y or z), and more...

### NOT
- `x = NOT y` is: `[-x, -y], [x, y]`

### Tseitin Transformation
For complex formulas, introduce auxiliary variables:
- `(x1 AND x2) OR (x3 AND x4)` → Introduce `a = x1 AND x2`, `b = x3 AND x4`, then `a OR b`

## Example: 3-Coloring

```python
from solvor import solve_sat

# Color a triangle graph with 3 colors (Red, Green, Blue)
# Nodes: 0, 1, 2
# Edges: (0,1), (1,2), (0,2)

# Variables: node_i_color_j (true if node i has color j)
# Numbering: node 0 colors [1,2,3], node 1 colors [4,5,6], node 2 colors [7,8,9]

clauses = []

# Each node has at least one color
for node in range(3):
    base = node * 3 + 1
    clauses.append([base, base+1, base+2])  # Red OR Green OR Blue

# Each node has at most one color (pairwise exclusions)
for node in range(3):
    base = node * 3 + 1
    clauses.append([-base, -(base+1)])      # NOT (Red AND Green)
    clauses.append([-base, -(base+2)])      # NOT (Red AND Blue)
    clauses.append([-(base+1), -(base+2)])  # NOT (Green AND Blue)

# Adjacent nodes have different colors
edges = [(0,1), (1,2), (0,2)]
for u, v in edges:
    for color in range(3):
        u_var = u * 3 + 1 + color
        v_var = v * 3 + 1 + color
        clauses.append([-u_var, -v_var])  # NOT (both have same color)

result = solve_sat(clauses)
if result.solution:
    for node in range(3):
        base = node * 3 + 1
        color = ['Red', 'Green', 'Blue'][[result.solution[base+c] for c in range(3)].index(True)]
        print(f"Node {node}: {color}")
```

## Tips for Better SAT Encoding

### 1. At-Most-One Constraints
For "at most one of these is true," use pairwise exclusions:
```python
# At most one of x1, x2, x3
clauses = [[-1, -2], [-1, -3], [-2, -3]]
```

For larger sets, consider "ladder" or "commander" encodings (more efficient but complex).

### 2. Exactly-One Constraints
"Exactly one" = "at least one" AND "at most one":
```python
# Exactly one of x1, x2, x3
clauses = [
    [1, 2, 3],           # At least one
    [-1, -2], [-1, -3], [-2, -3]  # At most one
]
```

### 3. Symmetry Breaking
Add constraints to eliminate symmetric solutions:
```python
# If all solutions are equivalent under permutation,
# force an ordering: x1 < x2 < x3 (encoded appropriately)
```

### 4. Polarity Heuristics
The solver's initial variable assignments matter. If you know certain variables are likely true/false, using assumptions can help.

## Real-World Applications

- **Hardware Verification:** Check if circuit design satisfies specification
- **Software Verification:** Prove program properties
- **Planning & Scheduling:** Find feasible schedules under boolean constraints
- **Cryptanalysis:** Break ciphers by encoding as SAT
- **Bioinformatics:** Haplotype inference, protein folding constraints

## Related Solvers

- [`Model`](Model-(CP-SAT)) - High-level CP that compiles to SAT (easier to use)
- [`solve_exact_cover`](solve_exact_cover) - Specialized for exact cover (faster than SAT encoding)
- [`solve_milp`](solve_milp) - For optimization with integer variables

## Advanced: Assumptions and Incremental Solving

```python
# Solve with assumptions (incrementally)
base_clauses = [[1, 2], [-1, 3]]

# First solve: assume x1 is true
result1 = solve_sat(base_clauses, assumptions=[1])

# Second solve: assume x1 is false
result2 = solve_sat(base_clauses, assumptions=[-1])

# Useful for: exploring different scenarios, Max-SAT, optimization
```

## See Also

- [Constraint Programming](Constraint-Programming) - Category overview
- [`Model`](Model-(CP-SAT)) - Higher-level CP interface
- [Cookbook: Graph Coloring](Cookbook-Graph-Coloring) - SAT for graph coloring

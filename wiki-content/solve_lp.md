# solve_lp

**Simplex solver for linear programming - walks along crystal edges to find optimal corners.**

## At a Glance

- **Category:** [Linear & Integer Programming](Linear-&-Integer-Programming)
- **Problem Type:** Linear optimization with continuous variables
- **Complexity:** O(exponential worst case, polynomial average case)
- **Guarantees:** Exact optimal solution for LP
- **Status:** Returns `OPTIMAL`, `INFEASIBLE`, `UNBOUNDED`, or `MAX_ITER`

## When to Use This

Perfect for problems where:
- Your objective is a linear function: `c₁x₁ + c₂x₂ + ... + cₙxₙ`
- All constraints are linear inequalities: `a₁x₁ + a₂x₂ + ... ≤ b`
- Variables can be fractional (no integer constraints)
- You need the *exact* optimum, not an approximation

**Real examples:**
- **Diet problem:** Minimize cost of food while meeting nutritional requirements
- **Blending:** Mix ingredients to meet specifications at minimum cost
- **Resource allocation:** Distribute continuous resources (time, budget, materials)
- **Production planning:** Decide how much of each product to make (when fractional amounts make sense)

## When NOT to Use This

- **Some variables must be integers** → Use [`solve_milp`](solve_milp) instead
- **Non-linear objective or constraints** → Use [`gradient_descent`](gradient_descent), [`anneal`](anneal), or [`evolve`](evolve)
- **No clear objective function** → Use constraint programming ([`Model`](Model-(CP-SAT)))
- **Boolean logic constraints** → Use [`solve_sat`](solve_sat) or [`Model`](Model-(CP-SAT))

## Quick Example

```python
from solvor import solve_lp

# Maximize 3x + 2y subject to:
#   x + y <= 4
#   x, y >= 0

# Convert to standard form (minimize -objective)
result = solve_lp(
    c=[-3, -2],      # Coefficients (negated for maximization)
    A=[[1, 1]],      # Constraint matrix
    b=[4],           # Right-hand side
    minimize=False   # Maximize instead
)

print(result.solution)   # (4.0, 0.0)
print(result.objective)  # 12.0
print(result.status)     # Status.OPTIMAL
```

## How It Works

Simplex visualizes your problem as a multidimensional crystal (polytope) where:
- Each vertex (corner) is a candidate solution
- Edges connect adjacent vertices
- The optimal solution is always at a vertex

**The algorithm:**
1. Start at an arbitrary vertex (found via Phase 1)
2. Look at neighboring vertices along edges
3. Move to a neighbor that improves the objective
4. Repeat until no improving neighbor exists (you're optimal!)

**Why it works:** Linear functions on polytopes achieve their optimum at vertices. Simplex walks from vertex to vertex,always uphill, until it reaches the peak.

**Metaphor:** You're walking along the edges of a giant crystal, always going uphill. Eventually you hit a corner where every direction is downhill - that's the optimum.

## Parameters Guide

```python
solve_lp(
    c,              # Objective coefficients [c₁, c₂, ..., cₙ]
    A,              # Constraint matrix [[a₁₁, a₁₂, ...], [...]]
    b,              # Right-hand side [b₁, b₂, ..., bₘ]
    *,
    minimize=True,  # True to minimize, False to maximize
    eps=1e-10,      # Numerical tolerance for zero
    max_iter=100_000 # Maximum iterations
)
```

**Problem form:** Minimize `c @ x` subject to `A @ x <= b`, `x >= 0`

- **c:** Cost coefficients for each variable
- **A:** Each row is one constraint, each column is one variable
- **b:** Right-hand side values for each constraint
- **minimize:** Set to `False` to maximize (negates objective internally)

**Important:** All variables are assumed `>= 0`. For unbounded variables, split into positive and negative parts: `x = x⁺ - x⁻` where `x⁺, x⁻ >= 0`.

## Common Gotchas

### 1. Forgetting Non-Negativity Constraints
```python
# Wrong: Variables can be negative
result = solve_lp(c=[1, 1], A=[[1, 1]], b=[10])

# Right: x >= 0 is automatic. For unbounded variables, split them.
# If you need x1 >= -5, add constraint: -x1 <= 5
```

### 2. Maximization vs Minimization
```python
# Wrong: Trying to maximize but passing positive coefficients
result = solve_lp(c=[3, 2], A=[[1,1]], b=[4])  # This minimizes!

# Right: Use minimize=False for maximization
result = solve_lp(c=[-3, -2], A=[[1,1]], b=[4], minimize=False)
# OR negate coefficients manually and minimize
```

### 3. Infeasibility vs Unboundedness
```python
# Infeasible: No solution satisfies all constraints
result = solve_lp(c=[1], A=[[1], [-1]], b=[5, -10])
# status == Status.INFEASIBLE

# Unbounded: Objective can improve infinitely
result = solve_lp(c=[-1], A=[[1]], b=[10], minimize=False)
# status == Status.UNBOUNDED (maximize -x with x <= 10? Go to infinity!)
```

### 4. Numerical Issues with Scaling
```python
# Bad: Huge range in coefficients
result = solve_lp(c=[1e-8, 1e8], A=[[1e-10, 1e10]], b=[1])

# Better: Scale constraints to similar magnitudes
# Divide constraint by 1e10: [1e-20, 1] @ x <= 1e-10
```

## Result Format

Returns a `Result` namedtuple:
```python
Result(
    solution=(x₁, x₂, ..., xₙ),  # Optimal values for variables
    objective=12.0,                # Optimal objective value
    iterations=5,                  # Simplex iterations
    evaluations=5,                 # Same as iterations for LP
    status=Status.OPTIMAL          # OPTIMAL, INFEASIBLE, UNBOUNDED, MAX_ITER
)
```

**Status values:**
- `OPTIMAL`: Found the exact optimum
- `INFEASIBLE`: No solution satisfies all constraints
- `UNBOUNDED`: Objective can improve infinitely
- `MAX_ITER`: Hit iteration limit (increase `max_iter`)

## Related Solvers

- [`solve_milp`](solve_milp) - LP with integer constraints (uses this solver internally)
- [`gradient_descent`](gradient_descent) - For non-linear continuous optimization
- [`network_simplex`](network_simplex) - Specialized simplex for flow networks
- [`Model`](Model-(CP-SAT)) - For boolean/integer constraints without clear objective

## Advanced: Two-Phase Simplex

Simplex needs a starting feasible solution (vertex). How do you find one if constraints are complex?

**Phase 1:** Solve an auxiliary LP with artificial variables to find an initial feasible solution.
**Phase 2:** Use that solution to start the real LP solve.

This implementation does both phases automatically. If Phase 1 fails to find a feasible solution, returns `INFEASIBLE`.

## See Also

- [Linear & Integer Programming](Linear-&-Integer-Programming) - Category overview
- [Cookbook: Resource Allocation](Cookbook-Resource-Allocation) - Full LP example

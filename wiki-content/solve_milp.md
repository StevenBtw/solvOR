# solve_milp

**Mixed-Integer Linear Programming - LP with integer constraints via branch and bound.**

## At a Glance

- **Category:** [Linear & Integer Programming](Linear-&-Integer-Programming)
- **Problem Type:** Linear optimization with integer variables
- **Complexity:** NP-hard (exponential worst case)
- **Guarantees:** Optimal integer solution (or prove none exists)
- **Status:** Returns `OPTIMAL`, `FEASIBLE`, `INFEASIBLE`, or `UNBOUNDED`

## When to Use This

Perfect for problems where:
- Your objective and constraints are linear
- Some variables must be integers (counts, binary decisions)
- You need provably optimal integer solutions
- The problem size is manageable (hundreds of variables, not millions)

**Real examples:**
- **Scheduling:** Assign workers to discrete time slots
- **Set covering:** Select facilities to cover all customers
- **Knapsack:** Choose items to pack (0/1 decisions)
- **Production planning:** Make discrete quantities of products
- **Network design:** Select which connections to build (binary)

## When NOT to Use This

- **All variables can be fractional** → Use [`solve_lp`](solve_lp) (much faster)
- **Boolean logic is more natural** → Use [`solve_sat`](solve_sat) or [`Model`](Model-(CP-SAT))
- **Non-linear objective/constraints** → Use metaheuristics ([`anneal`](anneal), [`evolve`](evolve))
- **Huge scale (10,000+ variables)** → Consider heuristics or specialized MILP solvers

## Quick Example

```python
from solvor import solve_milp

# Knapsack: Maximize value subject to weight constraint
# Items: (value=3, weight=2), (value=5, weight=3), (value=6, weight=5)
# Capacity: 7

result = solve_milp(
    c=[-3, -5, -6],           # Objective (negated for max)
    A=[[2, 3, 5]],            # Weight constraint
    b=[7],                    # Capacity
    integers=[0, 1, 2],       # All variables are integers
    minimize=False
)

print(result.solution)   # (1, 1, 0) -> take items 0 and 1
print(result.objective)  # 8.0 (value = 3 + 5)
```

## How It Works

**Branch and Bound:**
1. **Solve LP relaxation** (ignore integer constraints)
2. **If solution is already integer:** Done! It's optimal.
3. **Otherwise:** Pick a fractional variable (e.g., x₁ = 2.7)
4. **Branch:** Create two subproblems:
   - Left: x₁ ≤ 2 (floor)
   - Right: x₁ ≥ 3 (ceiling)
5. **Bound:** Solve LP relaxation for each subproblem
6. **Prune:** If a subproblem's LP bound is worse than the best integer solution found, discard it
7. **Repeat** until all branches explored or pruned

**Why it works:** The LP relaxation provides a bound. If relaxation is worse than current best integer solution, that whole branch can't contain the optimum.

**Metaphor:** Exploring a tree of possibilities, but intelligently pruning branches that can't possibly lead to better solutions.

## Parameters Guide

```python
solve_milp(
    c,                   # Objective coefficients [c₁, c₂, ..., cₙ]
    A,                   # Constraint matrix [[a₁₁, a₁₂, ...], [...]]
    b,                   # Right-hand side [b₁, b₂, ..., bₘ]
    integers,            # List of variable indices that must be integer
    *,
    minimize=True,       # True to minimize, False to maximize
    eps=1e-6,           # Tolerance for integrality and optimality
    max_iter=10_000,    # Max LP iterations per subproblem
    max_nodes=100_000,  # Max branch-and-bound nodes to explore
    gap_tol=1e-6        # Optimality gap tolerance
)
```

**Key parameters:**
- **integers:** List of indices for integer variables (0-indexed). E.g., `[0, 2]` means x₀ and x₂ must be integer.
- **max_nodes:** Limits the search tree size. Increase for harder problems.
- **gap_tol:** Relative optimality gap. E.g., 0.01 = stop when within 1% of optimal.

## Common Gotchas

### 1. Specifying Integer Variables
```python
# Wrong: Forgetting to specify which variables are integers
result = solve_milp(c=[1, 2], A=[[1, 1]], b=[10])  # Missing integers!

# Right: Explicitly list integer variable indices
result = solve_milp(c=[1, 2], A=[[1, 1]], b=[10], integers=[0, 1])
```

### 2. Binary Variables Need Explicit Bounds
```python
# For 0/1 binary variables, add upper bound constraints
# If x₀ is binary:
A = [[1, 0],    # x₀ <= 1
     [0, 1]]    # x₁ <= 1
b = [1, 1]
result = solve_milp(c=[...], A=A, b=b, integers=[0, 1])
```

### 3. Optimality Gap
```python
# Status.FEASIBLE means found a solution but not proven optimal
# (hit max_nodes or gap_tol)
if result.status == Status.FEASIBLE:
    print("Found good solution, but might not be optimal")
elif result.status == Status.OPTIMAL:
    print("Provably optimal!")
```

### 4. Scaling and Performance
```python
# MILP is NP-hard. For large problems:
# - Tighten LP relaxation with redundant constraints
# - Use smaller max_nodes and accept feasible solutions
# - Consider gap_tol > 0 (e.g., 0.01 for 1% from optimal)
```

## Result Format

Returns a `Result` namedtuple:
```python
Result(
    solution=(x₁, x₂, ..., xₙ),  # Optimal integer values
    objective=12.0,                # Optimal objective value
    iterations=50,                 # Number of branch-and-bound nodes explored
    evaluations=500,               # Total LP iterations across all nodes
    status=Status.OPTIMAL          # OPTIMAL, FEASIBLE, INFEASIBLE, UNBOUNDED
)
```

**Status values:**
- `OPTIMAL`: Provably optimal integer solution
- `FEASIBLE`: Found integer solution but not proven optimal (hit limits)
- `INFEASIBLE`: No integer solution exists
- `UNBOUNDED`: Objective unbounded (rare with integer variables)

## Tips for Faster MILP

### 1. Tight LP Relaxation
Add redundant constraints that don't change the integer feasible region but tighten the LP relaxation:
```python
# If x₁ + x₂ <= 10 and both are binary, also add:
# x₁ + x₂ <= 2  (tighter than 10!)
```

### 2. Good Branching Variables
The algorithm picks "most fractional" variables to branch on. This usually works well, but problem-specific heuristics can help.

### 3. Warm Starting
If solving similar problems repeatedly, the optimal basis from one can help the next (though this implementation doesn't expose warm starting).

### 4. Cutting Planes
Advanced MILP solvers add "cuts" - constraints that eliminate fractional solutions without removing integer ones. This implementation doesn't include cuts, but they're a key technique in commercial solvers.

## Real-World Applications

### Classic Problems
- **Knapsack:** Select items to maximize value within weight limit
- **Set Cover:** Choose facilities to cover all customers at minimum cost
- **Bin Packing:** Pack items into minimum number of bins
- **Facility Location:** Where to build warehouses to minimize cost

### Practical Use Cases
- **Crew Scheduling:** Assign shifts to workers (integer = number of workers)
- **Power Grid:** Which generators to run (binary on/off decisions)
- **Supply Chain:** How many units to produce/transport (integer quantities)
- **Portfolio:** Select assets to invest in (binary purchase decisions)

## Example: 0/1 Knapsack

```python
from solvor import solve_milp

# Items with (value, weight)
items = [(60, 10), (100, 20), (120, 30)]
capacity = 50

n = len(items)
values = [v for v, w in items]
weights = [w for v, w in items]

# Maximize sum of values, subject to weight <= capacity
# Variables x[i] in {0, 1} indicate whether to take item i

result = solve_milp(
    c=[-v for v in values],    # Maximize (negate for minimization)
    A=[weights,                 # Weight constraint
       *[[1 if i == j else 0 for i in range(n)] for j in range(n)]],  # x[i] <= 1
    b=[capacity, *[1]*n],
    integers=list(range(n)),
    minimize=False
)

print(f"Take items: {[i for i, x in enumerate(result.solution) if x > 0.5]}")
print(f"Total value: {-result.objective}")
```

## Related Solvers

- [`solve_lp`](solve_lp) - LP without integer constraints (faster, used internally)
- [`Model`](Model-(CP-SAT)) - For complex logical constraints
- [`solve_sat`](solve_sat) - For pure boolean (0/1) problems
- [`anneal`](anneal) / [`evolve`](evolve) - Heuristics for large-scale integer programming

## See Also

- [Linear & Integer Programming](Linear-&-Integer-Programming) - Category overview
- [Cookbook: Knapsack](Cookbook-Knapsack) - Full knapsack example
- [Cookbook: Resource Allocation](Cookbook-Resource-Allocation) - MILP for scheduling

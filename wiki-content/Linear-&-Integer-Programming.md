# Linear & Integer Programming

When your problem can be expressed as "maximize profit subject to resource constraints" or "minimize cost while meeting requirements," you're in LP/MILP territory. The crystal-walking, corner-finding algorithms that power production planning, resource allocation, and diet optimization.

## Solvers in this Category

### [solve_lp](solve_lp)
Linear programming with continuous variables. The simplex algorithm walks along edges of a multidimensional crystal, always uphill, until it hits an optimal corner. Use this when all variables can be fractional.

**Complexity:** O(exponential worst case, polynomial average case)
**Guarantees:** Finds the exact optimum for LP problems

### [solve_milp](solve_milp)
Mixed-Integer Linear Programming. Like `solve_lp` but some variables must be integers. Uses branch-and-bound: solves LP relaxations, branches on fractional values, prunes impossible subtrees. The workhorse for scheduling, set covering, and facility location.

**Complexity:** NP-hard (exponential worst case)
**Guarantees:** Finds provably optimal integer solutions

## When to Use This Category

**Perfect for:**
- Resource allocation (workers, machines, budget)
- Production planning (what to make, how much)
- Diet problems (minimize cost, meet nutrition requirements)
- Blending (mixing ingredients to meet specs)
- Scheduling with discrete time slots
- Facility location and network design
- Any problem with a linear objective and linear constraints

**The magic:** You get certificates of optimality. When MILP says "this is optimal," you know for sure.

## When NOT to Use This Category

- **Non-linear objectives or constraints** → Use gradient descent, anneal, or genetic algorithms
- **Boolean logic and complex constraints** → Use CP-SAT or SAT solvers
- **No clear objective to optimize** → Use constraint programming for satisfaction
- **Numerical instability** → Simplex can struggle with badly scaled coefficients

## Comparing Solvers

| Solver | Variables | Speed | Use When |
|--------|-----------|-------|----------|
| `solve_lp` | Continuous only | Fast | No integer constraints, just optimization |
| `solve_milp` | Mixed (continuous + integer) | Slower | Need discrete decisions (0/1, counts) |

Both use the same underlying simplex engine. MILP just adds the branch-and-bound layer on top for integer constraints.

## Quick Example

```python
from solvor import solve_lp, solve_milp

# LP: Maximize 3x + 2y subject to x + y <= 4, x,y >= 0
result = solve_lp(c=[-3, -2], A=[[1, 1]], b=[4], minimize=False)
print(result.solution)  # (4.0, 0.0), objective = 12.0

# MILP: Same problem but x and y must be integers
result = solve_milp(c=[-3, -2], A=[[1, 1]], b=[4], integers=[0, 1], minimize=False)
print(result.solution)  # (4, 0), objective = 12
```

## Tips & Tricks

1. **Start with LP relaxation** - Solve without integer constraints first. If the solution is already integer, you're done! If not, you know an upper bound on the optimal value.

2. **Scaling matters** - Keep coefficients in similar ranges. Coefficients like `[1e-8, 1e8]` will cause numerical issues.

3. **Tight formulations** - Add redundant constraints that tighten the LP relaxation. Better bounds = faster MILP solving.

4. **Warm starting** - For similar problems, the optimal basis from one LP often helps solve the next.

## Real-World Applications

- **Transportation:** Minimize shipping costs while meeting demand
- **Manufacturing:** Maximize profit given machine capacity and materials
- **Portfolio optimization:** Allocate investments to maximize return given risk constraints
- **Cutting stock:** Minimize waste when cutting materials to size
- **Crew scheduling:** Assign shifts to workers minimizing cost while covering all shifts

## See Also

- [Constraint Programming](Constraint-Programming) - Better for logical constraints
- [Network Flow](Network-Flow-&-MST) - Specialized LP for flow problems
- [Cookbook: Resource Allocation](Cookbook-Resource-Allocation) - Full example

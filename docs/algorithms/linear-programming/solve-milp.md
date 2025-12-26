# solve_milp

Mixed-Integer Linear Programming. Like `solve_lp` but some variables must be integers. Uses branch-and-bound: solves LP relaxations, branches on fractional values, prunes impossible subtrees.

## When to Use

- Scheduling with discrete time slots
- Facility location and network design
- Set covering problems
- Any LP where some decisions are discrete (yes/no, counts)

## Signature

```python
def solve_milp(
    c: Sequence[float],
    A: Sequence[Sequence[float]],
    b: Sequence[float],
    integers: Sequence[int],
    *,
    minimize: bool = True,
    eps: float = 1e-6,
    max_iter: int = 10_000,
    max_nodes: int = 100_000,
    gap_tol: float = 1e-6,
    warm_start: Sequence[float] | None = None,
    solution_limit: int = 1,
) -> Result[tuple[float, ...]]
```

## Parameters

| Parameter | Description |
|-----------|-------------|
| `c` | Objective coefficients |
| `A` | Constraint matrix (Ax ≤ b) |
| `b` | Constraint right-hand sides |
| `integers` | Indices of variables that must be integers (required) |
| `minimize` | If False, maximize instead |
| `eps` | Numerical tolerance for integrality |
| `max_iter` | Maximum LP iterations per node |
| `max_nodes` | Maximum branch-and-bound nodes to explore |
| `gap_tol` | Stop when gap between bound and incumbent is below this |
| `warm_start` | Initial feasible solution to start from |
| `solution_limit` | Stop after finding this many solutions |

## Example

```python
from solvor import solve_milp

# Maximize 3x + 2y, x must be integer, subject to x + y <= 4
result = solve_milp(
    c=[-3, -2],
    A=[[1, 1]],
    b=[4],
    integers=[0],
    minimize=False
)
print(result.solution)  # (4.0, 0.0)
print(result.objective)  # 12.0
```

## Finding Multiple Solutions

```python
# Find up to 5 different solutions
result = solve_milp(c, A, b, integers=[0, 1], solution_limit=5)
if result.solutions:
    for i, sol in enumerate(result.solutions):
        print(f"Solution {i+1}: {sol}")
```

## Binary Variables

For 0/1 decisions, specify the variable as integer and add bounds:

```python
# Binary variable x (0 or 1)
# Add constraint: x <= 1
result = solve_milp(c, A + [[1, 0]], b + [1], integers=[0])
```

## Complexity

- **Time:** NP-hard (exponential worst case)
- **Guarantees:** Finds provably optimal integer solutions

## Tips

1. **Start with LP relaxation.** Solve as LP first. If the solution is already integer, you're done. The LP objective is a bound on the optimal integer objective.
2. **Tight formulations.** Adding redundant constraints that tighten the LP relaxation speeds up MILP solving.
3. **Warm starting.** Pass a known feasible solution via `warm_start` to prune early.
4. **Gap tolerance.** For large problems, set `gap_tol=0.01` to accept solutions within 1% of optimal.

## See Also

- [solve_lp](solve-lp.md) - When all variables are continuous
- [Cookbook: Resource Allocation](../../cookbook/resource-allocation.md) - MILP example

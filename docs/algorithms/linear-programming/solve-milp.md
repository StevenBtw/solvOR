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
    *,
    integers: Sequence[int] | None = None,
    minimize: bool = True,
    max_iter: int = 100_000,
    max_solutions: int | None = None,
    warm_start: Sequence[float] | None = None,
    eps: float = 1e-10,
) -> Result[list[float]]
```

## Parameters

| Parameter | Description |
|-----------|-------------|
| `c` | Objective coefficients |
| `A` | Constraint matrix (Ax ≤ b) |
| `b` | Constraint right-hand sides |
| `integers` | Indices of variables that must be integers |
| `minimize` | If False, maximize instead |
| `max_iter` | Maximum branch-and-bound iterations |
| `max_solutions` | Stop after finding this many solutions |
| `warm_start` | Initial solution to start from |
| `eps` | Numerical tolerance |

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
print(result.solution)  # [4, 0]
print(result.objective)  # 12
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

## See Also

- [solve_lp](solve-lp.md) - When all variables are continuous
- [Cookbook: Resource Allocation](../../cookbook/resource-allocation.md) - MILP example

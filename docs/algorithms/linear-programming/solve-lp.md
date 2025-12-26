# solve_lp

Linear programming with continuous variables. The simplex algorithm walks along edges of a multidimensional crystal, always uphill, until it hits an optimal corner.

## When to Use

- Resource allocation (workers, machines, budget)
- Production planning (what to make, how much)
- Diet problems (minimize cost, meet nutrition requirements)
- Blending (mixing ingredients to meet specs)
- Any problem with a linear objective and linear constraints

## Signature

```python
def solve_lp(
    c: Sequence[float],
    A: Sequence[Sequence[float]],
    b: Sequence[float],
    *,
    minimize: bool = True,
    max_iter: int = 100_000,
    eps: float = 1e-10,
) -> Result[list[float]]
```

## Parameters

| Parameter | Description |
|-----------|-------------|
| `c` | Objective coefficients (minimize c·x) |
| `A` | Constraint matrix (Ax ≤ b) |
| `b` | Constraint right-hand sides |
| `minimize` | If False, maximize instead |
| `max_iter` | Maximum simplex iterations |
| `eps` | Numerical tolerance |

## Example

```python
from solvor import solve_lp

# Maximize 3x + 2y subject to x + y <= 4, x,y >= 0
result = solve_lp(c=[-3, -2], A=[[1, 1]], b=[4], minimize=False)
print(result.solution)  # [4.0, 0.0]
print(result.objective)  # 12.0
```

## Constraint Directions

All constraints are `Ax ≤ b`. For other directions:

```python
# Want: x + y >= 4
# Multiply by -1: -x - y <= -4
result = solve_lp(c, [[-1, -1]], [-4])

# Want: x + y == 4
# Add both directions: x + y <= 4 AND x + y >= 4
result = solve_lp(c, [[1, 1], [-1, -1]], [4, -4])
```

## Complexity

- **Time:** O(exponential worst case, polynomial average case)
- **Guarantees:** Finds the exact optimum for LP problems

## Tips

1. **Scaling matters.** Keep coefficients in similar ranges. Mixing 1e-8 and 1e8 causes numerical issues.
2. **Start with LP relaxation.** When solving MILP, solve without integer constraints first to get bounds.
3. **Check status.** Always verify `result.ok` before using the solution.

## See Also

- [solve_milp](solve-milp.md) - When variables must be integers
- [Cookbook: Production Planning](../../cookbook/production-planning.md) - Full example
- [Cookbook: Diet Problem](../../cookbook/diet.md) - Classic LP example

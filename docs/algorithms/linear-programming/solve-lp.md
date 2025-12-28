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
    eps: float = 1e-10,
    max_iter: int = 100_000,
) -> Result[tuple[float, ...]]
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
result = solve_lp(c=[3, 2], A=[[1, 1]], b=[4], minimize=False)
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

## How It Works

The simplex algorithm exploits a key insight: the optimal solution to a linear program always occurs at a vertex (corner) of the feasible region. Instead of searching the entire space, we hop from vertex to vertex, always improving the objective.

**The geometry:** Your constraints define a convex polytope in n-dimensional space. Each vertex is where n constraints meet. The objective function defines a direction—we're looking for the vertex furthest in that direction.

**The algebra:** We convert to standard form with slack variables:

```text
minimize c·x
subject to Ax + s = b, x ≥ 0, s ≥ 0
```

Each vertex corresponds to a *basic feasible solution*—setting n variables to zero and solving for the rest. The algorithm:

1. Start at a vertex (basic feasible solution)
2. Look at adjacent vertices (one pivot away)
3. Move to a neighbor with better objective value
4. Repeat until no improvement possible → optimal

**The pivot:** Each iteration picks an entering variable (improves objective) and leaving variable (maintains feasibility), then updates the tableau. This is the "walking along edges" part.

**Two-phase method:** If the origin isn't feasible (some constraints violated), Phase I finds a feasible starting vertex using artificial variables. Phase II then optimizes.

**Bland's rule:** Prevents cycling (revisiting the same vertex) by always picking the smallest index when ties occur.

For the full algorithm, see [Linear Programming on Wikipedia](https://en.wikipedia.org/wiki/Simplex_algorithm) or the classic textbook by Chvátal.

## Complexity

- **Time:** O(2^n) worst case, but O(n²m) average on random instances
- **Space:** O(nm) for the tableau
- **Guarantees:** Finds the exact global optimum (not approximate)

## Tips

1. **Scaling matters.** Keep coefficients in similar ranges. Mixing 1e-8 and 1e8 causes numerical issues.
2. **Start with LP relaxation.** When solving MILP, solve without integer constraints first to get bounds.
3. **Check status.** Always verify `result.ok` before using the solution.

## See Also

- [solve_milp](solve-milp.md) - When variables must be integers
- [Cookbook: Production Planning](../../cookbook/production-planning.md) - Full example
- [Cookbook: Diet Problem](../../cookbook/diet.md) - Classic LP example

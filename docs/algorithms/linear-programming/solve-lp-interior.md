# solve_lp_interior

Linear programming using interior point method. While simplex walks along edges of the feasible polytope, interior point cuts straight through the middle. Think of simplex as taking the stairs, interior point as taking the elevator, different path, same destination.

## When to Use

- Same problems as simplex (LP with continuous variables)
- When simplex is cycling or slow on degenerate problems
- When you want to understand how modern LP solvers work (HiGHS, CPLEX, Gurobi all use interior point)
- Educational purposes, learning the "other" way to solve LP

## Signature

```python
def solve_lp_interior(
    c: Sequence[float],
    A: Sequence[Sequence[float]],
    b: Sequence[float],
    *,
    minimize: bool = True,
    eps: float = 1e-8,
    max_iter: int = 100,
) -> Result[tuple[float, ...]]
```

## Parameters

| Parameter | Description |
|-----------|-------------|
| `c` | Objective coefficients (minimize c·x) |
| `A` | Constraint matrix (Ax ≤ b) |
| `b` | Constraint right-hand sides |
| `minimize` | If False, maximize instead |
| `max_iter` | Maximum Newton iterations |
| `eps` | Convergence tolerance |

## Example

```python
from solvor import solve_lp_interior

# Maximize 3x + 2y subject to x + y <= 4, x,y >= 0
result = solve_lp_interior(c=[3, 2], A=[[1, 1]], b=[4], minimize=False)
print(result.solution)  # (4.0, 0.0) approximately
print(result.objective)  # 12.0 approximately
```

## How It Works

While simplex hops between vertices, interior point methods stay strictly inside the feasible region and approach optimality along a curved path called the *central path*.

**The barrier idea:** Add a logarithmic penalty for approaching boundaries:

```text
minimize c·x - μ Σ log(xᵢ)
```

As μ→0, the solution approaches the true optimum. But we don't solve this directly.

**Primal-dual formulation:** Instead of just the primal problem, we solve primal and dual simultaneously. The optimality conditions (KKT) are:

```text
Ax = b           (primal feasibility)
A'y + z = c      (dual feasibility)
xᵢzᵢ = 0         (complementarity)
x, z ≥ 0
```

We relax complementarity to xᵢzᵢ = μ and drive μ→0.

**Newton's method:** Each iteration solves a linear system (the KKT system) to find a direction, then takes a step while staying positive. The magic is that convergence is polynomial, typically 20-50 iterations regardless of problem size.

**Mehrotra predictor-corrector:** Two Newton steps per iteration. First a "predictor" step toward the boundary, then a "corrector" step that recenters. Nearly doubles the practical speed.

**Normal equations:** The 3×3 block KKT system reduces to a smaller m×m system (A D A'), solved via Cholesky decomposition.

For the theory, see [Interior Point Methods on Wikipedia](https://en.wikipedia.org/wiki/Interior-point_method) or Nocedal & Wright's *Numerical Optimization*.

## Simplex vs Interior Point

| Aspect | Simplex | Interior Point |
|--------|---------|----------------|
| Path | Walks edges | Cuts through interior |
| Solution | Exact vertex | Approximate (converges) |
| Degeneracy | Can cycle | No cycling issues |
| Warm start | Easy | Difficult |
| Sparse problems | Good | Better |

## Complexity

- **Time:** O(n^3.5 log(1/ε)) theoretical, O(n² × iterations) practical
- **Iterations:** Typically 20-50 for convergence
- **Guarantees:** Converges to optimal with polynomial complexity

## Tips

1. **Tolerance matters.** Interior point finds approximate solutions. Use `eps=1e-8` for most problems.
2. **Fewer iterations.** Interior point typically needs 20-50 iterations vs thousands for simplex on large problems.
3. **Not vertex-exact.** Solutions are interior points that approach optimality, not exact vertices.

## See Also

- [solve_lp](solve-lp.md) — Simplex method (original LP algorithm)
- [solve_milp](solve-milp.md) — When variables must be integers

# powell

Conjugate direction method. Optimizes along each axis, then along the direction of total progress. No derivatives needed.

## Example

```python
from solvor import powell

def objective(x):
    return (x[0] - 2)**2 + (x[1] + 1)**2

result = powell(objective, x0=[0.0, 0.0])
print(result.solution)  # Close to [2, -1]
```

## With Bounds

```python
result = powell(objective, x0=[0.0, 0.0], bounds=[(-5, 5), (-5, 5)])
```

## When to Use

- Non-smooth objectives
- No gradient information
- Low to moderate dimensions

## How It Works

**The core idea:** Minimize along one axis, then the next, then the next... then along the direction you've traveled overall. This "conjugate direction" trick accelerates convergence dramatically.

**Why axis-by-axis isn't enough:** Simple coordinate descent (optimize x₁, then x₂, then x₃...) can zig-zag badly on elongated valleys. Imagine a narrow diagonal valley—you take tiny orthogonal steps that barely make progress. Powell's method fixes this.

**The algorithm:**

1. Start with coordinate directions {e₁, e₂, ..., eₙ}
2. Minimize along each direction in turn, updating position after each
3. Compute the overall displacement vector d = x_new − x_old
4. Minimize along d (this is the magic step)
5. Replace the direction that gave the least improvement with d
6. Repeat until converged

**The conjugate property:** After k iterations on a quadratic function, Powell's method generates directions that are *conjugate* with respect to the Hessian. This means they're independent in a certain sense—optimizing along one won't undo progress along another.

**No derivatives needed:** Each line search just needs function evaluations. The algorithm builds up curvature information implicitly through the direction updates, similar to how quasi-Newton methods approximate the Hessian.

**Trade-offs:** Fewer evaluations than Nelder-Mead on smooth problems, but can get stuck on highly non-convex landscapes. Works well up to ~20 dimensions.

## See Also

- [Nelder-Mead](nelder-mead.md) - Alternative derivative-free method

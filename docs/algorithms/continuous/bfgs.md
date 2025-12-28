# BFGS / L-BFGS

Quasi-Newton methods. Approximate the Hessian for faster convergence than gradient descent.

## bfgs

BFGS with optional line search.

```python
from solvor import bfgs

def objective(x):
    return x[0]**2 + x[1]**2

def grad(x):
    return [2*x[0], 2*x[1]]

result = bfgs(objective, grad, x0=[5.0, 5.0])
print(result.solution)  # Close to [0, 0]
```

**Memory:** O(n²) for Hessian approximation
**Best for:** Smooth functions, moderate dimensions

## lbfgs

Limited-memory BFGS. Uses limited history instead of full Hessian.

```python
from solvor import lbfgs

result = lbfgs(objective, grad, x0=[5.0, 5.0], memory=10)
```

**Memory:** O(n × m) where m is history size
**Best for:** Large-scale problems

## Comparison

| Method | Memory | Convergence | Use When |
|--------|--------|-------------|----------|
| BFGS | O(n²) | Superlinear | n < 1000, smooth function |
| L-BFGS | O(n × m) | Superlinear | n > 1000, memory limited |

## How It Works

**The Newton's method dream:** Newton's method converges quadratically—insanely fast—by using the Hessian (second derivatives) to find the exact step to the minimum of the local quadratic approximation:

```text
x_{k+1} = x_k - H⁻¹ ∇f(x_k)
```

**The problem:** Computing and inverting the Hessian is O(n²) storage and O(n³) per iteration. For large n, this is prohibitive.

**BFGS insight:** Build an approximation to H⁻¹ using only gradient information. Each iteration updates this approximation using the secant condition:

```text
B_{k+1} s_k = y_k
```

where s_k = x_{k+1} − x_k (step) and y_k = ∇f_{k+1} − ∇f_k (gradient change).

**The BFGS update:** A rank-2 update that maintains positive definiteness:

```text
B_{k+1} = (I - ρ s y') B_k (I - ρ y s') + ρ s s'
where ρ = 1 / (y' s)
```

This satisfies the secant condition and keeps B symmetric positive definite.

**L-BFGS trick:** Don't store B at all. Store the last m pairs (s_k, y_k) and compute B⁻¹∇f implicitly via a two-loop recursion. Storage drops from O(n²) to O(nm).

**Why it works:** After seeing enough curvature information, B approximates the true inverse Hessian well enough for superlinear convergence. You get Newton-like speed with gradient-only cost.

**The algorithm:**

1. Compute gradient ∇f(x)
2. Compute search direction p = −B∇f (using stored history for L-BFGS)
3. Line search to find step size α satisfying Wolfe conditions
4. Update x ← x + αp
5. Update B (or store (s, y) pair for L-BFGS)
6. Repeat until ||∇f|| < ε

## Tips

- **Smooth functions only.** These methods assume twice-differentiable objectives.
- **Good for ML.** Fast convergence on convex losses.
- **Line search built-in.** Automatically finds step size.

## See Also

- [Gradient Descent](gradient.md) - Simpler but slower
- [Nelder-Mead](nelder-mead.md) - No gradients needed

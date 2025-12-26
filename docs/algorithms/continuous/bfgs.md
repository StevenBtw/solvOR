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

## Tips

- **Smooth functions only.** These methods assume twice-differentiable objectives.
- **Good for ML.** Fast convergence on convex losses.
- **Line search built-in.** Automatically finds step size.

## See Also

- [Gradient Descent](gradient.md) - Simpler but slower
- [Nelder-Mead](nelder-mead.md) - No gradients needed

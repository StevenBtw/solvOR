# gradient_descent

**Vanilla gradient descent, follow the slope downhill.**

## At a Glance

- **Category:** [Continuous Optimization](Continuous-Optimization)
- **Problem Type:** Smooth continuous optimization
- **Complexity:** O(iterations × gradient_cost)
- **Guarantees:** Converges to local minimum
- **Status:** Returns solution or `MAX_ITER`

## When to Use This

Use when you have gradient information and a smooth objective. Simple, interpretable baseline.

## Quick Example

```python
from solvor import gradient_descent

def grad(x):
    return [2*x[0], 2*x[1]]  # Gradient of x^2 + y^2

result = gradient_descent(grad, x0=[5, 5], lr=0.1)
print(result.solution)  # Close to [0, 0]
```

## See Also
- [`adam`](adam) - Usually better
- [Continuous Optimization](Continuous-Optimization)

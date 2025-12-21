# adam

**Adam optimizer - momentum + RMSprop, the default choice.**

## At a Glance

- **Category:** [Continuous Optimization](Continuous-Optimization)
- **Problem Type:** General continuous optimization
- **Complexity:** O(iterations × gradient_cost)
- **Guarantees:** Robust, adaptive

## When to Use This

The default optimizer for deep learning and most continuous optimization. Works almost everywhere.

## Quick Example

```python
from solvor import adam

def grad(x):
    return [2*x[0], 2*x[1]]

result = adam(grad, x0=[5, 5])  # Default lr=0.001 usually works
print(result.solution)
```

## See Also
- [Continuous Optimization](Continuous-Optimization)

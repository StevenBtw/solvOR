# momentum

**Gradient descent with velocity - smooths oscillations.**

## At a Glance

- **Category:** [Continuous Optimization](Continuous-Optimization)
- **Complexity:** O(iterations × gradient_cost)
- **Guarantees:** Often faster than vanilla GD

## Quick Example

```python
from solvor import momentum

result = momentum(grad, x0=[5, 5], lr=0.01, beta=0.9)
```

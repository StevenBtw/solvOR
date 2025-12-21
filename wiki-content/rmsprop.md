# rmsprop

**Adaptive learning rate per parameter using RMS of gradients.**

## At a Glance

- **Category:** [Continuous Optimization](Continuous-Optimization)
- **Complexity:** O(iterations × gradient_cost)

## Quick Example

```python
from solvor import rmsprop

result = rmsprop(grad, x0=[5, 5], lr=0.01)
```

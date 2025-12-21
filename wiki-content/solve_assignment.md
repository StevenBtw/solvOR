# solve_assignment

**Assignment via min-cost flow reduction.**

## At a Glance

- **Category:** [Assignment](Assignment)
- **Complexity:** O(n³) via flow

Less efficient than Hungarian but conceptually simple.

## Quick Example

```python
from solvor import solve_assignment

costs = [[10, 5], [3, 9]]
result = solve_assignment(costs)
print(result.solution)  # [1, 0]
```

## See Also
- [`hungarian`](hungarian) - Faster

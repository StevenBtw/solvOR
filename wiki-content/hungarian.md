# hungarian

**Optimal assignment in O(n³) - the workhorse for matching.**

## At a Glance

- **Category:** [Assignment](Assignment)
- **Problem Type:** Optimal one-to-one matching
- **Complexity:** O(n³)
- **Guarantees:** Optimal assignment

## Quick Example

```python
from solvor import hungarian

costs = [
    [10, 5, 13],  # Worker 0 costs
    [3, 9, 18],   # Worker 1 costs
    [10, 6, 12]   # Worker 2 costs
]

result = hungarian(costs)
print(result.solution)  # [1, 0, 2] -> Worker 0→Task 1, etc
print(result.objective)  # 20 (total cost)
```

## See Also
- [Cookbook: Assignment](Cookbook-Assignment)
- [Assignment](Assignment)

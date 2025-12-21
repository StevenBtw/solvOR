# Cookbook: Knapsack Problem

Select items to maximize value within weight constraint.

## Example

```python
from solvor import solve_milp

# Items: (value, weight)
items = [(60, 10), (100, 20), (120, 30), (80, 15)]
capacity = 50

n = len(items)
values = [v for v, w in items]
weights = [w for v, w in items]

# Maximize value
result = solve_milp(
    c=[-v for v in values],
    A=[weights],
    b=[capacity],
    integers=list(range(n)),
    minimize=False
)

selected = [i for i, x in enumerate(result.solution) if x > 0.5]
print(f"Selected items: {selected}")
print(f"Total value: {-result.objective}")
print(f"Total weight: {sum(weights[i] for i in selected)}")
```

## See Also
- [`solve_milp`](solve_milp)
- [Linear & Integer Programming](Linear-&-Integer-Programming)

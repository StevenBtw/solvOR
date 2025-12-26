# solve_knapsack

The classic "what fits in your bag" problem. Select items to maximize value within capacity.

## Example

```python
from solvor import solve_knapsack

values = [60, 100, 120]
weights = [10, 20, 30]

result = solve_knapsack(values, weights, capacity=50)
print(result.solution)   # (1, 2) - indices of selected items
print(result.objective)  # 220 - total value
```

## Signature

```python
def solve_knapsack(
    values: Sequence[float],
    weights: Sequence[float],
    capacity: float,
    *,
    minimize: bool = False,
) -> Result[tuple[int, ...]]
```

## Returns

- `solution`: Tuple of indices of selected items (e.g., `(1, 2)` means items 1 and 2 were selected)
- `objective`: Total value of selected items

## Complexity

Uses dynamic programming: O(n × capacity) for integer weights.

## See Also

- [Cookbook: Knapsack](../../cookbook/knapsack.md) - Full example
- [Bin Packing](bin-packing.md) - Related problem

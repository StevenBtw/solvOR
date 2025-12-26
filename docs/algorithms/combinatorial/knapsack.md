# solve_knapsack

The classic "what fits in your bag" problem. Select items to maximize value within capacity.

## Example

```python
from solvor import solve_knapsack

values = [60, 100, 120]
weights = [10, 20, 30]

result = solve_knapsack(values, weights, capacity=50)
print(result.solution)   # [1, 1, 1] - which items to take
print(result.objective)  # 220 - total value
```

## Signature

```python
def solve_knapsack(
    values: Sequence[int | float],
    weights: Sequence[int | float],
    capacity: int | float,
) -> Result[list[int]]
```

## Returns

- `solution`: List of 0/1 indicating which items are selected
- `objective`: Total value of selected items

## Complexity

Uses dynamic programming: O(n × capacity) for integer weights.

## See Also

- [Cookbook: Knapsack](../../cookbook/knapsack.md) - Full example
- [Bin Packing](bin-packing.md) - Related problem

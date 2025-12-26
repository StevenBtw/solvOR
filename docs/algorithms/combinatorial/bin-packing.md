# solve_bin_pack

Bin packing with First Fit Decreasing. Minimize bins needed to fit items.

## Example

```python
from solvor import solve_bin_pack

items = [4, 8, 1, 4, 2, 1]

result = solve_bin_pack(items, bin_capacity=10)
print(result.solution)   # [[8, 2], [4, 4, 1, 1]]
print(result.objective)  # 2 bins
```

## Signature

```python
def solve_bin_pack(
    items: Sequence[int | float],
    bin_capacity: int | float,
) -> Result[list[list[int | float]]]
```

## Returns

- `solution`: List of bins, each bin is a list of item sizes
- `objective`: Number of bins used

## Algorithm

First Fit Decreasing:
1. Sort items by size (descending)
2. For each item, put it in the first bin that fits
3. If no bin fits, open a new bin

**Guarantee:** At most 11/9 × OPT + 6/9 bins (asymptotically optimal).

## See Also

- [Cookbook: Bin Packing](../../cookbook/bin-packing.md) - Full example
- [Knapsack](knapsack.md) - Related problem

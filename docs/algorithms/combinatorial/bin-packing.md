# solve_bin_pack

Bin packing heuristics. Minimize bins needed to fit items.

## Example

```python
from solvor import solve_bin_pack

items = [4, 8, 1, 4, 2, 1]

result = solve_bin_pack(items, bin_capacity=10)
print(result.solution)   # (1, 0, 0, 1, 0, 0) - bin index for each item
print(result.objective)  # 2 bins
```

## Signature

```python
def solve_bin_pack(
    item_sizes: Sequence[float],
    bin_capacity: float,
    *,
    algorithm: str = "best-fit-decreasing",
) -> Result[tuple[int, ...]]
```

## Returns

- `solution`: Tuple of bin assignments, `solution[i]` = bin index for item i
- `objective`: Number of bins used

## Algorithms

- `first-fit`: Place item in first bin that fits
- `best-fit`: Place item in bin with least remaining space
- `first-fit-decreasing`: Sort items descending, then first-fit
- `best-fit-decreasing`: Sort items descending, then best-fit (default)

Decreasing variants typically produce better results.

**Guarantee:** Best-fit-decreasing uses at most 11/9 × OPT + 6/9 bins.

## See Also

- [Cookbook: Bin Packing](../../cookbook/bin-packing.md) - Full example
- [Knapsack](knapsack.md) - Related problem

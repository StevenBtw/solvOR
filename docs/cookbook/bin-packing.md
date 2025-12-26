# Bin Packing

Pack items into the minimum number of fixed-capacity bins.

Classic combinatorial optimization problem with applications in container loading, memory allocation, and cutting stock.

## The Problem

Given items with sizes and bins with fixed capacity, assign each item to a bin without exceeding capacity, using as few bins as possible.

## Example

```python
from solvor import solve_bin_pack

items = [4, 8, 1, 4, 2, 1, 5, 3]
capacity = 10

result = solve_bin_pack(items, capacity, algorithm="best-fit-decreasing")
print(f"Bins used: {int(result.objective)}")

# Show bin contents
bins = {}
for item_idx, bin_idx in enumerate(result.solution):
    bins.setdefault(bin_idx, []).append(items[item_idx])
for bin_idx, contents in sorted(bins.items()):
    print(f"  Bin {bin_idx}: {contents} (sum={sum(contents)})")
```

Output:
```
Bins used: 3
  Bin 0: [8, 2] (sum=10)
  Bin 1: [5, 4, 1] (sum=10)
  Bin 2: [4, 3, 1] (sum=8)
```

## Algorithms

- `first-fit` - Place in first bin that fits
- `best-fit` - Place in tightest-fitting bin
- `first-fit-decreasing` - Sort items large-to-small, then first-fit
- `best-fit-decreasing` - Sort items large-to-small, then best-fit

FFD/BFD typically perform best. FFD is guaranteed to use at most 11/9 x OPT + 6/9 bins.

## See Also

- [Knapsack](knapsack.md)

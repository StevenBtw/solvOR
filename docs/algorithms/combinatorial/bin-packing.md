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

## How It Works

**The NP-hard reality:** Finding the optimal bin packing is NP-hard. But simple greedy heuristics work remarkably well in practice.

**First-fit:** Scan bins in order, place item in first bin where it fits. If none fit, open a new bin. Fast and simple.

**Best-fit:** Scan all bins, place item in the bin where it fits *most tightly* (least remaining space). Slightly slower but often better packing.

**The decreasing trick:** Sort items largest-first before packing. Big items are hard to place, so pack them while you have flexibility. Small items fit in gaps later.

```text
Items: [4, 8, 1, 4, 2, 1], capacity = 10

Best-fit-decreasing:
  Sorted: [8, 4, 4, 2, 1, 1]

  Bin 0: [8]        → 2 remaining
  Bin 1: [4, 4]     → 2 remaining
  Bin 1: [4, 4, 2]  → 0 remaining (full)
  Bin 0: [8, 1, 1]  → 0 remaining (full)

  Result: 2 bins
```

**Why decreasing helps:** Without sorting, you might fill bins with small items, then have nowhere to put the big ones. Sorting ensures big items get priority.

**The approximation guarantee:** Best-fit-decreasing is provably within 11/9 of optimal—at most 22% more bins than necessary, plus a small constant. For practical inputs it's often optimal or within one bin.

**When to go exact:** For critical applications, formulate as integer programming (MILP) or use branch-and-bound. But for most cases, BFD is fast and good enough.

## See Also

- [Cookbook: Bin Packing](../../cookbook/bin-packing.md) - Full example
- [Knapsack](knapsack.md) - Related problem

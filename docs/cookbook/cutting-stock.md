# Cutting Stock

Minimize the number of rolls used to cut required pieces.

## Problem

A paper mill has rolls of width 100 units. Customers need pieces of various sizes:

- 97 pieces of width 45
- 610 pieces of width 36
- 395 pieces of width 31
- 211 pieces of width 14

Find cutting patterns that minimize waste.

## Solution

```python
from solvor import solve_cg

result = solve_cg(
    demands=[97, 610, 395, 211],
    roll_width=100,
    piece_sizes=[45, 36, 31, 14],
)

print(f"Rolls needed: {int(result.objective)}")
print(f"Patterns used: {len(result.solution)}")

for pattern, count in sorted(result.solution.items(), key=lambda x: -x[1]):
    sizes = [45, 36, 31, 14]
    desc = " + ".join(f"{n}×{s}" for n, s in zip(pattern, sizes) if n > 0)
    waste = 100 - sum(n * s for n, s in zip(pattern, sizes))
    print(f"  {desc} (waste: {waste}) × {count}")
```

Output:

```text
Rolls needed: 454
Patterns used: 4
  1×36 + 2×31 (waste: 2) × 198
  2×36 + 2×14 (waste: 0) × 106
  2×36 (waste: 28) × 101
  2×45 (waste: 10) × 49
```

## Why Column Generation?

**The naive approach fails:** With 4 piece sizes that each fit 2-7 times per roll, there are hundreds of possible patterns. Enumerating them all in a MILP is slow.

**Column generation is smarter:** Start with simple patterns, solve a small LP, then ask "is there a better pattern?" using a knapsack subproblem. Only patterns that improve the solution are generated.

**Result:** 454 rolls vs. the theoretical minimum of 453 (LP lower bound). The 1-roll gap comes from rounding.

## Verifying Demands

```python
# Check all demands are satisfied
sizes = [45, 36, 31, 14]
demands = [97, 610, 395, 211]

for i, (size, demand) in enumerate(zip(sizes, demands)):
    produced = sum(pattern[i] * count for pattern, count in result.solution.items())
    print(f"Size {size}: need {demand}, produced {produced}")
```

## Custom Pricing

For non-cutting-stock problems, provide your own pricing function:

```python
def my_pricing(duals: list[float]) -> tuple[tuple[int, ...] | None, float]:
    # Find the most profitable column
    # Returns (column, reduced_cost) or (None, 0) if no improving column
    best_col = None
    best_rc = 0.0

    for candidate in generate_candidates():
        profit = sum(d * c for d, c in zip(duals, candidate))
        reduced_cost = 1.0 - profit  # Column cost is 1
        if reduced_cost < best_rc:
            best_rc = reduced_cost
            best_col = candidate

    return (best_col, -best_rc) if best_col else (None, 0.0)

result = solve_cg(
    demands=[10, 20, 30],
    pricing_fn=my_pricing,
    initial_columns=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
)
```

## See Also

- [solve_cg](../algorithms/linear-programming/solve-cg.md) - Full API reference
- [Bin Packing](bin-packing.md) - Related problem

# Knapsack Problem

Select items to maximize value within weight constraint. The problem that made dynamic programming famous.

## The Problem

You're packing for a hike. Each item has value and weight. Your bag has a limit. What do you take? This is knapsack - and it's everywhere: budget allocation, cargo loading, cutting stock.

## Example

```python
from solvor import solve_knapsack

values = [60, 100, 120, 80]
weights = [10, 20, 30, 15]
capacity = 50

result = solve_knapsack(values, weights, capacity)

print(f"Selected items: {result.solution}")
print(f"Total value: {result.objective}")
print(f"Total weight: {sum(weights[i] for i in result.solution)}")
```

## MILP Formulation

For more control, use MILP directly:

```python
from solvor import solve_milp

items = [(60, 10), (100, 20), (120, 30), (80, 15)]
capacity = 50

n = len(items)
values = [v for v, w in items]
weights = [w for v, w in items]

result = solve_milp(
    c=[-v for v in values],  # Negative for maximization
    A=[weights],
    b=[capacity],
    integers=list(range(n))
)

selected = [i for i, x in enumerate(result.solution) if x > 0.5]
print(f"Selected: {selected}")
print(f"Total value: {-result.objective}")
```

## Variations

- **0/1 Knapsack:** Each item picked at most once (default)
- **Bounded Knapsack:** Multiple copies of each item available
- **Unbounded Knapsack:** Unlimited copies

## See Also

- [Bin Packing](bin-packing.md)
- [Linear & Integer Programming](../algorithms/linear-programming/solve-milp.md)

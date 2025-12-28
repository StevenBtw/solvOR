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

## How It Works

**The greedy trap:** You might think "just pick items with best value/weight ratio." But that fails: a diamond worth $1000 weighing 1kg beats a gold bar worth $900 weighing 0.9kg by ratio, but if capacity is 0.9kg, you'd rather have the gold.

**Dynamic programming insight:** Build up solutions to smaller subproblems. Let dp\[i\]\[w\] = maximum value using items 0..i with capacity w.

**The recurrence:**

```text
dp[i][w] = max(
    dp[i-1][w],           # don't take item i
    dp[i-1][w-wᵢ] + vᵢ    # take item i (if it fits)
)
```

For each item, you either take it or leave it. If you take it, you get its value but lose that capacity for other items.

**Building the table:**

```text
        Capacity →
      0   1   2   3   4   5
    ┌───┬───┬───┬───┬───┬───┐
  0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │  (no items)
    ├───┼───┼───┼───┼───┼───┤
  1 │ 0 │ 0 │60 │60 │60 │60 │  (item 1: v=60, w=2)
    ├───┼───┼───┼───┼───┼───┤
  2 │ 0 │ 0 │60 │100│100│160│  (item 2: v=100, w=3)
    └───┴───┴───┴───┴───┴───┘
```

**Backtracking:** To find which items were selected, trace back through the table. If dp\[i\]\[w\] ≠ dp\[i-1\]\[w\], item i was taken.

**Why O(n × capacity):** We fill an n × W table, constant work per cell. Note: this is *pseudo-polynomial*—if capacity is huge (like 10⁹), use a different approach (branch & bound, or approximate).

## Complexity

Uses dynamic programming: O(n × capacity) for integer weights.

## See Also

- [Cookbook: Knapsack](../../cookbook/knapsack.md) - Full example
- [Bin Packing](bin-packing.md) - Related problem

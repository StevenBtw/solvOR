# solve_cg

Column generation for problems with exponentially many variables. Implements Dantzig-Wolfe decomposition.

## Example

```python
from solvor import solve_cg

# Cutting stock: minimize rolls to cut required pieces
result = solve_cg(
    demands=[97, 610, 395, 211],
    roll_width=100,
    piece_sizes=[45, 36, 31, 14],
)
print(result.objective)  # 454 rolls
print(result.solution)   # {pattern: count, ...}
```

## Signature

```python
def solve_cg(
    demands: Sequence[int],
    *,
    roll_width: float | None = None,
    piece_sizes: Sequence[float] | None = None,
    pricing_fn: Callable[[list[float]], tuple[tuple[int, ...] | None, float]] | None = None,
    initial_columns: Sequence[Sequence[int]] | None = None,
    max_iter: int = 1000,
    eps: float = 1e-9,
    on_progress: ProgressCallback | None = None,
    progress_interval: int = 0,
) -> Result[dict[tuple[int, ...], int]]
```

## Returns

- `solution`: Dict mapping column tuples to usage counts
- `objective`: Total number of columns/rolls used

## Two Modes

### Cutting Stock Mode

Provide `roll_width` and `piece_sizes`. Built-in knapsack pricing finds optimal patterns.

```python
result = solve_cg(
    demands=[10, 20, 30],
    roll_width=100,
    piece_sizes=[45, 36, 31],
)
```

### Custom Pricing Mode

Provide `pricing_fn` and `initial_columns` for custom problems.

```python
def my_pricing(duals: list[float]) -> tuple[tuple[int, ...] | None, float]:
    # Find column maximizing sum(dual[i] * col[i]) - cost
    # Return (column, reduced_cost) or (None, 0) if no improving column
    ...

result = solve_cg(
    demands=[10, 20, 30],
    pricing_fn=my_pricing,
    initial_columns=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
)
```

## How It Works

**The exponential problem:** Some problems have exponentially many variables. Cutting stock has a variable for every possible cutting pattern. Enumerating them all is impossible.

**Gilmore-Gomory's insight (1961):** Don't enumerate—generate on demand. Start with simple patterns, solve a restricted LP, then ask "is there a better pattern?"

**The master-pricing dance:**

1. **Restricted Master Problem:** Minimize total patterns used subject to meeting demands. This is an LP with only the patterns we know about.

2. **Extract Dual Values:** The LP solution gives dual prices—how much is each demand constraint "worth"?

3. **Pricing Subproblem:** Find the most profitable pattern given current prices. For cutting stock, this is a knapsack: maximize sum(dual[i] × count[i]) subject to fitting in one roll.

4. **Reduced Cost Check:** If the best pattern's profit > 1 (its cost), add it and repeat. If not, we're done—no improving pattern exists.

```text
Iteration 1:
  Patterns: [(2,0,0,0), (0,2,0,0), (0,0,3,0), (0,0,0,7)]
  LP objective: 515.3
  Duals: [0.5, 0.5, 0.33, 0.14]

  Pricing finds: (0,2,0,2) with profit 1.29 > 1
  → Add pattern, continue

Iteration 7:
  Patterns: [8 patterns]
  LP objective: 452.25

  Pricing finds: best profit = 0.98 < 1
  → No improving pattern, done

Round up LP solution → 454 rolls
```

**Why it converges:** Each iteration either adds a new pattern or proves optimality. With finite patterns, it must terminate.

**The rounding gap:** LP gives a lower bound. Rounding up gives a feasible integer solution. For cutting stock, this gap is typically tiny (often 0-1 roll).

## Use This For

- **Cutting stock:** Paper, steel, glass, fabric rolls
- **Bin packing:** Large-scale instances via set covering formulation
- **Vehicle routing:** Routes as columns
- **Crew scheduling:** Shifts/schedules as columns
- **Graph coloring:** Independent sets as columns

## Tips

- **Cutting stock is built-in:** Just provide `roll_width` and `piece_sizes`
- **Custom problems:** Implement a pricing function that returns the most profitable column
- **Monitor progress:** Use `on_progress` to track convergence
- **Large instances:** Column generation scales better than explicit MILP formulation

## References

- Gilmore, P. C., & Gomory, R. E. (1961). A linear programming approach to the cutting-stock problem.
- Desaulniers, G., Desrosiers, J., & Solomon, M. M. (2005). Column Generation.

## See Also

- [solve_bp](solve-bp.md) - Branch-and-price for proven integer optimality
- [solve_lp](solve-lp.md) - Underlying LP solver
- [solve_milp](solve-milp.md) - When you need exact integer solutions
- [Bin Packing](../combinatorial/bin-packing.md) - Heuristic alternative

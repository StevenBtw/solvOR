# solve_bp

Branch-and-price for optimal integer solutions. Combines column generation with branch-and-bound.

## Example

```python
from solvor import solve_bp

# Cutting stock with guaranteed integer optimality
result = solve_bp(
    demands=[97, 610, 395, 211],
    roll_width=100,
    piece_sizes=[45, 36, 31, 14],
)
print(result.objective)  # Integer optimal
print(result.solution)   # {pattern: count, ...}
```

## Signature

```python
def solve_bp(
    demands: Sequence[int],
    *,
    roll_width: float | None = None,
    piece_sizes: Sequence[float] | None = None,
    pricing_fn: Callable[[list[float]], tuple[tuple[int, ...] | None, float]] | None = None,
    initial_columns: Sequence[Sequence[int]] | None = None,
    max_iter: int = 1000,
    max_nodes: int = 10000,
    gap_tol: float = 1e-6,
    eps: float = 1e-9,
    on_progress: ProgressCallback | None = None,
    progress_interval: int = 0,
) -> Result[dict[tuple[int, ...], int]]
```

## Returns

- `solution`: Dict mapping column tuples to integer usage counts
- `objective`: Total number of columns/rolls used (integer)
- `status`: OPTIMAL if proven optimal, FEASIBLE if node limit reached

## Two Modes

Same as `solve_cg`:

### Cutting Stock Mode

```python
result = solve_bp(
    demands=[10, 20, 30],
    roll_width=100,
    piece_sizes=[45, 36, 31],
)
```

### Custom Pricing Mode

```python
result = solve_bp(
    demands=[10, 20, 30],
    pricing_fn=my_pricing,
    initial_columns=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
)
```

## How It Works

**The gap problem:** `solve_cg` gives an LP relaxation. Rounding up gives feasible integers, but may not be optimal. For most cutting stock problems the gap is tiny, but when you need guaranteed optimality, use branch-and-price.

**Branch-and-bound meets column generation:**

1. **Root Node:** Solve LP relaxation via column generation (same as `solve_cg`)
2. **Check Integrality:** If all pattern counts are integer, done
3. **Branch:** Pick most fractional variable x[i], create two children:
   - Left: x[i] ≤ floor(value)
   - Right: x[i] ≥ ceil(value)
4. **Solve Children:** Each child is a new column generation problem with added bounds
5. **Prune:** Skip nodes whose LP bound can't beat current best integer solution
6. **Global Column Pool:** Columns found at any node are available to all nodes

```text
Root: LP = 452.25, fractional
├── x[3] ≤ 45: LP = 453.1, fractional
│   ├── x[5] ≤ 12: LP = 454, integer → incumbent = 454
│   └── x[5] ≥ 13: LP = 455 ≥ 454 → pruned
└── x[3] ≥ 46: LP = 453.8, fractional
    └── ... eventually finds 453

Optimal: 453 rolls (proven)
```

**Why it's better than MILP:** Explicit MILP formulation has exponentially many variables. Branch-and-price generates only the patterns needed, keeping the problem tractable.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_nodes` | 10000 | Branch-and-bound node limit |
| `gap_tol` | 1e-6 | Stop when gap < tolerance |
| `max_iter` | 1000 | Column generation iterations per node |

## When to Use

- **Need proven optimality:** When the rounding gap matters
- **Moderate size:** Hundreds of demands, not thousands
- **Time available:** B&P explores a tree, slower than pure CG

For most practical cutting stock problems, `solve_cg` with rounding is sufficient and faster.

## vs solve_cg

| | solve_cg | solve_bp |
|---|---|---|
| Solution | LP + rounding | Integer optimal |
| Speed | Fast | Slower |
| Gap | Usually 0-1 roll | Zero (proven) |
| Use when | Speed matters | Optimality matters |

## Tips

- **Start with `solve_cg`:** If the gap is 0, you're done
- **Limit nodes:** Use `max_nodes` to get good solutions quickly
- **Monitor progress:** Track incumbent vs bound via `on_progress`

## See Also

- [solve_cg](solve-cg.md) - Column generation without branching (faster)
- [solve_milp](solve-milp.md) - For problems without exponential variables

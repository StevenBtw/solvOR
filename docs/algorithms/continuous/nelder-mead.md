# nelder_mead

Simplex method for derivative-free optimization. Explores using a simplex (triangle in 2D) that reflects, expands, contracts around the optimum.

## Signature

```python
def nelder_mead(
    objective_fn: Callable[[Sequence[float]], float],
    x0: Sequence[float],
    *,
    max_iter: int = 10_000,
    tol: float = 1e-8,
    on_progress: Callable[[Progress], bool | None] | None = None,
    progress_interval: int = 0,
) -> Result[list[float]]
```

## Example

```python
from solvor import nelder_mead

def objective(x):
    return (x[0] - 2)**2 + (x[1] + 1)**2

result = nelder_mead(objective, x0=[0.0, 0.0])
print(result.solution)  # Close to [2, -1]
```

## When to Use

- **No gradients available**
- **Noisy or non-smooth objectives**
- **"I have no idea what this function looks like"**

## How It Works

1. Start with simplex of n+1 points
2. Order by objective value
3. Reflect worst point through centroid
4. If better, try expanding further
5. If worse, try contracting
6. Repeat until convergence

## Tips

- **Works in low dimensions.** Performance degrades above ~20D.
- **No gradients needed.** Pure function evaluation.
- **Can escape local minima.** But not guaranteed.

## See Also

- [Powell](powell.md) - Another derivative-free method
- [Bayesian Optimization](bayesian.md) - When evaluations are expensive

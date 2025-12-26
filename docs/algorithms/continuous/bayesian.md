# bayesian_opt

Bayesian optimization for expensive function evaluations. Builds a Gaussian process surrogate model to intelligently choose where to sample next.

## When to Use

- **Each evaluation is expensive** (seconds to minutes)
- **Limited budget** (10-100 evaluations)
- **Black-box function** (no gradient, no structure)
- **Hyperparameter tuning**

## Example

```python
from solvor import bayesian_opt

def expensive_fn(x):
    # Imagine this takes 10 seconds
    return (x[0] - 2)**2 + (x[1] + 1)**2

result = bayesian_opt(
    expensive_fn,
    bounds=[(-5, 5), (-5, 5)],
    max_iter=30
)
print(result.solution)  # Close to [2, -1]
print(result.evaluations)  # 30
```

## How It Works

1. Evaluate a few random points
2. Fit Gaussian process to observations
3. Use acquisition function to pick next point
4. Evaluate, update model, repeat

## Tips

- **10-100 evaluations sweet spot.** Below 10, random search is fine. Above 100, other methods catch up.
- **Low dimensions (< 20).** GP struggles in high dimensions.
- **Initial random samples.** Start with 5-10 random points.

## See Also

- [Nelder-Mead](nelder-mead.md) - When evaluations are cheap
- [Differential Evolution](differential-evolution.md) - Population-based global search

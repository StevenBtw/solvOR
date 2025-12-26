# powell

Conjugate direction method. Optimizes along each axis, then along the direction of total progress. No derivatives needed.

## Example

```python
from solvor import powell

def objective(x):
    return (x[0] - 2)**2 + (x[1] + 1)**2

result = powell(objective, x0=[0.0, 0.0])
print(result.solution)  # Close to [2, -1]
```

## With Bounds

```python
result = powell(objective, x0=[0.0, 0.0], bounds=[(-5, 5), (-5, 5)])
```

## When to Use

- Non-smooth objectives
- No gradient information
- Low to moderate dimensions

## See Also

- [Nelder-Mead](nelder-mead.md) - Alternative derivative-free method

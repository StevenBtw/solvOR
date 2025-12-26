# Gradient-Based Optimizers

Follow the slope to find the valley.

## gradient_descent

The classic. Compute gradient, step in negative direction, repeat.

```python
from solvor import gradient_descent

def grad(x):
    return [2*x[0], 2*x[1]]  # Gradient of x² + y²

result = gradient_descent(grad, x0=[5.0, 5.0], lr=0.1, max_iter=100)
print(result.solution)  # Close to [0, 0]
```

### With Line Search

```python
def objective(x):
    return x[0]**2 + x[1]**2

result = gradient_descent(
    grad, x0=[5.0, 5.0],
    line_search=True,
    objective_fn=objective
)
```

## momentum

Gradient descent with velocity. Smooths oscillations, accelerates in consistent directions.

```python
from solvor import momentum

result = momentum(grad, x0=[5.0, 5.0], lr=0.1, beta=0.9)
```

**Beta = 0.9 is typical.** Higher = more smoothing.

## rmsprop

Adapts learning rate per parameter using RMS of gradients.

```python
from solvor import rmsprop

result = rmsprop(grad, x0=[5.0, 5.0], lr=0.01, decay=0.9)
```

Useful when parameters have different gradient scales.

## adam

The "works everywhere" optimizer. Combines momentum and RMSprop.

```python
from solvor import adam

result = adam(grad, x0=[5.0, 5.0], lr=0.001)
```

### Learning Rate Schedules

```python
result = adam(
    grad, x0=[5.0, 5.0],
    lr=0.01,
    lr_schedule='cosine'  # 'constant', 'step', 'cosine', 'warmup'
)
```

## Comparison

| Optimizer | Memory | When to Use |
|-----------|--------|-------------|
| `gradient_descent` | None | Simple problems, learning |
| `momentum` | O(n) | Oscillating gradients |
| `rmsprop` | O(n) | Different scales per parameter |
| `adam` | O(n) | Default choice, almost always works |

## Tips

- **Learning rate too high:** Diverges, bounces around
- **Learning rate too low:** Slow convergence
- **Start with adam.** Default hyperparameters usually work.

## See Also

- [BFGS](bfgs.md) - Faster for smooth functions
- [Nelder-Mead](nelder-mead.md) - No gradients needed

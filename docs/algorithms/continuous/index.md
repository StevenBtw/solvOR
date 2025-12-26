# Continuous Optimization

When your objective is a smooth function and you have gradient information, you want continuous optimization. These power machine learning, curve fitting, and any problem where "take a step downhill" makes sense.

## Solvers

### Gradient-Based

| Solver | Memory | Adaptive LR? | Best For |
|--------|--------|--------------|----------|
| [`gradient_descent`](gradient.md) | None | No | Simple problems, understanding |
| [`momentum`](gradient.md) | Velocity | No | Reducing oscillations |
| [`rmsprop`](gradient.md) | Gradient squares | Yes | Different scales per parameter |
| [`adam`](gradient.md) | Velocity + gradient² | Yes | Default choice, works almost everywhere |

### Quasi-Newton

| Solver | Memory | Best For |
|--------|--------|----------|
| [`bfgs`](bfgs.md) | O(n²) | Fast convergence, smooth functions |
| [`lbfgs`](bfgs.md) | O(n × m) | Large-scale problems |

### Derivative-Free

| Solver | Best For |
|--------|----------|
| [`nelder_mead`](nelder-mead.md) | No gradients, noisy objectives |
| [`powell`](powell.md) | Non-smooth, no derivatives |
| [`bayesian_opt`](bayesian.md) | Expensive evaluations (10-100 total) |
| [`differential_evolution`](differential-evolution.md) | Global search, continuous |
| [`particle_swarm`](particle-swarm.md) | Global search, swarm intelligence |

## When to Use

**Perfect for:**

- Machine learning training
- Curve fitting and regression
- Parameter tuning for differentiable systems
- Smooth, continuous objective functions

## When NOT to Use

- **No gradient information** - Use metaheuristics
- **Discrete variables** - Use MILP, CP-SAT
- **Massively non-convex** - Start with metaheuristics, refine with gradients

## Quick Example

```python
from solvor import adam, bayesian_opt

# Adam: Minimize x² + y²
def grad(x):
    return [2*x[0], 2*x[1]]

result = adam(grad, x0=[5.0, 5.0], lr=0.1)
print(result.solution)  # Close to [0, 0]

# Bayesian optimization: Expensive black-box
def expensive_objective(x):
    return (x[0]-2)**2 + (x[1]+1)**2

result = bayesian_opt(expensive_objective, bounds=[(-5, 5), (-5, 5)], max_iter=30)
print(result.solution)  # Close to [2, -1]
```

## Rule of Thumb

Start with `adam`. If it's too complex, try `gradient_descent` with line search. If evaluations are expensive, use `bayesian_opt`.

## See Also

- [Metaheuristics](../metaheuristics/index.md) - When you don't have gradients
- [Linear Programming](../linear-programming/solve-lp.md) - For linear objectives

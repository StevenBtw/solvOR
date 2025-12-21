# Continuous Optimization

When your objective is a smooth function and you have gradient information, you want continuous optimization. These are the algorithms that power machine learning, curve fitting, and any problem where "take a step downhill" makes sense.

## Solvers in this Category

### [gradient_descent](gradient_descent)
The classic. Compute the gradient, take a step in the negative direction, repeat. Simple, interpretable, and surprisingly effective with the right learning rate.

**Complexity:** O(iterations × gradient_cost)
**Guarantees:** Converges to local minimum (global if convex)

### [momentum](momentum)
Gradient descent with memory. Accumulates a velocity vector, smoothing out oscillations and accelerating in consistent directions. Like a ball rolling downhill gaining momentum.

**Complexity:** O(iterations × gradient_cost)
**Guarantees:** Often faster convergence than vanilla GD

### [rmsprop](rmsprop)
Adapts learning rate per parameter using RMS of gradients. Parameters with large gradients get smaller steps, parameters with small gradients get larger steps.

**Complexity:** O(iterations × gradient_cost)
**Guarantees:** Better handling of different scales per dimension

### [adam](adam)
The "works everywhere" optimizer. Combines momentum and RMSprop: maintains both velocity and adaptive learning rates. Usually the default choice for deep learning.

**Complexity:** O(iterations × gradient_cost)
**Guarantees:** Robust across different problem scales

### [bayesian_opt](bayesian_opt)
For expensive black-box functions. Builds a Gaussian process surrogate model to intelligently choose where to evaluate next. Use when evaluations cost real time/money.

**Complexity:** O(iterations × dimensions²)
**Guarantees:** Sample-efficient exploration

## When to Use This Category

**Perfect for:**
- Machine learning training
- Curve fitting and regression
- Parameter tuning for differentiable systems
- Refining solutions from other methods
- Smooth, continuous objective functions
- When you have gradient information (or can compute it)

**The magic:** Follow the slope, find the valley. If the landscape is smooth enough, these methods are fast and reliable.

## When NOT to Use This Category

- **No gradient information** → Use metaheuristics (anneal, genetic)
- **Discrete variables** → Use MILP, CP-SAT, or discrete optimization
- **Non-differentiable objectives** → Use metaheuristics
- **Massively non-convex with many local minima** → Consider metaheuristics for global search, then refine with gradients

## Comparing Solvers

| Solver | Memory | Adaptive LR? | Best For |
|--------|--------|--------------|----------|
| `gradient_descent` | None | No | Simple problems, understanding, line search available |
| `momentum` | Velocity | No | Reducing oscillations, accelerating convergence |
| `rmsprop` | Gradient squares | Yes | Different scales per parameter |
| `adam` | Velocity + gradient squares | Yes | Default choice, works almost everywhere |
| `bayesian_opt` | Full history | N/A | Expensive evaluations, no gradients |

**Rule of thumb:** Start with `adam`. If it's too complex, try `gradient_descent` with line search. If evaluations are expensive, use `bayesian_opt`.

## Quick Example

```python
from solvor import gradient_descent, adam, bayesian_opt

# Gradient descent: Minimize x^2 + y^2
def grad(x):
    return [2*x[0], 2*x[1]]

result = gradient_descent(grad, x0=[5.0, 5.0], lr=0.1, max_iter=100)
print(result.solution)  # Close to [0, 0]

# Adam: Same problem (more robust to learning rate choice)
result = adam(grad, x0=[5.0, 5.0], lr=0.1)
print(result.solution)  # Close to [0, 0], often converges faster

# Bayesian optimization: Expensive black-box function
def expensive_objective(x):
    # Imagine this takes 10 seconds to evaluate
    return (x[0]-2)**2 + (x[1]+1)**2

result = bayesian_opt(expensive_objective, bounds=[(-5, 5), (-5, 5)], max_iter=30)
print(result.solution)  # Close to [2, -1] with only 30 evaluations
```

## Tips & Tricks

### Learning Rates
- **Too high:** Diverges, bounces around
- **Too low:** Slow convergence, gets stuck
- **Line search:** Let the algorithm choose the step size (available in `gradient_descent`)
- **Decay schedules:** Start high, decrease over time

### Momentum
- **Beta = 0.9 is typical** - Higher = more smoothing, lower = more responsive
- **Helps with ravines** - Long, narrow valleys where vanilla GD oscillates

### Adam
- **Default hyperparameters usually work**, (lr=0.001, beta1=0.9, beta2=0.999)
- **Bias correction matters early**, Adam corrects for zero initialization
- **Works across scales**, Adapts to each parameter's gradient history

### Bayesian Optimization
- **10-100 evaluations sweet spot**, Below 10, just use random search. Above 100, other methods catch up.
- **Low dimensions (< 20)**, Gaussian processes struggle in high dimensions
- **Initial random samples**, Start with 5-10 random points to seed the model

## Local vs Global Minima

**Warning:** All gradient-based methods find local minima, not necessarily global ones. On non-convex landscapes:

1. **Multi-start:** Run from multiple random initializations
2. **Coarse-to-fine:** Use metaheuristics for global search, then refine with gradients
3. **Convexity:** If your problem is convex, local = global

## Real-World Applications

- **Machine Learning:** Training neural networks (adam is the workhorse)
- **Curve Fitting:** Least squares regression, model calibration
- **Robotics:** Trajectory optimization, controller tuning
- **Signal Processing:** Filter design, deconvolution
- **Physics Simulations:** Parameter fitting to experimental data

## Gradient Descent Variants Not Included

This library focuses on pure Python implementations. For production ML:
- **AdaGrad:** Adaptive learning rate, rare in modern use
- **Adadelta:** Extension of AdaGrad, largely superseded by Adam
- **Nadam:** Adam + Nesterov momentum, marginal improvements
- **Heavy ball:** Classical momentum variant

Adam probably covers 80% of use cases. For the other 20%, consider specialized libraries like PyTorch or JAX.

## See Also

- [Metaheuristics](Metaheuristics) - When you don't have gradients
- [Linear & Integer Programming](Linear-&-Integer-Programming) - For linear objectives
- [Cookbook: Resource Allocation](Cookbook-Resource-Allocation) - Gradient-free optimization

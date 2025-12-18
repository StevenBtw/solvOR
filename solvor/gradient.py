"""
Gradient Descent, for smooth continuous optimization.

The idea is simple: compute the slope at your current position, take a step
downhill, repeat. The gradient tells you which direction is steepest, the
learning rate controls how big a step you take. Momentum and Adam add memory
of previous steps to avoid oscillation and adapt step sizes per dimension.

Great for refining solutions from other methods. Found a rough solution with
genetic or anneal? Use gradient descent to polish it if your objective is
differentiable. Also useful for smoothing out noisy landscapes.

    from solvor.gradient import gradient_descent, adam

    result = gradient_descent(grad_fn, x0, lr=0.01)
    result = adam(grad_fn, x0)  # adaptive learning rates, often works better

Variants:
    gradient_descent : vanilla, just follows the gradient
    momentum         : remembers previous direction, smoother convergence
    adam             : adapts learning rate per parameter, usually the default choice

Warning: gradient descent finds local minima, not global ones. For non-convex
problems, your starting point matters a lot. If you suspect multiple optima,
use anneal or genetic to explore first, then refine with gradient descent.

Don't use this for: non-differentiable functions, discrete problems, or when
you don't have access to gradients.
"""

from collections.abc import Callable, Sequence
from math import sqrt
from solvor.types import Status, Result

__all__ = ["gradient_descent", "momentum", "adam", "Status", "Result"]

def gradient_descent(
    grad_fn: Callable[[Sequence[float]], Sequence[float]],
    x0: Sequence[float],
    *,
    minimize: bool = True,
    lr: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> Result:

    sign = 1 if minimize else -1
    x = list(x0)
    n = len(x)
    evals = 0

    for iteration in range(max_iter):
        grad = grad_fn(x)
        evals += 1

        grad_norm = sqrt(sum(g * g for g in grad))
        if grad_norm < tol:
            return Result(x, grad_norm, iteration, evals, Status.OPTIMAL)

        for i in range(n):
            x[i] -= sign * lr * grad[i]

    grad_norm = sqrt(sum(g * g for g in grad_fn(x)))
    return Result(x, grad_norm, max_iter, evals + 1, Status.MAX_ITER)

def momentum(
    grad_fn: Callable[[Sequence[float]], Sequence[float]],
    x0: Sequence[float],
    *,
    minimize: bool = True,
    lr: float = 0.01,
    beta: float = 0.9,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> Result:

    sign = 1 if minimize else -1
    x = list(x0)
    n = len(x)
    v = [0.0] * n
    evals = 0

    for iteration in range(max_iter):
        grad = grad_fn(x)
        evals += 1

        grad_norm = sqrt(sum(g * g for g in grad))
        if grad_norm < tol:
            return Result(x, grad_norm, iteration, evals, Status.OPTIMAL)

        for i in range(n):
            v[i] = beta * v[i] + sign * grad[i]
            x[i] -= lr * v[i]

    grad_norm = sqrt(sum(g * g for g in grad_fn(x)))
    return Result(x, grad_norm, max_iter, evals + 1, Status.MAX_ITER)

def adam(
    grad_fn: Callable[[Sequence[float]], Sequence[float]],
    x0: Sequence[float],
    *,
    minimize: bool = True,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> Result:
    
    sign = 1 if minimize else -1
    x = list(x0)
    n = len(x)
    m = [0.0] * n
    v = [0.0] * n
    evals = 0

    for iteration in range(1, max_iter + 1):
        grad = grad_fn(x)
        evals += 1

        grad_norm = sqrt(sum(g * g for g in grad))
        if grad_norm < tol:
            return Result(x, grad_norm, iteration, evals, Status.OPTIMAL)

        for i in range(n):
            g = sign * grad[i]
            m[i] = beta1 * m[i] + (1 - beta1) * g
            v[i] = beta2 * v[i] + (1 - beta2) * g * g

            m_hat = m[i] / (1 - beta1 ** iteration)
            v_hat = v[i] / (1 - beta2 ** iteration)

            x[i] -= lr * m_hat / (sqrt(v_hat) + eps)

    grad_norm = sqrt(sum(g * g for g in grad_fn(x)))
    return Result(x, grad_norm, max_iter, evals + 1, Status.MAX_ITER)

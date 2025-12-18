"""
Gradient Descent - First-Order Continuous Optimization

Minimizes differentiable functions by iteratively moving in the direction
of steepest descent. Supports vanilla GD, momentum, and Adam variants.

Usage:
    from solvor.gradient import gradient_descent, adam, Status
    result = gradient_descent(grad_fn, x0, lr=0.01)
    result = adam(grad_fn, x0)

Parameters:
    grad_fn       : x -> gradient (function returning gradient at x)
    x0            : Initial point (list of floats)
    minimize      : True for min, False for max (default: True)
    lr            : Learning rate (default: 0.01)
    max_iter      : Maximum iterations (default: 1000)
    tol           : Stop when gradient norm < tol (default: 1e-6)

Adam additional parameters:
    beta1         : Exponential decay for first moment (default: 0.9)
    beta2         : Exponential decay for second moment (default: 0.999)
    eps           : Numerical stability (default: 1e-8)

Returns Result(solution, objective, iterations, evaluations, status)
    solution = final point (list of floats)
    objective = gradient norm at solution (0 = stationary point)
    evaluations = number of gradient evaluations

Note: For objective value, evaluate your objective function on result.solution.
These methods only use gradients, not function values.
"""

from collections import namedtuple
from collections.abc import Callable, Sequence
from enum import IntEnum, auto
from math import sqrt

__all__ = ["gradient_descent", "momentum", "adam", "Status", "Result"]

class Status(IntEnum):
    OPTIMAL = auto()
    FEASIBLE = auto()
    INFEASIBLE = auto()
    UNBOUNDED = auto()
    MAX_ITER = auto()

Result = namedtuple('Result', ['solution', 'objective', 'iterations', 'evaluations', 'status'])

def gradient_descent(
    grad_fn: Callable[[Sequence[float]], Sequence[float]],
    x0: Sequence[float],
    *,
    minimize: bool = True,
    lr: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> Result:
    """(grad_fn, x0, opts) -> Result with stationary point."""
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
    """(grad_fn, x0, opts) -> Result with stationary point using momentum."""
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
    """(grad_fn, x0, opts) -> Result with stationary point using Adam."""
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

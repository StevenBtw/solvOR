r"""
Gradient Descent, for smooth continuous optimization.

The idea is simple: compute the slope at your current position, take a step
downhill, repeat. Great for refining solutions from other methods.

    from solvor.gradient import gradient_descent, adam, rmsprop

    result = gradient_descent(grad_fn, x0, lr=0.01)
    result = gradient_descent(grad_fn, x0, objective_fn=f, line_search=True)
    result = adam(grad_fn, x0)  # adaptive learning rates, often works better

How it works: the gradient tells you which direction is steepest, the learning
rate controls how big a step you take. Momentum adds memory of previous steps
to avoid oscillation. Adam adapts step sizes per dimension using running
averages of gradients and squared gradients.

Use this for:

- Polishing solutions from other methods (genetic, anneal)
- Smooth differentiable objectives
- Machine learning model training
- Local refinement when starting point is close to optimum

Variants:

    gradient_descent: vanilla gradient following (supports line search)
    momentum: remembers previous direction, smoother convergence
    rmsprop: adapts learning rate per parameter using RMS of gradients
    adam: combines momentum + rmsprop, usually the default choice

Parameters:

    grad_fn: function returning gradient at a point
    x0: starting point
    lr: learning rate / step size
    objective_fn: optional, enables line search
    tol: convergence tolerance on gradient norm

Warning: gradient descent finds local minima, not global ones. If you suspect
multiple optima, use anneal or genetic to explore first.

Don't use this for: non-differentiable functions, discrete problems, or when
you don't have access to gradients.
"""

from collections.abc import Callable, Sequence
from math import sqrt

from solvor.types import ProgressCallback, Result, Status
from solvor.utils.helpers import report_progress

__all__ = ["gradient_descent", "momentum", "rmsprop", "adam"]


def _armijo_line_search(
    x: list[float],
    grad: Sequence[float],
    objective_fn: Callable[[Sequence[float]], float],
    sign: int,
    initial_lr: float,
    c: float = 1e-4,
    rho: float = 0.5,
    max_backtracks: int = 20,
) -> tuple[float, int]:
    f_x = objective_fn(x)
    grad_norm_sq = sum(g * g for g in grad)
    evals = 1

    lr = initial_lr
    for _ in range(max_backtracks):
        x_new = [x[i] - sign * lr * grad[i] for i in range(len(x))]
        f_new = objective_fn(x_new)
        evals += 1

        if f_new <= f_x - c * lr * grad_norm_sq:
            return lr, evals

        lr *= rho

    return lr, evals


def gradient_descent(
    grad_fn: Callable[[Sequence[float]], Sequence[float]],
    x0: Sequence[float],
    *,
    minimize: bool = True,
    lr: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-6,
    line_search: bool = False,
    objective_fn: Callable[[Sequence[float]], float] | None = None,
    on_progress: ProgressCallback | None = None,
    progress_interval: int = 0,
) -> Result:
    if line_search and objective_fn is None:
        raise ValueError("line_search=True requires objective_fn to be provided")

    sign = 1 if minimize else -1
    x = list(x0)
    n = len(x)
    evals = 0

    for iteration in range(max_iter):
        grad = grad_fn(x)
        evals += 1

        grad_norm = sqrt(sum(g * g for g in grad))
        if grad_norm < tol:
            return Result(x, grad_norm, iteration, evals)

        if line_search and objective_fn is not None:
            step, ls_evals = _armijo_line_search(x, grad, objective_fn, sign, lr)
            evals += ls_evals
            for i in range(n):
                x[i] -= sign * step * grad[i]
        else:
            for i in range(n):
                x[i] -= sign * lr * grad[i]

        if report_progress(on_progress, progress_interval, iteration + 1, grad_norm, grad_norm, evals):
            return Result(x, grad_norm, iteration + 1, evals, Status.FEASIBLE)

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
    on_progress: ProgressCallback | None = None,
    progress_interval: int = 0,
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
            return Result(x, grad_norm, iteration, evals)

        for i in range(n):
            v[i] = beta * v[i] + sign * grad[i]
            x[i] -= lr * v[i]

        if report_progress(on_progress, progress_interval, iteration + 1, grad_norm, grad_norm, evals):
            return Result(x, grad_norm, iteration + 1, evals, Status.FEASIBLE)

    grad_norm = sqrt(sum(g * g for g in grad_fn(x)))
    return Result(x, grad_norm, max_iter, evals + 1, Status.MAX_ITER)


def rmsprop(
    grad_fn: Callable[[Sequence[float]], Sequence[float]],
    x0: Sequence[float],
    *,
    minimize: bool = True,
    lr: float = 0.01,
    decay: float = 0.9,
    eps: float = 1e-8,
    max_iter: int = 1000,
    tol: float = 1e-6,
    on_progress: ProgressCallback | None = None,
    progress_interval: int = 0,
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
            return Result(x, grad_norm, iteration, evals)

        for i in range(n):
            g = sign * grad[i]
            v[i] = decay * v[i] + (1 - decay) * g * g
            x[i] -= lr * g / (sqrt(v[i]) + eps)

        if report_progress(on_progress, progress_interval, iteration + 1, grad_norm, grad_norm, evals):
            return Result(x, grad_norm, iteration + 1, evals, Status.FEASIBLE)

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
    lr_schedule: str = "constant",
    warmup_steps: int = 0,
    decay_rate: float = 0.1,
    on_progress: ProgressCallback | None = None,
    progress_interval: int = 0,
) -> Result:
    """
    Adam optimizer with optional learning rate schedules.

    lr_schedule options:
        "constant" : Fixed learning rate (default)
        "step"     : Decay by decay_rate at 50% and 75% of max_iter
        "cosine"   : Cosine annealing from lr to 0
        "warmup"   : Linear warmup for warmup_steps, then constant
    """
    if lr_schedule not in ("constant", "step", "cosine", "warmup"):
        raise ValueError(f"lr_schedule must be 'constant', 'step', 'cosine', or 'warmup', got '{lr_schedule}'")

    sign = 1 if minimize else -1
    x = list(x0)
    n = len(x)
    m = [0.0] * n
    v = [0.0] * n
    evals = 0

    from math import cos, pi

    def get_lr(t):
        if lr_schedule == "constant":
            return lr
        elif lr_schedule == "step":
            if t >= int(0.75 * max_iter):
                return lr * decay_rate * decay_rate
            elif t >= int(0.5 * max_iter):
                return lr * decay_rate
            return lr
        elif lr_schedule == "cosine":
            return lr * (1 + cos(pi * t / max_iter)) / 2
        elif lr_schedule == "warmup":
            if t < warmup_steps:
                return lr * t / max(warmup_steps, 1)
            return lr
        return lr

    for iteration in range(1, max_iter + 1):
        grad = grad_fn(x)
        evals += 1

        grad_norm = sqrt(sum(g * g for g in grad))
        if grad_norm < tol:
            return Result(x, grad_norm, iteration, evals)

        current_lr = get_lr(iteration)

        for i in range(n):
            g = sign * grad[i]
            m[i] = beta1 * m[i] + (1 - beta1) * g
            v[i] = beta2 * v[i] + (1 - beta2) * g * g

            m_hat = m[i] / (1 - beta1**iteration)
            v_hat = v[i] / (1 - beta2**iteration)

            x[i] -= current_lr * m_hat / (sqrt(v_hat) + eps)

        if report_progress(on_progress, progress_interval, iteration, grad_norm, grad_norm, evals):
            return Result(x, grad_norm, iteration, evals, Status.FEASIBLE)

    grad_norm = sqrt(sum(g * g for g in grad_fn(x)))
    return Result(x, grad_norm, max_iter, evals + 1, Status.MAX_ITER)

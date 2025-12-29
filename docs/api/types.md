# Types

Core types used across all solvOR solvers.

## Overview

Every solver returns a `Result` object containing the solution, objective value, and status information. The `Status` enum tells you whether the solve succeeded, and `Progress` is used for monitoring long-running solvers.

| Type | Purpose |
|------|---------|
| `Result[T]` | Returned by all solvers - contains solution, objective, status, iterations, evaluations |
| `Status` | Enum: OPTIMAL, FEASIBLE, INFEASIBLE, UNBOUNDED, MAX_ITER |
| `Progress` | Progress info passed to `on_progress` callbacks during optimization |
| `ProgressCallback` | Type alias for progress callback functions |

## Result

The `Result` dataclass is generic over the solution type. Common fields:

```python
result.solution    # The best solution found (type depends on solver)
result.objective   # Objective value (float)
result.iterations  # Number of solver iterations
result.evaluations # Number of objective function evaluations
result.status      # Status enum value
result.ok          # True if OPTIMAL or FEASIBLE
result.error       # Error message if failed (None on success)
result.solutions   # Multiple solutions when solution_limit > 1
```

## Status Values

| Status | Meaning |
|--------|---------|
| `OPTIMAL` | Proven optimal solution (exact solvers) |
| `FEASIBLE` | Feasible but not proven optimal (heuristics) |
| `INFEASIBLE` | No feasible solution exists |
| `UNBOUNDED` | Objective can improve infinitely |
| `MAX_ITER` | Iteration limit reached |

## Quick Usage

```python
from solvor import solve_lp, Status

result = solve_lp(c=[1, 2], A=[[1, 1]], b=[4])

if result.ok:  # True if OPTIMAL or FEASIBLE
    print(result.solution)
    print(result.objective)
elif result.status == Status.INFEASIBLE:
    print("No solution exists")
```

## Progress Callbacks

Long-running solvers support progress monitoring via callbacks:

```python
from solvor import anneal, Progress

def monitor(p: Progress) -> bool | None:
    print(f"iter {p.iteration}: obj={p.objective}, best={p.best}")
    if p.objective < 0.01:  # Early stopping
        return True
    return None

result = anneal(initial, cost_fn, neighbors, on_progress=monitor, progress_interval=100)
```

## Reference

::: solvor.types

# Contributing to solvOR

Thanks for your interest in contributing to solvor!

**Python 3.12+** is required. The project is tested on Python 3.12, 3.13, and 3.14, and developed with 3.14 in mind primarily.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/StevenBtw/solvor.git
cd solvor

# Install with dev dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Run linter
uv run ruff check solvor/
```

## Code Style

Follow the project's style:

- **Pure Python** no external dependencies
- **snake_case** everywhere
- **Type hints** on public APIs, skip for internal helpers
- **Keyword-only** for optional parameters (use `*`)
- **Minimal comments** explain *why*, not *what*
- **Sets for membership** O(1) lookup, not lists
- **Immutable state** solutions passed between iterations should be immutable; working structures can mutate

## Terminology

Use consistent naming across all solvers:

### Core Concepts

| Term | Meaning | Used In |
|------|---------|---------|
| `matrix` | Core data structure | LP, distances, graph |
| `solution` | Current candidate being evaluated | All solvers |
| `best_solution` | Best solution found so far | All solvers |
| `objective` | Value being minimized | All solvers |
| `objective_fn` | Function computing objective | Metaheuristics |
| `weight` | Coefficients/costs | Edges, constraints |
| `neighbors` | Adjacent solutions | Metaheuristics |
| `warm_start` | Initialize from previous solution | All solvers |
| `local_optimum` | Best in neighborhood, stuck | Metaheuristics |
| `degeneracy` | Stuck state / redundant solutions | LP, search stagnation |
| `minimize` | Boolean flag (default `True`) | All solvers |
| `start` / `goal` | Pathfinding endpoints | Graph solvers |
| `cost` / `weight` | Edge weights in graphs | Graph solvers |
| `n_nodes` | Graph size | Graph solvers |
| `edges` / `arcs` | Graph connections | Graph solvers |

### Metaheuristic-Specific

| Term | Meaning | Used In |
|------|---------|---------|
| `cooldown` | How long a move stays forbidden | Tabu |
| `temperature` | Acceptance probability control | SA |
| `cooling` | Rate temperature decreases | SA |

---

<details>
<summary><strong>Naming & File Conventions</strong></summary>

### Naming

```python
# snake_case everywhere
parse_input, build_graph, max_iterations

# Short names when obvious
x, y, i, j, n, m          # coordinates, indices, counts
xs, ys, vals              # plurals for collections

# Verb-based functions
solve(), build_graph(), get_neighbors()

# Predicate functions
is_valid(), can_place(), has_cycle()
```

### Imports

```python
from collections import defaultdict
from functools import cache
from itertools import pairwise, batched
from operator import attrgetter, itemgetter
from statistics import mean, stdev
from array import array
import re
```

Group: stdlib, then local. No blank lines between.

### File Naming Convention for Solvers

1. **Use snake_case** - all lowercase, underscores for spaces
2. **Algorithm names preferred** over problem names
   - Good: `dijkstra.py`, `simplex.py`, `hungarian.py`
   - Exception: Well-known problem acronyms (`sat.py`, `milp.py`)
3. **Multi-word names use underscores**
   - `bellman_ford.py`, `a_star.py`, `differential_evolution.py`
4. **Abbreviate only well-established acronyms**
   - OK: `bfs.py`, `sat.py`, `milp.py`, `bfgs.py`
   - Avoid: `diffevo.py`, `pso.py` (use `particle_swarm.py`)
5. **Group algorithm families in one file**
   - `gradient.py` contains gradient_descent, momentum, rmsprop, adam
   - Split only if file exceeds ~300 lines
6. **Primary function matches filename**
   - `anneal.py` → `anneal()`
   - Problem-based: add `solve_` prefix (`milp.py` → `solve_milp()`)

</details>

<details>
<summary><strong>Modern Python Features (3.12+)</strong></summary>

### Type Hints

Always type public APIs. Skip for internal helpers:

```python
# Public API - always type
def solve_lp(
    c: Sequence[float],
    A: Sequence[Sequence[float]],
    b: Sequence[float],
    minimize: bool = True,
) -> Result:

# Internal helpers - skip types
def _pivot(matrix, row, col, eps):
```

### Type Parameter Syntax (PEP 695)

Use inline type parameters instead of `TypeVar`:

```python
# Good: PEP 695 inline syntax
def anneal[T](initial: T, objective_fn: Callable[[T], float]) -> Result:
    ...

# Avoid: old TypeVar style
# T = TypeVar('T')
# def anneal(initial: T, objective_fn: Callable[[T], float]) -> Result:
```

### TypedDict + Unpack for **kwargs (PEP 692)

```python
from typing import TypedDict, Unpack

class SolverOpts(TypedDict, total=False):
    eps: float
    max_iter: int
    minimize: bool

def solve(c, A, b, **kwargs: Unpack[SolverOpts]) -> Result:
    ...
```

### Useful Stdlib

```python
from math import sumprod          # dot product: sumprod(c, x)
from itertools import batched     # chunk iterables: batched(items, n)
from copy import replace          # modify namedtuples: replace(result, status=NEW)

# sumprod replaces: sum(a*b for a,b in zip(c, x))
objective = sumprod(costs, quantities)

# batched replaces manual chunking
for chunk in batched(range(10000), 100):
    process(chunk)
```

### InterpreterPoolExecutor (3.14) - True Parallelism

```python
from concurrent.futures import InterpreterPoolExecutor

def evaluate_neighbors_parallel(neighbors, objective_fn):
    """Evaluate neighbors across interpreters (no GIL interference)."""
    with InterpreterPoolExecutor() as executor:
        costs = list(executor.map(objective_fn, neighbors))
    return costs
```

### Exception Groups (parallel error handling)

```python
def solve_parallel(problems):
    errors = []
    results = []
    with InterpreterPoolExecutor() as ex:
        futures = [ex.submit(solve, p) for p in problems]
        for f in futures:
            try:
                results.append(f.result())
            except Exception as e:
                errors.append(e)
    if errors:
        raise ExceptionGroup("solver failures", errors)
    return results
```

</details>

<details>
<summary><strong>Data Structures & Patterns</strong></summary>

### Comments

Minimal. Explain *why*, not *what*. No docstrings for internal functions:

```python
# Bland's rule prevents cycling
if ratio < min_ratio - eps:

# tableau stores -z, so z > 0 means -z < 0
if tab[-1][-1] < -eps:
```

### Data Structures

```python
# Dicts for graphs, state, memos
graph = {node: neighbors}
memo = {}

# Sets for membership
visited = set()
basis_set = set(basis)

# Tuples for coordinates, immutable records
point = (x, y, z)

# Lists for ordered, mutable sequences
path = [start]

# array.array for memory-critical numeric data
from array import array
tab = array('d', [0.0] * n)  # 8 bytes vs ~28 for float
```

### Frozen Dataclasses for Solutions

```python
from dataclasses import dataclass

@dataclass(slots=True, frozen=True)
class Solution:
    assignment: tuple[int, ...]  # immutable, hashable
    cost: float
```

`slots=True` cuts memory ~40%, speeds attribute access. `frozen=True` makes it hashable for sets, dict keys, tabu lists.

### Protocols for Duck Typing

```python
from typing import Protocol
from collections.abc import Iterator

class Evaluator(Protocol):
    def __call__(self, solution: Solution) -> float: ...

class Neighborhood(Protocol):
    def __call__(self, solution: Solution) -> Iterator[Solution]: ...
```

### Memoization

```python
# @cache: unbounded, use for finite problem-scoped lookups
@cache
def distance(i: int, j: int) -> float:  # problem size bounds this

# @lru_cache(maxsize=N): bounded, use when cache could grow unboundedly
@lru_cache(maxsize=10_000)
def evaluate_partial(state: tuple) -> float:  # search space is huge
```

### Generator Pipelines

```python
def neighborhood(s: Solution) -> Iterator[Move]:
    yield from swap_moves(s)
    yield from insert_moves(s)

def feasible(moves: Iterator[Move]) -> Iterator[Move]:
    return (m for m in moves if is_feasible(m))

# Lazy composition - only generates what you consume
first_valid = next(feasible(neighborhood(current)), None)
```

### Zip Tricks

```python
# Transpose (rows ↔ columns)
cols = list(zip(*rows))

# Dict from parallel sequences
mapping = dict(zip(keys, values))
```

### Match Statement for Move Types

```python
match move:
    case Swap(i, j):
        delta = calc_swap_delta(i, j)
    case Insert(src, dst):
        delta = calc_insert_delta(src, dst)
    case _:
        raise ValueError(f"Unknown move type: {move}")
```

### operator Module for Hot Paths

```python
from operator import attrgetter, itemgetter

# Instead of lambda s: s.cost
best = min(population, key=attrgetter('cost'))

# Instead of lambda x: x[1]
sorted_moves = sorted(move_values, key=itemgetter(1))
```

Faster than lambdas in tight loops.

</details>

<details>
<summary><strong>Project Idioms</strong></summary>

### Procedural Style

No class wrappers for solvers. Functions at module level:

```python
# Good: procedural flow
def parse_input(filename): ...
def build_graph(data): ...
def solve(graph): ...

result = solve(build_graph(parse_input("input.txt")))
```

### Debug via Environment

Use environment variables, not boolean parameters:

```python
from os import environ

if environ.get("DEBUG"):
    print(f"iteration {i}: {current_cost}")
```

Run with `DEBUG=1 python solve.py` to enable.

### Early Returns

Exit functions as soon as possible:

```python
def find_path(start, goal, graph):
    if start == goal:
        return [start]
    if start not in graph:
        return None
    # main logic here
```

### Walrus Operator

```python
# Regex parsing
if m := re.search(r'(\d+)', line):
    value = int(m.group(1))

# First-match search
if better := next((n for n in neighbors if cost(n) < best), None):
    best = better
```

### Consistent Graph Representation

```python
# Unweighted
graph: dict[Node, list[Node]] = {
    'a': ['b', 'c'],
    'b': ['a', 'd'],
}

# Weighted
graph: dict[Node, list[tuple[Node, float]]] = {
    'a': [('b', 1.0), ('c', 2.5)],
}
```

### Progress Callbacks

Iterative solvers support optional progress callbacks:

```python
from solvor.types import Progress

def monitor(p: Progress) -> bool | None:
    print(f"iter {p.iteration}: obj={p.objective}")
    if p.objective < 0.01:  # early stopping
        return True

result = anneal(
    initial, objective_fn, neighbors,
    on_progress=monitor,
    progress_interval=100,  # call every 100 iterations
)
```

- `on_progress`: Callback receiving `Progress`. Return `True` to stop early.
- `progress_interval`: `0` = disabled, `1` = every iteration, `N` = every Nth.

**Available in:** `anneal`, `tabu_search`, `evolve`, `differential_evolution`, `particle_swarm`, `gradient_descent`, `momentum`, `adam`, `bfgs`, `lbfgs`, `powell`, `nelder_mead`, `bayesian_opt`

### Input Validation

Use utilities from `solvor.utils.validate` for consistent error messages:

```python
from solvor.utils.validate import (
    check_positive,
    check_in_range,
    check_matrix_dims,
    check_edge_nodes,
    check_sequence_lengths,
    check_non_negative,
    warn_large_coefficients,
)

def bellman_ford(n_nodes, edges, start):
    check_positive(n_nodes, name="n_nodes")
    check_in_range(start, 0, n_nodes - 1, name="start")
    check_edge_nodes(edges, n_nodes)
    # ... solver logic
```

| Function | Purpose |
|----------|---------|
| `check_positive(val, name)` | val > 0 |
| `check_in_range(val, lo, hi, name)` | lo ≤ val ≤ hi |
| `check_matrix_dims(c, A, b)` | LP/MILP dimension consistency |
| `check_edge_nodes(edges, n_nodes)` | edge endpoints valid |
| `check_sequence_lengths(seqs, names)` | parallel sequences same length |
| `check_non_negative(seq, name)` | all values ≥ 0 |
| `warn_large_coefficients(A)` | warns if max > 1e6 |

### Result Handling

Use `.ok` for success checks, `.error` for failure context:

```python
result = solve(problem)

if result.ok:
    print(f"Solution: {result.solution}, cost: {result.objective}")

if not result.ok:
    print(f"Failed: {result.error}")

# Debug logging (only prints when DEBUG=1)
return solve(problem).log("solver: ")
```

</details>

<details>
<summary><strong>Result & Status Structure</strong></summary>

### Result Dataclass

All solvers return a frozen dataclass:

```python
@dataclass(frozen=True, slots=True)
class Result:
    solution: object
    objective: float
    iterations: int = 0
    evaluations: int = 0
    status: Status = Status.OPTIMAL
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True if solution is usable (OPTIMAL or FEASIBLE)."""
        return self.status in (Status.OPTIMAL, Status.FEASIBLE)

    def log(self, prefix: str = "") -> 'Result':
        """Print debug info if DEBUG=1. Returns self for chaining."""
```

### Status Enum

```python
class Status(IntEnum):
    OPTIMAL = auto()      # proven optimal (exact solvers)
    FEASIBLE = auto()     # feasible but not proven optimal (heuristics)
    INFEASIBLE = auto()   # no feasible solution exists
    UNBOUNDED = auto()    # objective can improve infinitely
    MAX_ITER = auto()     # iteration limit reached
```

### Usage Patterns

```python
# Success - use defaults
return Result(path, total_cost, iterations, evaluations)

# Failure - include error context
return Result(None, float('inf'), iterations, status=Status.INFEASIBLE,
              error="negative cycle detected")

# Minimize (default) vs maximize
solve_lp(c, A, b)                    # minimize c @ x
solve_lp(c, A, b, minimize=False)    # maximize c @ x
```

### Module Structure Template

```python
"""One-line description."""

from solvor.types import Result, Status

__all__ = ["solve_foo", "Result", "Status"]

EPS = 1e-10

def solve_foo(...) -> Result:
    """Brief docstring for public API."""
    ...

def _helper(...):
    # no docstring for internal functions
    ...
```

</details>

---

## Adding a New Solvor

1. Create `solvor/<solver_name>.py`
2. Import shared types: `from solvor.types import Status, Result`
3. Export `Status`, `Result`, and main solver function in `__all__`
4. Add exports to `solvor/__init__.py`
5. Create `tests/solvors/test_<solver_name>.py` with comprehensive tests
6. Add a quick example in `examples/quick_examples/<solver_name>_example.py`
7. Update `README.md` with usage examples

The CI will automatically pick up new tests - no workflow changes needed.

## Testing

Tests are organized into two categories:

```text
tests/
├── solvers/     # Solver unit tests (run by default)
│   ├── test_simplex.py
│   ├── test_milp.py
│   └── ...
└── examples/    # Example integration tests (opt-in)
    ├── test_quick_examples.py
    ├── test_puzzles.py
    ├── test_classic.py
    ├── test_graph_algorithms.py
    ├── test_linear_programming.py
    ├── test_machine_learning.py
    └── test_real_world.py
```

### Solver Tests

Each solver has its own test file. Tests should cover:
- Basic functionality
- Edge cases (empty input, infeasible, single variable, etc.)
- Minimize and maximize modes
- Parameter variations
- Stress tests with larger inputs

```bash
# Run all solver tests with coverage (default)
uv run pytest

# Run tests for a specific solver
uv run pytest tests/solvors/test_simplex.py -v --no-cov
```

### Example Tests

Example tests verify that all scripts in `examples/` run without errors. They are **disabled by default** to keep CI fast.

```bash
# Run all example tests
uv run pytest -m examples --no-cov

# Run specific example folder
uv run pytest -m quick_examples --no-cov
uv run pytest -m puzzles --no-cov
uv run pytest -m classic --no-cov
uv run pytest -m graph_algorithms --no-cov
uv run pytest -m linear_programming --no-cov
uv run pytest -m machine_learning --no-cov
uv run pytest -m real_world --no-cov
```

## Code Coverage

We maintain **88% minimum coverage** enforced by CI. Coverage runs automatically with pytest.

```bash
# Run tests with coverage (default)
uv run pytest

# Generate HTML report for detailed view
uv run pytest --cov-report=html
# Open htmlcov/index.html in browser

# Skip coverage for quick iteration
uv run pytest tests/test_simplex.py --no-cov
```

Coverage is configured in `pyproject.toml`:

- Source: `solvor/` (excludes `__init__.py`)
- Excludes: `TYPE_CHECKING` blocks, `NotImplementedError`, `pragma: no cover`

The full test suite with coverage runs on `main` branch and uploads to [Codecov](https://codecov.io).

## CI/CD

The project uses GitHub Actions (`.github/workflows/`):

**ci.yml** - Runs on push to `main`/`dev` and PRs:

- `lint` - Ruff linter
- `typecheck` - ty type checker (Python 3.12, 3.13, 3.14)
- `test-solvers` - All solver tests with coverage (88% minimum)
- `test-examples` - Example file tests

**publish.yml** - Runs on GitHub releases:

- Builds and publishes to PyPI using `uv build` and `uv publish`

## Type Checking

We use [ty](https://github.com/astral-sh/ty) for static type checking, enforced by CI.

```bash
# Run type checker
uv run ty check solvor/
```

Type hints are required on public APIs but optional for internal helpers.

## Pull Requests

1. Fork the repo
2. Create a feature branch
3. Make your changes
4. Run tests and linter
5. Update `CHANGELOG.md` under `[Unreleased]` if your changes are user-facing
6. Submit a PR with a clear description

## Changelog

We follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.

**When to update the changelog:**
- Adding new solvers or features
- Fixing bugs that affect users
- Changing public APIs
- Performance improvements

**When NOT to update:**
- Internal refactoring with no API changes
- Documentation-only changes
- Test additions/fixes

Add entries under `[Unreleased]` in the appropriate category:
- `Added` - new features
- `Changed` - changes in existing functionality
- `Fixed` - bug fixes
- `Removed` - removed features

Maintainers will move unreleased entries to a versioned section during releases.

## Philosophy

1. Working > perfect
2. Readable > clever
3. Simple > general

Any performance optimization is welcome, but not at the cost of significant complexity.

```python
model.maximize(readability + performance)
model.add(complexity <= maintainable)
```

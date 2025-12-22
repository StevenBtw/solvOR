# Contributing to solvOR

Thanks for your interest in contributing to solvor!

**Python 3.12+** is required. The project is tested on Python 3.12, 3.13, and 3.14.

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

- **Pure Python** - no external dependencies
- **snake_case** everywhere
- **Type hints** on public APIs, skip for internal helpers
- **Keyword-only** for optional parameters (use `*`)
- **Minimal comments** - explain *why*, not *what*
- **Sets for membership** - O(1) lookup, not lists
- **Immutable state** - solutions passed between iterations should be immutable; working structures can mutate

## Terminology

Use consistent naming across all solvors:

| Term | Usage |
|------|-------|
| `solution` | Current candidate being evaluated |
| `best_solution` | Best found so far |
| `objective` | Value being optimized |
| `objective_fn` | Function computing objective |
| `neighbors` | Adjacent solutions/states |
| `minimize` | Boolean flag (default `True`) |
| `start` / `goal` | Pathfinding endpoints |
| `cost` / `weight` | Edge weights in graphs |
| `n_nodes` | Graph size |
| `edges` / `arcs` | Graph connections |

## Adding a New Solvor

1. Create `solvor/<solver_name>.py`
2. Import shared types: `from solvor.types import Status, Result`
3. Export `Status`, `Result`, and main solver function in `__all__`
4. Add exports to `solvor/__init__.py`
5. Create `tests/test_<solver_name>.py` with comprehensive tests
6. Update `README.md` with usage examples
7. Add the solver to `.github/workflows/ci.yml`:
   - Add output: `<solver_name>: ${{ steps.filter.outputs.<solver_name> }}`
   - Add path filter under `filters:`:

     ```yaml
     <solver_name>:
       - 'solvor/<solver_name>.py'
       - 'tests/test_<solver_name>.py'
     ```

   - Add test job:

     ```yaml
     test-<solver_name>:
       needs: changes
       if: ${{ needs.changes.outputs.<solver_name> == 'true' || needs.changes.outputs.types == 'true' }}
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - uses: astral-sh/setup-uv@v4
         - run: uv sync --extra dev
         - run: uv run pytest tests/test_<solver_name>.py -v
     ```

## Testing

Each solver has its own test file (`tests/test_<solver_name>.py`). Tests should cover:
- Basic functionality
- Edge cases (empty input, infeasible, single variable, etc.)
- Minimize and maximize modes
- Parameter variations
- Stress tests with larger inputs

```bash
# Run all tests with coverage
uv run pytest

# Run tests for a specific solver (no coverage)
uv run pytest tests/test_simplex.py -v --no-cov
```

The CI runs tests selectively based on which files changed, only the affected solver's tests run.

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

## Type Checking

We use mypy for static type checking, enforced by CI.

```bash
# Run type checker
uv run mypy solvor/
```

Type hints are required on public APIs but optional for internal helpers. Some solvers have relaxed type checking rules configured in `pyproject.toml`.

## Pull Requests

1. Fork the repo
2. Create a feature branch
3. Make your changes
4. Run tests and linter
5. Submit a PR with a clear description

## Philosophy

1. Working > perfect
2. Readable > clever
3. Simple > general

Any performance optimization is welcome, but not at the cost of significant complexity.

```python
model.maximize(readability + performance)
model.add(complexity <= maintainable)
```

# Contributing to solvOR

Thanks for your interest in contributing to solvor!

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
| `neighbors` | Adjacent solutions (metaheuristics) |
| `minimize` | Boolean flag (default `True`) |

## Adding a New Solvor

1. Create `solvor/<solver_name>.py`
2. Import shared types: `from solvor.types import Status, Result`
3. Export `Status`, `Result`, and main solver function in `__all__`
4. Add exports to `solvor/__init__.py`
5. Create `tests/test_<solver_name>.py` with comprehensive tests
6. Update `README.md` with usage examples
7. Add the solver to `.github/workflows/ci.yml` path filters

## Testing

Each solver has its own test file (`tests/test_<solver_name>.py`). Tests should cover:
- Basic functionality
- Edge cases (empty input, infeasible, single variable, etc.)
- Minimize and maximize modes
- Parameter variations
- Stress tests with larger inputs

```bash
# Run all tests
uv run pytest tests/ -v

# Run tests for a specific solver
uv run pytest tests/test_simplex.py -v
```

The CI runs tests selectively based on which files changed - only the affected solver's tests run.

## Pull Requests

1. Fork the repo
2. Create a feature branch
3. Make your changes
4. Run tests and linter
5. Submit a PR with a clear description

## Philosophy

1. Usable > perfect
2. Readable > clever
3. Simple > general
4. Fast, but understandable
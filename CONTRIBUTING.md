# Contributing to Solvor

Thanks for your interest in contributing to Solvor!

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
uv run ruff check src/
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

Use consistent naming across all solvers:

| Term | Usage |
|------|-------|
| `solution` | Current candidate being evaluated |
| `best_solution` | Best found so far |
| `objective` | Value being optimized |
| `objective_fn` | Function computing objective |
| `neighbors` | Adjacent solutions (metaheuristics) |
| `minimize` | Boolean flag (default `True`) |

## Adding a New Solver

1. Create `src/solvor/<solver_name>.py`
2. Import shared types: `from solvor.types import Status, Result`
3. Export `Status`, `Result`, and main solver function in `__all__`
4. Add exports to `src/solvor/__init__.py`
5. Add tests to `tests/test_solvers.py`
6. Update `README.md` with usage examples

## Testing

All solvers must have tests covering:
- Basic functionality
- Edge cases (empty input, infeasible, etc.)
- Minimize and maximize modes

```bash
uv run pytest tests/ -v
```

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

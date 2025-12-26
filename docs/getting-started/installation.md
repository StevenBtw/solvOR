# Installation

## Requirements

- Python 3.12 or higher
- No external dependencies

## Install from PyPI

```bash
uv add solvor
```

Or with pip:

```bash
pip install solvor
```

## Install from source

For the latest development version:

```bash
git clone https://github.com/StevenBtw/solvOR.git
cd solvOR
uv sync
```

## Verify installation

```python
import solvor
print(solvor.__version__)
```

## Development setup

If you want to contribute or run tests:

```bash
git clone https://github.com/StevenBtw/solvOR.git
cd solvOR
uv sync --extra dev
```

This installs pytest, ruff, and type checkers.

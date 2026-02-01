"""Rust backend wrapper with fallback handling.

This module provides utilities for detecting and routing between
Python and Rust implementations of solvOR algorithms.

The main pattern is a decorator-based approach:

    from solvor._rust import with_rust_backend

    @with_rust_backend
    def my_algorithm(...) -> Result:
        # Pure Python implementation - stays readable
        ...

Rust adapters are registered separately in _rust_adapters.py.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from types import ModuleType

__all__ = ["get_backend", "get_rust_module", "rust_available", "rust_adapter", "with_rust_backend"]

_rust_available: bool | None = None
_warned = False
_logger = logging.getLogger("solvor")
_adapters: dict[str, Callable] = {}


def rust_available() -> bool:
    """Check if Rust backend is available.

    Returns:
        True if the Rust extension is installed and loadable.
    """
    global _rust_available
    if _rust_available is None:
        try:
            import solvor._solvor_rust  # noqa: F401  # ty: ignore[unresolved-import]

            _rust_available = True
        except ImportError:
            _rust_available = False
    return _rust_available


def _warn_fallback() -> None:
    """Log a warning once when falling back to Python backend."""
    global _warned
    if not _warned:
        _logger.warning(
            "Rust backend unavailable, using Python implementation. Performance may be reduced for large problems."
        )
        _warned = True


def get_backend(requested: Literal["auto", "rust", "python"] | None = None) -> Literal["rust", "python"]:
    """Determine which backend to use.

    Args:
        requested: Explicitly request "rust" or "python", or "auto" (default)
                  to automatically select based on availability.

    Returns:
        "rust" if Rust backend will be used, "python" otherwise.

    Raises:
        ImportError: If "rust" is explicitly requested but not available.
    """
    if requested == "python":
        return "python"

    if requested == "rust":
        if not rust_available():
            raise ImportError(
                "Rust backend explicitly requested but not available. "
                "Install with: pip install solvor (pre-built wheels include Rust)"
            )
        return "rust"

    # Auto mode (None or "auto")
    if rust_available():
        return "rust"

    _warn_fallback()
    return "python"


def get_rust_module() -> ModuleType:
    """Get the Rust extension module.

    Returns:
        The _solvor_rust module.

    Raises:
        ImportError: If Rust backend is not available.
    """
    if not rust_available():
        raise ImportError("Rust backend not available")
    import solvor._solvor_rust  # ty: ignore[unresolved-import]

    return solvor._solvor_rust


def rust_adapter[**P, T](name: str) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Register a Rust adapter function for an algorithm.

    Args:
        name: The name of the Python function to intercept.

    Example:
        @rust_adapter("floyd_warshall")
        def _floyd_warshall_rust(n_nodes, edges, *, directed=True):
            rust = get_rust_module()
            ...
    """

    def decorator(fn: Callable[P, T]) -> Callable[P, T]:
        _adapters[name] = fn
        return fn

    return decorator


def with_rust_backend[**P, T](fn: Callable[P, T]) -> Callable[..., T]:
    """Decorator that adds Rust backend support with minimal code changes.

    Adds a `backend` parameter to the function. Routes to registered Rust
    adapter if available and selected, otherwise calls the original Python.

    Example:
        @with_rust_backend
        def floyd_warshall(n_nodes, edges, *, directed=True) -> Result:
            # Original Python implementation - stays readable
            ...
    """

    @functools.wraps(fn)
    def wrapper(
        *args: P.args,
        backend: Literal["auto", "rust", "python"] | None = None,
        **kwargs: P.kwargs,
    ) -> T:
        selected = get_backend(backend)
        if selected == "rust":
            adapter = _adapters.get(fn.__name__)  # ty: ignore[unresolved-attribute]
            if adapter:
                return adapter(*args, **kwargs)
        return fn(*args, **kwargs)

    return wrapper

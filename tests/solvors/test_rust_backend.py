"""Tests for Rust backend routing logic."""

import pytest

from solvor._rust import get_backend, rust_available


class TestBackendRouting:
    """Test backend selection logic."""

    def test_python_backend_always_works(self):
        """backend='python' should always return 'python'."""
        assert get_backend("python") == "python"

    def test_auto_returns_valid_backend(self):
        """backend='auto' should return either 'rust' or 'python'."""
        result = get_backend("auto")
        assert result in ("rust", "python")

    def test_none_same_as_auto(self):
        """backend=None should behave like 'auto'."""
        result = get_backend(None)
        assert result in ("rust", "python")

    def test_rust_raises_if_unavailable(self):
        """backend='rust' should raise ImportError if Rust not available."""
        if rust_available():
            # Rust is available, so this should work
            assert get_backend("rust") == "rust"
        else:
            # Rust not available, should raise
            with pytest.raises(ImportError, match="Rust backend explicitly requested"):
                get_backend("rust")


class TestRustAvailability:
    """Test Rust availability detection."""

    def test_rust_available_returns_bool(self):
        """rust_available() should return a boolean."""
        result = rust_available()
        assert isinstance(result, bool)

    def test_rust_available_consistent(self):
        """rust_available() should return same value on repeated calls."""
        first = rust_available()
        second = rust_available()
        assert first == second


class TestBackendIntegration:
    """Test that backend parameter works in actual functions."""

    def test_floyd_warshall_python_backend(self):
        from solvor import floyd_warshall

        edges = [(0, 1, 1), (1, 2, 2)]
        result = floyd_warshall(3, edges, backend="python")
        assert result.solution[0][2] == 3

    def test_bellman_ford_python_backend(self):
        from solvor import bellman_ford

        edges = [(0, 1, 1), (1, 2, 2)]
        result = bellman_ford(0, edges, 3, target=2, backend="python")
        assert result.solution == [0, 1, 2]

    def test_kruskal_python_backend(self):
        from solvor import kruskal

        edges = [(0, 1, 1), (1, 2, 2)]
        result = kruskal(3, edges, backend="python")
        assert result.objective == 3

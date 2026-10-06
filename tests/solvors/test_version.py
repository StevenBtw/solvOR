"""solvor.__version__ must match the installed distribution and the Rust extension."""

import importlib.metadata
import subprocess
import sys

import pytest

import solvor
from solvor.rust import get_rust_module, rust_available


def test_version_matches_installed_distribution():
    assert solvor.__version__ == importlib.metadata.version("solvor")


def test_rust_extension_reports_the_same_version():
    if not rust_available():
        pytest.skip("Rust backend not available")
    assert get_rust_module().__version__ == solvor.__version__


def test_uninstalled_source_tree_does_not_raise():
    script = (
        "import importlib.metadata as md\n"
        "def missing(name):\n"
        "    raise md.PackageNotFoundError(name)\n"
        "md.version = missing\n"
        "import solvor\n"
        "print(solvor.__version__)\n"
    )
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert proc.stdout.strip() == "0.0.0+unknown"


def test_version_lookup_leaves_no_names_behind():
    assert "PackageNotFoundError" not in dir(solvor)
    assert "_dist_version" not in dir(solvor)

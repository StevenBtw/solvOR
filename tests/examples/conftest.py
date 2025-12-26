"""Shared fixtures for example tests."""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"

EXCLUDE = {
    "timetabling_common.py",
    "__init__.py",
}


@pytest.fixture
def run_example():
    """Fixture that returns a function to run example scripts."""

    def _run(example_path: Path):
        result = subprocess.run(
            [sys.executable, str(example_path)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=EXAMPLES_DIR.parent,
        )
        if result.returncode != 0:
            error_msg = f"Example failed with exit code {result.returncode}"
            if result.stderr:
                error_msg += f"\n\nSTDERR:\n{result.stderr}"
            if result.stdout:
                error_msg += f"\n\nSTDOUT:\n{result.stdout}"
            pytest.fail(error_msg)

    return _run

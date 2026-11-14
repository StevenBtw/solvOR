"""Test that documentation builds without errors."""

import subprocess
import sys

import pytest


@pytest.mark.docs
def test_docs_build():
    """Ensure the docs build with Zensical in strict mode.

    This catches:
    - Missing type annotations on public APIs
    - Broken internal links
    - Invalid markdown
    - Missing referenced files
    """
    result = subprocess.run(
        [sys.executable, "-m", "zensical", "build", "--strict"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        # Show the error output for debugging
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

    assert result.returncode == 0, f"zensical build --strict failed:\n{result.stderr}"

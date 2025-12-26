"""Test graph_algorithms folder."""

from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"
EXCLUDE = {"timetabling_common.py", "__init__.py"}
FOLDER = "graph_algorithms"

EXAMPLES = sorted(p for p in (EXAMPLES_DIR / FOLDER).rglob("*.py") if p.name not in EXCLUDE)


def _id(path: Path) -> str:
    return str(path.relative_to(EXAMPLES_DIR)).replace("\\", "/")


@pytest.mark.examples
@pytest.mark.graph_algorithms
@pytest.mark.parametrize("example_path", EXAMPLES, ids=[_id(p) for p in EXAMPLES])
def test_runs(example_path, run_example):
    run_example(example_path)

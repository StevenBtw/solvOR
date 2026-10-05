"""LP/MILP results must match solvor 0.6.2 on its own input form (status and optimal objective).

The fixture was recorded with solvor 0.6.2 before the bounded simplex and presolve landed:
300 random instances, dense rows, every row "<=", x >= 0, half of them binary via identity rows.
Solutions may differ where several optima exist, so only status and objective are compared.
"""

import json
import warnings
from pathlib import Path

import pytest

from solvor import solve_lp, solve_milp

SNAPSHOT = json.loads((Path(__file__).parent / "data" / "lp_milp_snapshot_0_6_2.json").read_text())


def _same(result, expected):
    status, objective = expected
    if result.status.name != status:
        return False
    return status != "OPTIMAL" or abs(result.objective - objective) <= 1e-6 * (1 + abs(objective))


@pytest.mark.parametrize("chunk", range(10))
def test_matches_0_6_2(chunk):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for inst in SNAPSHOT[chunk::10]:
            milp = solve_milp(inst["c"], inst["A"], inst["b"], inst["integers"], minimize=inst["minimize"])
            lp = solve_lp(inst["c"], inst["A"], inst["b"], minimize=inst["minimize"])
            assert _same(milp, inst["milp"]), inst
            assert _same(lp, inst["lp"]), inst

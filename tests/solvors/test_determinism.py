"""Graph results must not depend on set iteration order (PYTHONHASHSEED)."""

import os
import subprocess
import sys

from solvor.utils import canonical_order

HASH_SEED_SCRIPT = r"""
import json, random
from solvor import articulation_points, bridges, kcore_decomposition, louvain, pagerank
rng = random.Random(42)
nodes = [f"n{i}" for i in range(300)]
und = {v: set() for v in nodes}
for _ in range(900):
    a, b = rng.choice(nodes), rng.choice(nodes)
    if a != b:
        und[a].add(b)
        und[b].add(a)
node_set = set(nodes)
nb = lambda v: und[v]
print(json.dumps({
    "louvain": [sorted(c) for c in louvain(node_set, nb).solution],
    "louvain_foreign_neighbors": repr(louvain(node_set, lambda v: und[v] | {None}).objective),
    "pagerank_python": [[k, repr(v)] for k, v in pagerank(node_set, nb, backend="python").solution.items()],
    "pagerank_auto": [[k, repr(v)] for k, v in pagerank(node_set, nb).solution.items()],
    "kcore": sorted(kcore_decomposition(node_set, nb).solution.items()),
    "articulation": sorted(articulation_points(node_set, nb).solution),
    "bridges": sorted(bridges(node_set, nb).solution),
}))
"""


def _run_with_hash_seed(seed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": seed}
    proc = subprocess.run(
        [sys.executable, "-c", HASH_SEED_SCRIPT], env=env, capture_output=True, text=True, check=True, timeout=120
    )
    return proc.stdout


class TestHashSeedIndependence:
    def test_same_results_for_three_hash_seeds(self):
        outputs = [_run_with_hash_seed(seed) for seed in ("1", "2", "3")]
        assert outputs[0] == outputs[1] == outputs[2]


class TestCanonicalOrder:
    def test_sorts_comparable_items(self):
        assert canonical_order({"b", "a", "c"}) == ["a", "b", "c"]

    def test_removes_duplicates(self):
        assert canonical_order([3, 1, 3, 2, 1]) == [1, 2, 3]

    def test_keeps_first_seen_order_for_incomparable_items(self):
        assert canonical_order([2, "a", 1, "a"]) == [2, "a", 1]

    def test_accepts_generators(self):
        assert canonical_order(x % 3 for x in range(7)) == [0, 1, 2]

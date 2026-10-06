"""Solver outputs pinned bit for bit on fixed, Deriva-shaped inputs.

Downstream projects cache prompts built from these outputs (rounded scores,
percentiles, community ids, selections), so any change, even in the last bit of
a float, must be deliberate. If a test here fails after an intended change:
update its digest AND list the change under "### Result changes" in
CHANGELOG.md. Patch releases should keep every digest unchanged.

Every backend must produce the same digest (Rust is bit-identical to Python).
The inputs avoid libm functions, so digests are the same on every platform.
"""

import hashlib
import json
import random

import pytest

from solvor import (
    MilpModel,
    articulation_points,
    bridges,
    kcore_decomposition,
    louvain,
    pagerank,
    solve_lexicographic,
    solve_lp,
    solve_milp,
)
from solvor.rust import rust_available

BACKENDS = ["python", "rust"] if rust_available() else ["python"]


def _digest(value) -> str:
    """sha256 of canonical JSON; floats are written with repr, so every bit counts."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def _graph():
    """3,000 nodes: a local tree, hub edges and chords (like a code dependency graph)."""
    rng = random.Random(2026)
    n = 3000
    adj: dict[int, set[int]] = {v: set() for v in range(n)}
    for v in range(1, n):
        u = rng.randrange(max(0, v - 30), v)
        adj[u].add(v)
        adj[v].add(u)
    for _ in range(2500):
        a = rng.randrange(25) if rng.random() < 0.5 else rng.randrange(n)
        b = rng.randrange(n)
        if a != b:
            adj[a].add(b)
            adj[b].add(a)
    return {v: tuple(sorted(nbrs)) for v, nbrs in adj.items()}


GRAPH = _graph()


def _neighbors(v):
    return GRAPH[v]


class TestGraphGolden:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_pagerank(self, backend):
        result = pagerank(sorted(GRAPH), _neighbors, backend=backend)
        assert _digest([result.iterations, sorted(result.solution.items())]) == "51d9a5e9d1c80e65"

    def test_louvain(self):
        result = louvain(sorted(GRAPH), _neighbors)
        communities = [sorted(c) for c in result.solution]
        assert _digest([result.objective, result.iterations, communities]) == "0e7faafe813bdda6"

    def test_kcore(self):
        result = kcore_decomposition(set(GRAPH), _neighbors)
        assert _digest(sorted(result.solution.items())) == "ae04ba9436faa361"

    def test_articulation_points_and_bridges(self):
        points = articulation_points(set(GRAPH), _neighbors).solution
        cut_edges = bridges(set(GRAPH), _neighbors).solution
        assert _digest([sorted(points), sorted(cut_edges)]) == "06c1f7bf04cfd902"


def _knapsack(seed):
    rng = random.Random(seed)
    n = 35
    rows = [{j: float(rng.randint(5, 40)) for j in range(n)} for _ in range(4)]
    return [float(rng.randint(10, 60)) for _ in range(n)], rows, [sum(r.values()) * 0.35 for r in rows]


def _selection(seed):
    """Deriva-like selection: packing rows over 0/1 proposals, three tiers."""
    rng = random.Random(seed)
    n = 120
    rows = [{j: 1.0 for j in rng.sample(range(n), rng.randint(2, 4))} for _ in range(45)]
    tiers = [rng.choice((1, 1, 2, 3)) for _ in range(n)]
    objectives = [[1.0 if t == k else 0.0 for t in tiers] for k in (1, 2, 3)]
    objectives.append([float(n - j) for j in range(n)])  # final tie-break by rank
    return n, rows, objectives


def _assignment(seed):
    """Generalized assignment: unit rows next to capacity rows, small costs with many ties."""
    rng = random.Random(seed)
    agents, jobs = rng.randint(2, 4), rng.randint(4, 9)
    n = agents * jobs
    rows, rhs, senses = [], [], []
    for j in range(jobs):
        rows.append({a * jobs + j: 1.0 for a in range(agents)})
        rhs.append(1.0)
        senses.append("=")
    for a in range(agents):
        weights = {a * jobs + j: float(rng.randint(2, 9)) for j in range(jobs)}
        rows.append(weights)
        rhs.append(float(int(sum(weights.values()) / agents * 1.15)))
        senses.append("<=")
    return [float(rng.randint(1, 4)) for _ in range(n)], rows, rhs, senses


class TestMilpGolden:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_search_counts_of_well_scaled_models(self, backend):
        """Node and LP iteration counts too: well-scaled models must not change in any output field."""
        out = []
        for seed in range(3):
            c, rows, rhs = _knapsack(seed)
            r = solve_milp(c, rows, rhs, binary=range(len(c)), minimize=False, backend=backend)
            out.append([r.status.name, r.objective, r.solution, r.iterations, r.evaluations])
        for seed in range(3):
            n, rows, objectives = _selection(seed)
            model = MilpModel(n, binary=range(n), backend=backend)
            model.add_rows(rows, [1.0] * len(rows))
            r = solve_lexicographic(model, objectives, minimize=False)
            out.append([r.status.name, r.objective, r.solution, r.iterations, r.evaluations])
        for seed in range(20):
            c, rows, rhs, senses = _assignment(seed)
            r = solve_milp(c, rows, rhs, binary=range(len(c)), senses=senses, backend=backend)
            out.append([r.status.name, r.objective, r.solution, r.iterations, r.evaluations])
        assert _digest(out) == "6af56692afca5614"

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_knapsacks(self, backend):
        out = []
        for seed in range(3):
            c, rows, rhs = _knapsack(seed)
            result = solve_milp(c, rows, rhs, binary=range(len(c)), minimize=False, backend=backend)
            out.append([result.status.name, result.objective, result.solution, result.iterations])
        assert _digest(out) == "97b64d5baa640b0a"

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_lexicographic_selection(self, backend):
        out = []
        for seed in range(3):
            n, rows, objectives = _selection(seed)
            model = MilpModel(n, binary=range(n), backend=backend)
            model.add_rows(rows, [1.0] * len(rows))
            result = solve_lexicographic(model, objectives, minimize=False)
            out.append([result.status.name, result.objective, result.solution])
        assert _digest(out) == "e4dfa41f0d59df24"

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_lp_vertex(self, backend):
        rng = random.Random(5)
        n, m = 30, 20
        rows = [{j: float(rng.randint(1, 9)) for j in range(n) if rng.random() < 0.3} for _ in range(m)]
        c = [float(rng.randint(-9, 9)) for _ in range(n)]
        result = solve_lp(c, rows, [float(rng.randint(10, 40)) for _ in range(m)], ub=[3.0] * n, backend=backend)
        assert _digest([result.status.name, result.objective, result.solution, result.iterations]) == "41e2c4ef810d544c"

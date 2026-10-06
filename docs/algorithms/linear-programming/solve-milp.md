# solve_milp

Mixed-Integer Linear Programming. Like `solve_lp` but some variables must be integers. Uses branch-and-bound: solves LP relaxations, branches on fractional values, prunes impossible subtrees. Child nodes are re-solved with the dual simplex from their parent's basis; the search dives depth-first in the rounding direction and falls back to the best queued bound when a dive ends.

## When to Use

- Scheduling with discrete time slots
- Facility location and network design
- Set covering problems
- Any LP where some decisions are discrete (yes/no, counts)

## Signature

```python
def solve_milp(
    c: Sequence[float],
    A: Sequence[Sequence[float] | Mapping[int, float]],
    b: Sequence[float],
    integers: Sequence[int] = (),
    *,
    binary: Sequence[int] | None = None,
    senses: Sequence[str] | None = None,
    lb: Sequence[float] | None = None,
    ub: Sequence[float] | None = None,
    minimize: bool = True,
    eps: float = 1e-6,
    max_iter: int = 10_000,
    max_nodes: int = 100_000,
    gap_tol: float = 1e-6,
    warm_start: Sequence[float] | None = None,
    solution_limit: int = 1,
    heuristics: bool = True,
    lns_iterations: int = 0,
    lns_destroy_frac: float = 0.3,
    seed: int | None = None,
    backend: Literal["auto", "rust", "python"] | None = None,
) -> Result[tuple[float, ...]]
```

## Parameters

| Parameter | Description |
|-----------|-------------|
| `c` | Objective coefficients |
| `A` | Constraint rows: dense lists, or sparse `{column: coefficient}` dicts |
| `b` | Constraint right-hand sides |
| `integers` | Indices of integer variables (default none) |
| `binary` | Indices of 0/1 variables: integer with bounds [0, 1] |
| `senses` | Per row `"<="`, `">="` or `"="` (default all `"<="`) |
| `lb`, `ub` | Variable bounds (default `0` and `inf`; use `float("-inf")` for a free variable) |
| `minimize` | If False, maximize instead |
| `eps` | Numerical tolerance for integrality |
| `max_iter` | Maximum LP iterations per node |
| `max_nodes` | Maximum branch-and-bound nodes to explore |
| `gap_tol` | Stop when gap between bound and incumbent is below this |
| `warm_start` | Initial feasible solution to start from |
| `solution_limit` | Stop after finding this many solutions |
| `heuristics` | Rounding and local search for an early incumbent (default True) |
| `lns_iterations` | Large neighborhood search passes that improve the incumbent, 0 = off (default 0) |
| `lns_destroy_frac` | Fraction of the variables unfixed in each LNS pass (default 0.3) |
| `seed` | Random seed for LNS, so runs repeat exactly |
| `backend` | LP kernel: `"auto"` (default, Rust when installed), `"rust"` or `"python"`; both give identical results, bit for bit |

## Example

```python
from solvor import solve_milp

# Maximize 3x + 2y, x must be integer, subject to x + y <= 4
result = solve_milp(
    c=[-3, -2],
    A=[[1, 1]],
    b=[4],
    integers=[0],
    minimize=False
)
print(result.solution)  # (4.0, 0.0)
print(result.objective)  # 12.0
```

## Finding Multiple Solutions

```python
# Find up to 5 different solutions
result = solve_milp(c, A, b, integers=[0, 1], solution_limit=5)
if result.solutions:
    for i, sol in enumerate(result.solutions):
        print(f"Solution {i+1}: {sol}")
```

## Binary Variables

Pass `binary=` instead of adding `x_j <= 1` rows. Rows like that still work: presolve turns any single-variable row into a bound.

```python
# Pick at most one of each conflicting pair, maximize value
rows = [{0: 1, 1: 1}, {1: 1, 2: 1}]
result = solve_milp([3, 2, 3], rows, [1, 1], binary=range(3), minimize=False)
```

## Presolve

Before branching, `solve_milp` turns single-variable rows into bounds, rounds integer variables' bounds, and tightens rows whose variables and coefficients are all integers: the row is divided by the gcd of its coefficients and the right-hand side is rounded (down for `<=`, up for `>=`). A lexicographic "lock" row such as `-c1·x <= -best1 + 0.5` therefore becomes `-c1·x <= -best1`, which keeps every integer solution but gives a much tighter LP relaxation (on one 270-variable model: 1,391 nodes before, 3 after).

## Badly Scaled Models

Models whose coefficients differ by many orders of magnitude, such as big-M rows like `x <= 1e6 * z`, are hard for any floating point solver. `solve_milp` handles them in six ways:

- A row or column whose nonzero coefficients differ by a factor of 256 or more is scaled by a power of two, which brings them nearer 1 without rounding. Other rows and columns keep the factor 1.
- The LP relaxations use their own tolerance (1e-9); `eps` stays the integrality tolerance.
- In a model with such a row or column, a reduced cost within that tolerance no longer ends the LP when its column can still move far enough to gain more than `1e-9 * (1 + |objective|)`: next to a large coefficient, a slack costs almost nothing per unit but can move very far.
- In a model with such a row or column, a node whose warm-started LP looks infeasible is solved again from scratch before it is pruned: a tableau that has drifted can report a false infeasibility.
- An integer solution is accepted only after it is snapped to exact integers and every row is checked: a row holds when it is off by at most `eps * max(1, |activity|, |rhs|)`. Without the check, `x0 = 1e-6` would count as integral and the snap to 0 would move a `1e6 * x0` term by 1.
- If a node's LP solution is visibly inaccurate (outside its bounds, or integral but still violating a row), nothing that depends on it is proven: the result is `FEASIBLE` (or `INFEASIBLE`) with an explanation in `result.error`, never `OPTIMAL`.

Keep big-M values as small as the model allows; it also gives tighter LP bounds.

## Incremental Models

`MilpModel` keeps the LP between solves, so adding rows (lazy cuts, objective locks) or changing the objective re-solves warm with the dual simplex instead of starting over.

```python
from solvor import MilpModel, solve_lexicographic

model = MilpModel(n_vars=3, binary=range(3))
model.add_rows([{0: 1, 1: 1}, {1: 1, 2: 1}], [1, 1])
first = model.solve([3, 2, 3], minimize=False)    # picks x0 and x2
model.add_rows([{0: 1, 2: 1}], [1])               # lazy cut: not both
second = model.solve([3, 2, 3], minimize=False)   # warm re-solve

# Several objectives in priority order: best for c1, then best for c2 among those, ...
result = solve_lexicographic(model, [c1, c2, c3], minimize=False)
```

`solve_lexicographic` returns the last stage's result. It is `OPTIMAL` only if every stage was solved to optimality: when an earlier stage stops at a limit (`FEASIBLE`), the value it locks may not be the best, so the result is `FEASIBLE` too.

Each solve also starts from the previous solution as an incumbent when it is still feasible. Results depend only on the rows, bounds and objective you passed and the order of the calls: the same sequence of calls gives exactly the same result.

## Complexity

- **Time:** NP-hard (exponential worst case)
- **Guarantees:** Finds provably optimal integer solutions
- **Speed:** with the Rust extension the LP relaxations (the simplex tableau) run in Rust while branch and bound, presolve and heuristics stay in Python. Results are identical; LP-heavy models get 50 to 100 times faster, small models that spend their time branching 3 to 5 times. `MilpModel(..., backend=...)` takes the same keyword.

## Tips

1. **Start with LP relaxation.** Solve as LP first. If the solution is already integer, you're done. The LP objective is a bound on the optimal integer objective.
2. **Tight formulations.** Presolve already tightens all-integer rows; for rows with continuous variables, prefer the tightest valid coefficients yourself.
3. **Warm starting.** Pass a known feasible solution via `warm_start` to prune early.
4. **Gap tolerance.** For large problems, set `gap_tol=0.01` to accept solutions within 1% of optimal.
5. **Big-M values.** Use the smallest M that is valid; see Badly Scaled Models above.

## See Also

- [solve_lp](solve-lp.md) - When all variables are continuous
- [Cookbook: Resource Allocation](../../cookbook/resource-allocation.md) - MILP example

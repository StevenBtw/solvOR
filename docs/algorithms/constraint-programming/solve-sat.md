# solve_sat

Boolean satisfiability ([SAT](https://en.wikipedia.org/wiki/Boolean_satisfiability_problem)). Feed it clauses in CNF (conjunctive normal form), get back a satisfying assignment. This is the engine under the hood of constraint programming and many other solvers.

## When to Use

- Boolean constraint satisfaction
- Configuration validity checking
- Dependencies, exclusions, implications
- When your problem is naturally boolean

## Signature

```python
def solve_sat(
    clauses: Sequence[Sequence[int]],
    *,
    assumptions: Sequence[int] | None = None,
    max_conflicts: int = 100_000,
    max_restarts: int = 10_000,
    solution_limit: int = 1,
    luby_factor: int = 100,
) -> Result[dict[int, bool]]
```

## Parameters

| Parameter | Description |
|-----------|-------------|
| `clauses` | List of clauses in CNF. Each clause is a list of literals. Positive = variable, negative = NOT variable. |
| `assumptions` | Force certain literals to be true/false before solving |
| `max_conflicts` | Maximum conflicts before giving up |
| `solution_limit` | Find up to this many solutions (use `result.solutions`) |

## Example

```python
from solvor import solve_sat

# (x1 OR x2) AND (NOT x1 OR x3) AND (NOT x2 OR NOT x3)
# CNF: [[1, 2], [-1, 3], [-2, -3]]
result = solve_sat([[1, 2], [-1, 3], [-2, -3]])
print(result.solution)  # {1: True, 2: False, 3: True} or similar
```

## CNF Format

- Each clause is a disjunction (OR)
- Clauses are conjuncted (AND)
- Positive integer = variable is true
- Negative integer = variable is false

```python
# x1 AND (x2 OR x3) AND (NOT x1 OR NOT x2)
clauses = [[1], [2, 3], [-1, -2]]
```

## How It Works

**The brute force disaster:** With n variables, there are 2ⁿ possible assignments. Trying them all is hopeless for n > 30 or so.

**DPLL foundation:** The classic algorithm uses two key ideas:

1. **Unit propagation:** If a clause has only one unassigned literal, that literal must be true. This triggers a cascade—setting one variable often forces others.

2. **Pure literal elimination:** If a variable appears only positive (or only negative) across all clauses, set it to satisfy those clauses for free.

**CDCL improvement:** Modern SAT solvers add conflict-driven clause learning:

1. When you hit a conflict (a clause becomes false), analyze *why*
2. Learn a new clause that prevents this conflict pattern
3. Backjump directly to the decision that caused the conflict

**The algorithm:**

1. **Propagate:** Apply unit propagation until fixpoint
2. **Conflict?** If a clause is falsified:
   - Analyze conflict to learn a new clause
   - Backjump to the decision level that caused it
3. **Decide:** Pick an unassigned variable, guess a value
4. **Repeat** until all variables assigned (SAT) or no decisions left (UNSAT)

**Watched literals:** Instead of checking every clause after each assignment, each clause "watches" two literals. Only when a watched literal becomes false do we check if propagation is needed. This makes propagation nearly O(1) per assignment.

**Restarts:** Periodically restart the search but keep learned clauses. This escapes bad decision orderings. Luby sequence controls restart frequency.

**Why it works:** Learned clauses prune the search tree exponentially. A conflict that took 1000 decisions to find gets encoded in a clause, so you never repeat that mistake. Good variable ordering (VSIDS heuristic) focuses on "active" variables involved in recent conflicts.

## Complexity

- **Time:** NP-complete (the canonical NP-complete problem)
- **Guarantees:** Finds a solution or proves none exists

## Tips

1. **Variable numbering.** Variables must be positive integers. Use 1, 2, 3... not 0, 1, 2.
2. **Unit propagation.** The solver handles this automatically. Single-literal clauses force assignments.
3. **For complex constraints.** Use the `Model` class instead of encoding everything to CNF manually.

## See Also

- [Model (CP)](cp.md) - Higher-level constraint programming
- [solve_exact_cover](solve-exact-cover.md) - For exact cover problems

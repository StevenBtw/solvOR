# solve_sat

Boolean satisfiability. Feed it clauses in CNF (conjunctive normal form), get back a satisfying assignment. This is the engine under the hood of CP-SAT and many other solvers.

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
    max_iter: int = 1_000_000,
) -> Result[dict[int, bool]]
```

## Parameters

| Parameter | Description |
|-----------|-------------|
| `clauses` | List of clauses in CNF. Each clause is a list of literals. Positive = variable, negative = NOT variable. |
| `max_iter` | Maximum solver iterations |

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

## Complexity

- **Time:** NP-complete
- **Guarantees:** Finds a solution or proves none exists

## Tips

1. **Variable numbering.** Variables must be positive integers. Use 1, 2, 3... not 0, 1, 2.
2. **Unit propagation.** The solver handles this automatically. Single-literal clauses force assignments.
3. **For complex constraints.** Use the `Model` class (CP-SAT) instead of encoding everything to CNF manually.

## See Also

- [CP-SAT Model](cp-sat.md) - Higher-level constraint programming
- [solve_exact_cover](solve-exact-cover.md) - For exact cover problems

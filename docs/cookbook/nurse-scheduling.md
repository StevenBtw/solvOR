# Nurse Scheduling

Schedule nurses across shifts ensuring coverage and fair workloads.

A classic constraint satisfaction problem with real-world applications in healthcare, call centers, and 24/7 operations.

## The Problem

- Multiple nurses, multiple days, multiple shifts per day
- Each shift needs minimum coverage
- Nurses can't work too many shifts in a row
- Fair distribution of workload

## Approach: SAT Encoding

Model as boolean satisfiability:

- Variable `x[nurse][day][shift]` = True if nurse works that shift
- Cardinality constraints for coverage requirements
- Sequential constraints for consecutive shift limits

## Example

```python
from solvor import solve_sat

# 3 nurses, 3 days, each day needs 2 nurses
nurses, days = 3, 3

def var(n, d):
    return n * days + d + 1

clauses = []

# Each day needs at least 2 nurses
for d in range(days):
    for n1 in range(nurses):
        for n2 in range(n1+1, nurses):
            remaining = [var(n, d) for n in range(nurses) if n != n1 and n != n2]
            clauses.append([-var(n1, d), -var(n2, d)] + remaining if remaining else [var(n1, d), var(n2, d)])

result = solve_sat(clauses)
if result.status.is_success:
    for d in range(days):
        working = [n for n in range(nurses) if result.solution.get(var(n, d))]
        print(f"Day {d}: nurses {working}")
```

## Full Example

See `examples/real_world/nurse_scheduling.py` for a complete version with sequential counter constraints and workload balancing.

## Why SAT?

Hard constraints (coverage, no double shifts) encode naturally as boolean clauses. SAT solvers are highly optimized for this.

## See Also

- [solve_sat](../algorithms/constraint-programming/solve-sat.md)
- [Job Shop](job-shop.md)

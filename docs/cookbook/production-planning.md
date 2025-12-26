# Production Planning

Maximize profit by deciding how many units of each product to manufacture within resource constraints.

## The Problem

A factory produces multiple products, each requiring different amounts of resources (machine time, labor, materials). Given resource capacities and product profits, find the production mix that maximizes total profit.

## Example

```python
from solvor import solve_lp

# Products: (name, profit, machine_hours, labor_hours, material_units)
products = [
    ("Widget A", 30, 2, 3, 4),
    ("Widget B", 40, 3, 2, 5),
    ("Widget C", 25, 1, 4, 3),
    ("Gadget X", 50, 4, 3, 6),
    ("Gadget Y", 35, 2, 5, 4),
]

# Resource capacities per day
capacities = [100, 120, 150]  # machine, labor, material

n = len(products)
profits = [p[1] for p in products]
requirements = [p[2:] for p in products]

# Build constraint matrix
A = []
for j in range(3):  # For each resource
    row = [requirements[i][j] for i in range(n)]
    A.append(row)

result = solve_lp(profits, A, capacities, minimize=False)

if result.status.is_success:
    print(f"Maximum profit: ${result.objective:.2f}")
    print("\nProduction quantities:")
    for i, (name, *_) in enumerate(products):
        qty = result.solution[i]
        if qty > 0.01:
            print(f"  {name}: {qty:.2f} units")
```

**Output:**
```
Maximum profit: $1400.00

Production quantities:
  Widget B: 20.00 units
  Gadget X: 15.00 units
```

## Interpretation

- **Binding constraints** (100% utilized) are bottlenecks
- **Slack resources** indicate unused capacity
- **Shadow prices** tell you the value of additional resources

## Integer Production (MILP)

For whole units only:

```python
from solvor import solve_milp

result = solve_milp(
    c=[-p for p in profits],
    A=A,
    b=capacities,
    integers=list(range(n))
)
```

## See Also

- [Diet Problem](diet.md) - Minimum cost LP
- [Portfolio Optimization](portfolio.md)

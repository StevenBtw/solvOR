# Diet Problem

Find the cheapest combination of foods that meets nutritional requirements.

One of the earliest LP applications, studied by Stigler in 1945.

## The Problem

Given foods with costs and nutritional content, find the minimum-cost diet that meets all nutritional requirements.

## Example

```python
from solvor import solve_lp

# Foods: (cost, calories, protein, vitamin_c)
foods = [
    (2.0, 250, 8, 0),    # Bread
    (1.5, 150, 8, 2),    # Milk
    (3.0, 155, 13, 0),   # Eggs
    (0.5, 60, 1, 70),    # Orange
]

# Requirements: calories >= 2000, protein >= 50, vitamin_c >= 60
# LP form: -Ax <= -b for >= constraints

costs = [f[0] for f in foods]
A = [
    [-f[1] for f in foods],  # -calories <= -2000
    [-f[2] for f in foods],  # -protein <= -50
    [-f[3] for f in foods],  # -vitamin_c <= -60
]
b = [-2000, -50, -60]

result = solve_lp(costs, A, b)
print(f"Cost: ${result.objective:.2f}")
for i, qty in enumerate(result.solution):
    if qty > 0.01:
        print(f"  {['Bread', 'Milk', 'Eggs', 'Orange'][i]}: {qty:.2f} units")
```

## Full Example

See `examples/linear_programming/diet_problem.py` for a complete version with more foods and nutrients.

## Why LP?

Properties blend linearly by quantity. No integer constraints needed, you can eat 2.5 eggs. For realistic discrete portions, use MILP.

## See Also

- [solve_lp](../algorithms/linear-programming/solve-lp.md)
- [Production Planning](production-planning.md)

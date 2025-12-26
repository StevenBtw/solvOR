"""
0/1 Knapsack Problem with MILP

Demonstrates how knapsack can be modeled as an integer program.

Formulation:
    Variables: x[i] in {0, 1} (binary: take item or not)
    Maximize: sum(value[i] * x[i])
    Subject to: sum(weight[i] * x[i]) <= capacity

Why this approach:
    MILP is overkill for pure knapsack (use solve_knapsack instead),
    but this pattern extends to variants: multiple knapsacks, item
    dependencies, or combining with other constraints.

Expected result:
    Items 0, 1, 3 selected, total value 240, total weight 45.

Note: For practical use, prefer solve_knapsack() which uses
specialized dynamic programming - much faster and more reliable.
"""

from solvor import solve_milp

# Items: (value, weight) - 4 items for MILP demo
items = [
    (60, 10),
    (100, 20),
    (120, 30),
    (80, 15),
]

capacity = 50
n = len(items)

# Maximize sum(value[i] * x[i]) = Minimize -sum(value[i] * x[i])
c = [-float(v) for v, w in items]

# Constraints: Ax <= b
A = []
b = []

# 1. Weight capacity: sum(weight[i] * x[i]) <= capacity
A.append([float(w) for v, w in items])
b.append(float(capacity))

# 2. Upper bounds: x[i] <= 1 for all i (binary variables)
for i in range(n):
    row = [0.0] * n
    row[i] = 1.0
    A.append(row)
    b.append(1.0)

# Note: x >= 0 is implicit in MILP (simplex standard form)

# All variables are integers (binary in practice due to bounds)
integers = list(range(n))

result = solve_milp(c, A, b, integers)

if result.solution is None:
    print(f"No solution found: {result.status}")
else:
    selected = [i for i in range(n) if result.solution[i] > 0.5]
    print(f"Selected items: {selected}")
    print(f"Total value: {-result.objective:.0f}")
    print(f"Total weight: {sum(items[i][1] for i in selected)}")

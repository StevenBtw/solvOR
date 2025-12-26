"""
0/1 Knapsack Problem (Dynamic Programming)

Maximize value of items in knapsack without exceeding capacity.

Formulation:
    Given: n items with values v[i] and weights w[i], capacity C
    Find: subset S maximizing sum(v[i] for i in S)
    Subject to: sum(w[i] for i in S) <= C

Why this solver:
    The dedicated knapsack solver uses O(nC) dynamic programming.
    Exact solution, fast for reasonable capacities. For very large C,
    consider MILP or approximation algorithms.

Expected result:
    Items 1, 2, 4 selected (indices 1, 2, 4)
    Total value: 270, Total weight: 50 (exactly at capacity)
"""

from solvor import solve_knapsack

# Items: (value, weight)
items = [
    (60, 10),  # Item 0
    (100, 20),  # Item 1
    (120, 30),  # Item 2
    (80, 15),  # Item 3
    (50, 10),  # Item 4
]

capacity = 50

values = [v for v, w in items]
weights = [w for v, w in items]

result = solve_knapsack(values, weights, capacity)

print(f"Selected items: {result.solution}")
print(f"Total value: {result.objective}")
print(f"Total weight: {sum(weights[i] for i in result.solution)}")

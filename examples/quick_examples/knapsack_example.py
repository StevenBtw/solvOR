"""
Knapsack Example

Select items to maximize value while staying within capacity.
"""

from solvor import solve_knapsack

# Items: (value, weight)
values = [60, 100, 120, 80, 50]
weights = [10, 20, 30, 15, 25]
capacity = 50

result = solve_knapsack(values, weights, capacity)
print(f"Selected items: {result.solution}")
print(f"Total value: {result.objective}")

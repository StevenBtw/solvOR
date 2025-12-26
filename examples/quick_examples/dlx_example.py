"""
Dancing Links (DLX) Exact Cover Example

Find subset of rows that covers each column exactly once.
"""

from solvor import solve_exact_cover

# Matrix where 1s indicate coverage
matrix = [
    [1, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 1],
    [0, 0, 1, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 1],
    [0, 1, 0, 0, 0, 0, 1],
]

result = solve_exact_cover(matrix)
print(f"Selected rows: {result.solution}")
# Verify: selected rows cover each column exactly once

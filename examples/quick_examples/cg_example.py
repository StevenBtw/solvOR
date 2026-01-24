"""
Column Generation Example

Solve cutting stock problem: minimize rolls to cut required pieces.
"""

from solvor import solve_cg

# Classic Gilmore-Gomory cutting stock
# Roll width: 100 units
# Need: 97 pieces of 45, 610 of 36, 395 of 31, 211 of 14
result = solve_cg(
    demands=[97, 610, 395, 211],
    roll_width=100,
    piece_sizes=[45, 36, 31, 14],
)

print(f"Rolls needed: {int(result.objective)}")
print(f"Patterns used: {len(result.solution)}")
for pattern, count in sorted(result.solution.items(), key=lambda x: -x[1]):
    print(f"  {pattern} x {count}")

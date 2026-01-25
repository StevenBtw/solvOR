"""Branch-and-price for optimal integer solutions."""

from solvor import solve_bp

# Cutting stock: minimize rolls with guaranteed integer optimality
result = solve_bp(
    demands=[97, 610, 395, 211],
    roll_width=100,
    piece_sizes=[45, 36, 31, 14],
)

print(f"Status: {result.status}")
print(f"Rolls needed: {int(result.objective)}")
print(f"Nodes explored: {result.iterations}")
print(f"Patterns used:")
for pattern, count in sorted(result.solution.items()):
    print(f"  {pattern}: {count} times")

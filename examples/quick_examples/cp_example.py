"""
Constraint Programming Example

Find 3 different numbers that sum to 10.
Uses all_different and sum constraints.
"""

from solvor.cp import Model

model = Model()

# Three integer variables in range 1-9
x = model.int_var(1, 9, "x")
y = model.int_var(1, 9, "y")
z = model.int_var(1, 9, "z")

# All three must be different
model.add(model.all_different([x, y, z]))

# They must sum to 10
model.add(model.sum_eq([x, y, z], 10))

result = model.solve()
print(f"Status: {result.status.name}")
if result.ok:
    print(f"Solution: x={result.solution['x']}, y={result.solution['y']}, z={result.solution['z']}")

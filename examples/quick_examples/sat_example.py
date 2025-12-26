"""
Boolean Satisfiability (SAT) Example

Find assignment satisfying:
  (x1 OR x2) AND (NOT x1 OR x3) AND (NOT x2 OR NOT x3)

Clauses in DIMACS-like format: positive = true, negative = false
"""

from solvor import solve_sat

clauses = [
    [1, 2],  # x1 OR x2
    [-1, 3],  # NOT x1 OR x3
    [-2, -3],  # NOT x2 OR NOT x3
]

result = solve_sat(clauses)
print(f"Status: {result.status.name}")
print(f"Solution: x1={result.solution[1]}, x2={result.solution[2]}, x3={result.solution[3]}")

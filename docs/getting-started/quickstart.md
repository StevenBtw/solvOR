# Quick Start

This guide walks you through your first optimization problems with solvOR.

## Your First Linear Program

Let's solve a simple production planning problem:

> A factory makes chairs and tables. Each chair gives $3 profit, each table $2.
> You can make at most 4 items total, at most 2 chairs, and at most 3 tables.
> How many of each should you make?

```python
from solvor import solve_lp

result = solve_lp(
    c=[-3, -2],  # Negative because we minimize (so -max = min)
    A=[[1, 1], [1, 0], [0, 1]],  # Constraints: x+y <= 4, x <= 2, y <= 3
    b=[4, 2, 3]
)

print(f"Make {result.solution[0]:.0f} chairs and {result.solution[1]:.0f} tables")
print(f"Total profit: ${-result.objective:.0f}")
```

Output:
```
Make 2 chairs and 2 tables
Total profit: $10
```

## Solving a SAT Problem

Check if a boolean formula is satisfiable:

```python
from solvor import solve_sat

# (x OR y) AND (NOT x OR z) AND (NOT y OR NOT z)
clauses = [[1, 2], [-1, 3], [-2, -3]]
result = solve_sat(clauses)

if result.ok:
    print(f"Satisfiable: {result.solution}")
else:
    print("Unsatisfiable")
```

## Finding the Shortest Path

```python
from solvor import dijkstra

graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('C', 2), ('D', 5)],
    'C': [('D', 1)],
    'D': []
}

result = dijkstra('A', 'D', lambda n: graph.get(n, []))
print(f"Path: {result.solution}")  # ['A', 'B', 'C', 'D']
print(f"Distance: {result.objective}")  # 4
```

## Solving the Knapsack Problem

```python
from solvor import solve_knapsack

values = [3, 4, 5, 6]
weights = [2, 3, 4, 5]
capacity = 8

result = solve_knapsack(values, weights, capacity)
print(f"Selected items: {result.solution}")  # Indices of selected items
print(f"Total value: {result.objective}")
```

## Understanding Results

All solvers return a `Result` object with:

- `ok` - True if solution is usable (OPTIMAL or FEASIBLE)
- `status` - Status enum (OPTIMAL, FEASIBLE, INFEASIBLE, UNBOUNDED, MAX_ITER)
- `solution` - The solution (format depends on solver)
- `objective` - Objective function value (if applicable)
- `iterations` - Number of iterations/steps taken
- `error` - Error message if failed (None on success)

```python
result = solve_lp(c=[-1, -1], A=[[1, 1]], b=[10])

if result.ok:
    print(f"Solved in {result.iterations} iterations")
    print(f"Solution: {result.solution}")
else:
    print(f"Failed: {result.status}, {result.error}")
```

## Next Steps

- [Choosing a Solver](choosing-solver.md) - Pick the right algorithm for your problem
- [Examples](../examples/index.md) - See 47+ example scripts
- [Algorithm Reference](../algorithms/index.md) - Deep dive into each solver

# Troubleshooting

Common issues and how to fix them.

## Installation

### "Python version not supported"

solvOR requires **Python 3.12+**.

```bash
python --version  # Check your version
```

If you have multiple versions:

```bash
uv add solvor --python 3.14
```

### "Module not found" after installation

Make sure you're in the right environment:

```bash
uv sync
uv run python -c "import solvor; print('OK')"
```

## Solver Returns Unexpected Results

### Status is `INFEASIBLE` but I expected a solution

**For LP/MILP:**

All constraints are `Ax ≤ b`. For `>=` constraints, multiply by -1:

```python
# Want: x + y >= 4
# Wrong:
solve_lp([1, 2], [[1, 1]], [4])  # This says x + y <= 4

# Correct:
solve_lp([1, 2], [[-1, -1]], [-4])  # -x - y <= -4 means x + y >= 4
```

**For SAT/CP:**

- Print your clauses/constraints to verify encoding
- Check for conflicting constraints
- Try a simpler version first

### Status is `UNBOUNDED`

Your LP/MILP has no finite optimum:

- Missing constraints
- Wrong constraint direction
- Maximizing without upper bounds

```python
# Unbounded: maximize x with no upper limit
solve_lp([-1], [], [], minimize=False)  # No constraints!

# Fixed: add constraint
solve_lp([-1], [[1]], [10], minimize=False)  # x <= 10
```

### Status is `MAX_ITER`

Solver hit its iteration limit:

```python
# Increase iteration limit
result = solve_lp(c, A, b, max_iter=500_000)
result = anneal(initial, objective, neighbors, max_iter=200_000)
```

### Metaheuristic stuck at local optimum

**Simulated annealing:**

```python
result = anneal(
    initial, objective, neighbors,
    temperature=10000,  # Higher starting temp
    cooling=0.9999,     # Slower cooling
    max_iter=200_000
)
```

**Tabu search:**

- Increase `cooldown` to remember more moves
- Generate more diverse neighbors

**General tips:**

- Run multiple times with different seeds
- Try different starting solutions

## Numerical Issues

### "Large coefficients detected" warning

Large coefficient differences cause numerical instability:

```python
# Problematic: mixing 1e-6 and 1e6
A = [[1e-6, 1e6], [1e6, 1e-6]]

# Better: rescale your problem
```

### Gradient-based optimizer diverging

**Symptoms:** objective goes to infinity, NaN values

**Solutions:**

- Reduce learning rate: `lr=0.0001`
- Check gradient function for bugs
- Normalize input data
- Use line search: `gradient_descent(..., line_search=True, objective_fn=f)`

## Performance

### Solver is too slow

| Problem | Slow Solver | Faster Alternative |
|---------|-------------|-------------------|
| Assignment | `min_cost_flow` | `solve_hungarian` |
| Large min-cost flow | `min_cost_flow` | `network_simplex` |
| Pathfinding | `dijkstra` | `astar` with good heuristic |

**Use Python 3.14.** It's measurably faster for pure Python code.

## Common Mistakes

### Wrong input format

```python
# LP: c is 1D, A is 2D, b is 1D
solve_lp(c=[1, 2], A=[[1, 1], [2, 1]], b=[4, 5])  # Correct
solve_lp(c=[1, 2], A=[1, 1, 2, 1], b=[4, 5])      # Wrong: A must be 2D

# Pathfinding: neighbors returns iterable
dijkstra('A', 'B', lambda n: graph[n])  # Correct
dijkstra('A', 'B', graph)               # Wrong: pass function, not dict
```

### Forgetting to check `result.ok`

```python
result = solve_lp(c, A, b)

if result.ok:
    print(f"Solution: {result.solution}")
else:
    print(f"Failed: {result.status}, {result.error}")
```

### Using wrong solver

| Problem Type | Wrong Choice | Right Choice |
|-------------|--------------|--------------|
| Integer variables | `solve_lp` | `solve_milp` |
| Weighted shortest path | `bfs` | `dijkstra` |
| One-to-one matching | `min_cost_flow` | `solve_hungarian` |
| Need shortest path | `dfs` | `bfs` or `dijkstra` |

## Still Stuck?

1. Check the `examples/` folder
2. Read the solver docstring
3. Simplify your problem to a minimal example
4. Open an issue on [GitHub](https://github.com/StevenBtw/solvOR/issues)

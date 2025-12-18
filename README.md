# Solvor

![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE) 

Solvor your optimization needs..

## What's in the box?

| Category | Solvers | Use Case |
|----------|---------|----------|
| **Linear/Integer** | `solve_lp`, `solve_milp` | Resource allocation, scheduling |
| **Constraint** | `solve_sat`| Sudoku, configuration, puzzles |
| **Local Search** | `anneal`, `tabu_search` | TSP, combinatorial optimization |
| **Population** | `evolve` | When you want nature to do the work |
| **Continuous** | `gradient_descent`, `momentum`, `adam` | ML, curve fitting |
| **Black-box** | `bayesian_opt` | Hyperparameter tuning, expensive functions |
| **Graph** | `max_flow`, `min_cost_flow`, `solve_assignment` | Matching, transportation |
| **Exact Cover** | `solve_exact_cover` | Sudoku, N-Queens, tiling puzzles |

---

## Quickstart

```bash
uv add solvor
```

```python
from solvor import solve_lp, solve_tsp, anneal, Model

# Linear Programming
result = solve_lp(c=[1, 2], A=[[1, 1], [2, 1]], b=[4, 5])
print(result.solution)  # optimal x

# TSP with tabu search
distances = [[0, 10, 15], [10, 0, 20], [15, 20, 0]]
result = solve_tsp(distances)
print(result.solution)  # best tour found

# Constraint satisfaction
m = Model()
x = m.int_var(1, 9, 'x')
y = m.int_var(1, 9, 'y')
m.add(m.all_different([x, y]))
m.add(m.sum_eq([x, y], 10))
result = m.solve()
print(result.solution)  # {'x': 3, 'y': 7}
```

---

## Solvers

<details>
<summary><strong>Linear & Integer Programming</strong></summary>

### solve_lp
For resource allocation, blending, production planning. Finds the exact optimum for linear objectives with linear constraints.

```python
# minimize 2x + 3y subject to x + y >= 4, x <= 3
result = solve_lp(
    c=[2, 3],
    A=[[-1, -1], [1, 0]],  # constraints as Ax <= b
    b=[-4, 3]
)
```

### solve_milp
When some variables must be integers. Diet problems, scheduling with discrete slots, set covering.

```python
# same as above, but x must be integer
result = solve_milp(c=[2, 3], A=[[-1, -1], [1, 0]], b=[-4, 3], integers=[0])
```

</details>

<details>
<summary><strong>Constraint Programming</strong></summary>

### solve_sat
For "is this configuration valid?" problems. Dependencies, exclusions, implications - anything that boils down to boolean constraints.

```python
# (x1 OR x2) AND (NOT x1 OR x3) AND (NOT x2 OR NOT x3)
result = solve_sat([[1, 2], [-1, 3], [-2, -3]])
print(result.solution)  # {1: True, 2: False, 3: True}
```

### Model (CP-SAT)
For puzzles and scheduling with "all different", arithmetic, and logical constraints. Sudoku, N-Queens, timetabling.

```python
m = Model()
cells = [[m.int_var(1, 9, f'c{i}{j}') for j in range(9)] for i in range(9)]

# All different in each row
for row in cells:
    m.add(m.all_different(row))

result = m.solve()
```

</details>

<details>
<summary><strong>Metaheuristics</strong></summary>

### anneal
Black-box optimization that handles local optima well. Fast to prototype when you have an objective and can generate neighbors.

```python
result = anneal(
    initial=initial_solution,
    objective_fn=cost_function,
    neighbors=random_neighbor,
    temperature=1000,
    cooling=0.9995
)
```

### tabu_search
Greedy local search with memory. Prevents cycling back to recent solutions, forcing exploration of new territory. More deterministic than anneal.

```python
result = tabu_search(
    initial=initial_solution,
    objective_fn=cost_function,
    neighbors=get_neighbors,  # returns [(move, solution), ...]
    cooldown=10
)
```

### evolve
Population-based search. More overhead than anneal/tabu, but better diversity and parallelizable.

```python
result = evolve(
    objective_fn=fitness,
    population=initial_pop,
    crossover=my_crossover,
    mutate=my_mutate,
    max_gen=100
)
```

</details>

<details>
<summary><strong>Continuous Optimization</strong></summary>

### gradient_descent / momentum / adam
Follow the slope downhill. Great for polishing solutions from other methods if your objective is differentiable. Adam adapts learning rates per parameter - usually the default choice.

```python
def grad_fn(x):
    return [2 * x[0], 2 * x[1]]  # gradient of x^2 + y^2

result = adam(grad_fn, x0=[5.0, 5.0])
print(result.solution)  # [~0, ~0]
```

### bayesian_opt
When each evaluation is expensive (think hyperparameter tuning, simulations). Builds a surrogate model to guess where to sample next instead of brute-forcing.

```python
def expensive_fn(x):
    # imagine this takes 10 minutes to evaluate
    return (x[0] - 0.3)**2 + (x[1] - 0.7)**2

result = bayesian_opt(expensive_fn, bounds=[(0, 1), (0, 1)], max_iter=30)
```

</details>

<details>
<summary><strong>Network Flow</strong></summary>

### max_flow
"How much can I push through this network?" Assigning workers to tasks, finding bottlenecks. The max-flow min-cut theorem gives you bottleneck analysis for free.

```python
graph = {
    's': [('a', 10, 0), ('b', 5, 0)],
    'a': [('b', 15, 0), ('t', 10, 0)],
    'b': [('t', 10, 0)],
    't': []
}
result = max_flow(graph, 's', 't')
print(result.objective)  # total flow
print(result.solution)   # edge flows dict
```

### min_cost_flow / solve_assignment
"What's the cheapest way to route X units?" Transportation, logistics, matching with costs.

```python
# Assignment problem: 3 workers, 3 tasks
costs = [
    [10, 5, 13],
    [3, 9, 18],
    [10, 6, 12]
]
result = solve_assignment(costs)
# result.solution[i] = task assigned to worker i
# result.objective = total cost
```

</details>

<details>
<summary><strong>Exact Cover</strong></summary>

### solve_exact_cover
For "place these pieces without overlap" or "fill this grid with exactly one of each" problems. Sudoku, pentomino tiling, scheduling where every slot must be filled exactly once.

```python
# Tiling problem: cover all columns with non-overlapping rows
matrix = [
    [1, 1, 0, 0],  # row 0 covers columns 0, 1
    [0, 1, 1, 0],  # row 1 covers columns 1, 2
    [0, 0, 1, 1],  # row 2 covers columns 2, 3
    [1, 0, 0, 1],  # row 3 covers columns 0, 3
]
result = solve_exact_cover(matrix)
# result.solution = (0, 2) or (1, 3) - rows that cover all columns exactly once
```

</details>

---

## Result Format

All solvers return a consistent `Result` namedtuple:

```python
Result(
    solution,     # best solution found
    objective,    # objective value
    iterations,   # solver iterations (pivots, generations, etc.)
    evaluations,  # function evaluations
    status        # OPTIMAL, FEASIBLE, INFEASIBLE, UNBOUNDED, MAX_ITER
)
```

---

## When to use what?

| Problem | Solver |
|---------|--------|
| Linear constraints, continuous variables | `solve_lp` |
| Linear constraints, some integers | `solve_milp` |
| Boolean satisfiability | `solve_sat` |
| Discrete variables, complex constraints | `Model` |
| Combinatorial, good initial solution | `tabu_search`, `anneal` |
| Combinatorial, no clue where to start | `evolve` |
| Smooth, differentiable | `adam` |
| Expensive black-box | `bayesian_opt` |
| Assignment, matching, flow | `max_flow`, `solve_assignment` |
| Exact cover, tiling, N-Queens | `solve_exact_cover` |

---

## Philosophy

1. **Pure Python** - no numpy, no scipy, no compiled extensions
2. **Readable** - each solvor fits in one file you can actually read
3. **Consistent** - same Result format, same minimize/maximize convention
4. **Practical** - solves real problems, or AoC puzzles

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

[Apache 2.0 License](LICENSE) - free for personal and commercial use.

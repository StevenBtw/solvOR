# solvOR

[![Build Status](https://github.com/StevenBtw/solvOR/actions/workflows/ci.yml/badge.svg)](https://github.com/StevenBtw/solvOR/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/StevenBtw/solvOR/graph/badge.svg?token=A3H2COO119)](https://codecov.io/gh/StevenBtw/solvOR)
[![PyPI](https://img.shields.io/pypi/v/solvor)](https://pypi.org/project/solvor/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://pypi.org/project/solvor/)
[![Typing: ty](https://img.shields.io/badge/typing-ty-EFC621.svg)](https://github.com/astral-sh/ty)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)


Solvor all your optimization needs.

## What's in the box?

| Category | Solvors | Use Case |
|----------|---------|----------|
| **Linear/Integer** | `solve_lp`, `solve_milp` | Resource allocation, scheduling |
| **Constraint** | `solve_sat`, `Model` | Sudoku, configuration, puzzles |
| **Local Search** | `anneal`, `tabu_search` | TSP, combinatorial optimization |
| **Population** | `evolve` | When you want nature to do the work |
| **Continuous** | `gradient_descent`, `momentum`, `adam` | ML, curve fitting |
| **Black-box** | `bayesian_opt` | Hyperparameter tuning, expensive functions |
| **Pathfinding** | `bfs`, `dfs`, `dijkstra`, `astar`, `bellman_ford`, `floyd_warshall` | Shortest paths, graph traversal |
| **Graph** | `max_flow`, `min_cost_flow`, `kruskal`, `prim` | Flow, MST, connectivity |
| **Assignment** | `solve_assignment`, `hungarian`, `network_simplex` | Matching, min-cost flow |
| **Exact Cover** | `solve_exact_cover` | N-Queens, tiling puzzles |

---

## Quickstart

```bash
uv add solvor
```

```python
from solvor import solve_lp, solve_tsp, dijkstra, hungarian

# Linear Programming
result = solve_lp(c=[1, 2], A=[[1, 1], [2, 1]], b=[4, 5])
print(result.solution)  # optimal x

# TSP with tabu search
distances = [[0, 10, 15], [10, 0, 20], [15, 20, 0]]
result = solve_tsp(distances)
print(result.solution)  # best tour found

# Shortest path
graph = {'A': [('B', 1), ('C', 4)], 'B': [('C', 2)], 'C': []}
result = dijkstra('A', 'C', lambda n: graph.get(n, []))
print(result.solution)  # ['A', 'B', 'C']

# Assignment
costs = [[10, 5], [3, 9]]
result = hungarian(costs)
print(result.solution)  # [1, 0] - row 0 gets col 1, row 1 gets col 0
```

---

## Solvors

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
Simulated annealing, accepts worse solutions probabilistically.

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
<summary><strong>Pathfinding</strong></summary>

### bfs / dfs
Unweighted graph traversal. BFS finds shortest paths (fewest edges), DFS explores depth-first. Both work with any state type and neighbor function.

```python
# Find shortest path in a grid
def neighbors(pos):
    x, y = pos
    return [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

result = bfs(start=(0, 0), goal=(5, 5), neighbors=neighbors)
print(result.solution)  # path from start to goal
```

### dijkstra
Weighted shortest paths. Classic algorithm for when edges have costs. Stops early when goal is found.

```python
def neighbors(node):
    return graph[node]  # returns [(neighbor, cost), ...]

result = dijkstra(start='A', goal='Z', neighbors=neighbors)
```

### astar / astar_grid
A* with heuristics. Faster than Dijkstra when you have a good distance estimate. `astar_grid` handles 2D grids with built-in heuristics.

```python
# Grid pathfinding with obstacles
grid = [[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]]
result = astar_grid(grid, start=(0, 0), goal=(2, 3))
```

### bellman_ford
Handles negative edge weights. Slower than Dijkstra but detects negative cycles.

```python
result = bellman_ford(start=0, edges=[(0, 1, 5), (1, 2, -3), ...], n_nodes=4)
```

### floyd_warshall
All-pairs shortest paths. O(n³) but gives you every shortest path at once.

```python
result = floyd_warshall(n_nodes=4, edges=[(0, 1, 3), (1, 2, 1), ...])
# result.solution[i][j] = shortest distance from i to j
```

</details>

<details>
<summary><strong>Network Flow & MST</strong></summary>

### max_flow
"How much can I push through this network?" Assigning workers to tasks, finding bottlenecks.

```python
graph = {
    's': [('a', 10, 0), ('b', 5, 0)],
    'a': [('b', 15, 0), ('t', 10, 0)],
    'b': [('t', 10, 0)],
    't': []
}
result = max_flow(graph, 's', 't')
```

### min_cost_flow / network_simplex
"What's the cheapest way to route X units?" Use `min_cost_flow` for simplicity, `network_simplex` for large instances.

```python
# network_simplex for large min-cost flow
arcs = [(0, 1, 10, 2), (0, 2, 5, 3), (1, 2, 15, 1)]  # (from, to, cap, cost)
supplies = [10, 0, -10]  # positive = source, negative = sink
result = network_simplex(n_nodes=3, arcs=arcs, supplies=supplies)
```

### kruskal / prim
Minimum spanning tree. Connect all nodes with minimum total edge weight. Kruskal sorts edges, Prim grows from a start node.

```python
edges = [(0, 1, 4), (0, 2, 3), (1, 2, 2)]  # (u, v, weight)
result = kruskal(n_nodes=3, edges=edges)
# result.solution = [(1, 2, 2), (0, 2, 3)] - MST edges
```

</details>

<details>
<summary><strong>Assignment</strong></summary>

### solve_assignment / hungarian
Optimal one-to-one matching. `hungarian` is O(n³), direct algorithm for assignment problems.

```python
costs = [
    [10, 5, 13],
    [3, 9, 18],
    [10, 6, 12]
]
result = hungarian(costs)
# result.solution[i] = column assigned to row i
# result.objective = total cost

# For maximization
result = hungarian(costs, minimize=False)
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

All solvors return a consistent `Result` namedtuple:

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

| Problem | Solvor |
|---------|--------|
| Linear constraints, continuous variables | `solve_lp` |
| Linear constraints, some integers | `solve_milp` |
| Boolean satisfiability | `solve_sat` |
| Discrete variables, complex constraints | `Model` |
| Combinatorial, good initial solution | `tabu_search`, `anneal` |
| Combinatorial, no clue where to start | `evolve` |
| Smooth, differentiable | `adam` |
| Expensive black-box | `bayesian_opt` |
| Shortest path, unweighted | `bfs`, `dfs` |
| Shortest path, weighted | `dijkstra`, `astar` |
| Shortest path, negative weights | `bellman_ford` |
| All-pairs shortest paths | `floyd_warshall` |
| Minimum spanning tree | `kruskal`, `prim` |
| Maximum flow | `max_flow` |
| Min-cost flow, small | `min_cost_flow` |
| Min-cost flow, large | `network_simplex` |
| Assignment, matching | `hungarian`, `solve_assignment` |
| Exact cover, tiling | `solve_exact_cover` |

---

## Philosophy

1. **Pure Python:** no numpy, no scipy, no compiled extensions
2. **Readable:** each solvor fits in one file you can actually read
3. **Consistent:** same Result format, same minimize/maximize convention
4. **Practical:** solves real problems (or AoC puzzles)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

[Apache 2.0 License](LICENSE), free for personal and commercial use.

## Background of solvOR
<details>
<summary><strong>A little bit of history..</strong></summary>
I learned about solvers back in 2011, working with some great minds. I started writing python myself around 2018, always as a hobby, and in 2024 I got back into solvers for an energy management system (EMS) and wrote a few tools (simplex, milp, genetic) myself mainly to improve my understanding.

Over time this toolbox got larger and larger, so I decided to publish it on GitHub so others can use it and improve it even further. Since I am learning Rust, I will eventually replace some performance critical operations with a high performance Rust implementation. But since I work on this project (and others) in my spare time, what and when is uncertain. The name solvOR is a mix of solver(s) and OR (Operations Research).

Disclaimer; I am not a professional software engineer, so there are probably some obvious improvements possible. If so let me know, or create a PR!

</details>

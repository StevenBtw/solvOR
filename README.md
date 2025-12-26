<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/logo-light.svg">
  <img alt="solvOR" src="docs/assets/logo-light.svg" width="200">
</picture>

[![Build Status](https://github.com/StevenBtw/solvOR/actions/workflows/ci.yml/badge.svg)](https://github.com/StevenBtw/solvOR/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/StevenBtw/solvOR/graph/badge.svg?token=A3H2COO119)](https://codecov.io/gh/StevenBtw/solvOR)
[![PyPI](https://img.shields.io/pypi/v/solvor)](https://pypi.org/project/solvor/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

[![Python 3.12 | 3.13 | 3.14](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://pypi.org/project/solvor/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Typing: ty](https://img.shields.io/badge/typing-ty-EFC621.svg)](https://github.com/astral-sh/ty)

Solvor all your optimization needs.

## What's in the box?

| Category | Solvors | Use Case |
|----------|---------|----------|
| **Linear/Integer** | `solve_lp`, `solve_milp` | Resource allocation, scheduling |
| **Constraint** | `solve_sat`, `Model` | Sudoku, puzzles, and that one config problem that's been bugging you |
| **Combinatorial** | `solve_knapsack`, `solve_bin_pack`, `solve_job_shop`, `solve_vrptw` | Packing, scheduling, routing |
| **Local Search** | `anneal`, `tabu_search`, `lns`, `alns` | TSP, combinatorial optimization |
| **Population** | `evolve`, `differential_evolution`, `particle_swarm` | Global search, nature-inspired |
| **Gradient** | `gradient_descent`, `momentum`, `rmsprop`, `adam` | ML, curve fitting |
| **Quasi-Newton** | `bfgs`, `lbfgs` | Fast convergence, smooth functions |
| **Derivative-Free** | `nelder_mead`, `powell`, `bayesian_opt` | Black-box, expensive functions |
| **Pathfinding** | `bfs`, `dfs`, `dijkstra`, `astar`, `bellman_ford`, `floyd_warshall` | Shortest paths, graph traversal |
| **Graph** | `max_flow`, `min_cost_flow`, `kruskal`, `prim` | Flow, MST, connectivity |
| **Assignment** | `solve_assignment`, `solve_hungarian`, `network_simplex` | Matching, min-cost flow |
| **Exact Cover** | `solve_exact_cover` | N-Queens, tiling puzzles |
| **Utilities** | `FenwickTree`, `UnionFind` | Data structures for algorithms |

---

## Quickstart

```bash
uv add solvor
```

```python
from solvor import solve_lp, solve_tsp, dijkstra, solve_hungarian

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
result = solve_hungarian(costs)
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
Simulated annealing, sometimes you have to go downhill to find a higher peak.

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
Greedy local search with a "don't go back there" list. Simple but surprisingly effective.

```python
result = tabu_search(
    initial=initial_solution,
    objective_fn=cost_function,
    neighbors=get_neighbors,  # returns [(move, solution), ...]
    cooldown=10
)
```

### lns / alns
Large Neighborhood Search, destroy part of your solution and rebuild it better. ALNS learns which operators work best.

```python
result = lns(initial, objective_fn, destroy_ops, repair_ops, max_iter=1000)
result = alns(initial, objective_fn, destroy_ops, repair_ops, max_iter=1000)  # adaptive
```

### evolve / differential_evolution / particle_swarm
Population-based global search. Let a swarm of candidates explore for you. DE and PSO work on continuous spaces.

```python
result = evolve(objective_fn=fitness, population=pop, crossover=cx, mutate=mut)
result = differential_evolution(objective_fn, bounds=[(0, 10)] * n, pop_size=50)
result = particle_swarm(objective_fn, bounds=[(0, 10)] * n, n_particles=30)
```

</details>

<details>
<summary><strong>Continuous Optimization</strong></summary>

### Gradient-Based

`gradient_descent`, `momentum`, `rmsprop`, `adam` - follow the slope. Adam adapts learning rates per parameter.

```python
def grad_fn(x):
    return [2 * x[0], 2 * x[1]]  # gradient of x^2 + y^2

result = adam(grad_fn, x0=[5.0, 5.0])
```

### Quasi-Newton

`bfgs`, `lbfgs` - approximate Hessian for faster convergence. L-BFGS uses limited memory.

```python
result = bfgs(objective_fn, grad_fn, x0=[5.0, 5.0])
result = lbfgs(objective_fn, grad_fn, x0=[5.0, 5.0], memory=10)
```

### Derivative-Free

`nelder_mead`, `powell` - no gradients needed. Good for noisy, non-smooth, or "I have no idea what this function looks like" situations.

```python
result = nelder_mead(objective_fn, x0=[5.0, 5.0])
result = powell(objective_fn, x0=[5.0, 5.0])
```

### bayesian_opt

When each evaluation is expensive. Builds a surrogate model to sample intelligently. Perfect for hyperparameter tuning or when your simulation takes 10 minutes per run.

```python
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
Handles negative edge weights. Slower than Dijkstra but detects negative cycles, which is useful when costs can go negative.

```python
result = bellman_ford(start=0, edges=[(0, 1, 5), (1, 2, -3), ...], n_nodes=4)
```

### floyd_warshall
All-pairs shortest paths. O(n³) but gives you every shortest path at once. Worth it when you need the full picture.

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

### solve_assignment / solve_hungarian
Optimal one-to-one matching. `solve_hungarian` is O(n³), direct algorithm for assignment problems.

```python
costs = [
    [10, 5, 13],
    [3, 9, 18],
    [10, 6, 12]
]
result = solve_hungarian(costs)
# result.solution[i] = column assigned to row i
# result.objective = total cost

# For maximization
result = solve_hungarian(costs, minimize=False)
```

</details>

<details>
<summary><strong>Exact Cover</strong></summary>

### solve_exact_cover
For "place these pieces without overlap" problems. Sudoku, pentomino tiling, N-Queens, or any puzzle where you stare at a grid wondering why nothing fits.

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

<details>
<summary><strong>Combinatorial Solvers</strong></summary>

### solve_knapsack

The classic "what fits in your bag" problem. Select items to maximize value within capacity.

```python
values = [60, 100, 120]
weights = [10, 20, 30]
result = solve_knapsack(values, weights, capacity=50)
# result.solution = [1, 1, 1] - which items to take
```

### solve_bin_pack

Bin packing with First Fit Decreasing. Minimize bins needed for items.

```python
items = [4, 8, 1, 4, 2, 1]
result = solve_bin_pack(items, bin_capacity=10)
# result.solution = [[8, 2], [4, 4, 1, 1]] - items per bin
```

### solve_job_shop

Job shop scheduling. Minimize makespan for jobs on machines.

```python
# jobs[i] = [(machine, duration), ...] - operations for job i
jobs = [[(0, 3), (1, 2)], [(1, 2), (0, 4)]]
result = solve_job_shop(jobs, n_machines=2)
```

### solve_vrptw

Vehicle Routing with Time Windows. Serve customers with capacity and time constraints.

```python
from solvor import Customer, solve_vrptw

customers = [Customer(x=0, y=0, demand=0, ready=0, due=100, service=0)]  # depot
customers += [Customer(x=i, y=i, demand=10, ready=0, due=50, service=5) for i in range(1, 5)]
result = solve_vrptw(customers, vehicle_capacity=30, n_vehicles=2)
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
| Linear constraints, continuous | `solve_lp` |
| Linear constraints, integers | `solve_milp` |
| Boolean satisfiability | `solve_sat` |
| Discrete vars, complex constraints | `Model` |
| Knapsack, subset selection | `solve_knapsack` |
| Bin packing | `solve_bin_pack` |
| Job shop scheduling | `solve_job_shop` |
| Vehicle routing | `solve_vrptw` |
| Combinatorial, local search | `tabu_search`, `anneal` |
| Combinatorial, destroy-repair | `lns`, `alns` |
| Global search, continuous | `differential_evolution`, `particle_swarm` |
| Global search, discrete | `evolve` |
| Smooth, differentiable, fast | `bfgs`, `lbfgs` |
| Smooth, differentiable, ML | `adam`, `rmsprop` |
| Non-smooth, no gradients | `nelder_mead`, `powell` |
| Expensive black-box | `bayesian_opt` |
| Shortest path, unweighted | `bfs`, `dfs` |
| Shortest path, weighted | `dijkstra`, `astar` |
| Shortest path, negative weights | `bellman_ford` |
| All-pairs shortest paths | `floyd_warshall` |
| Minimum spanning tree | `kruskal`, `prim` |
| Maximum flow | `max_flow` |
| Min-cost flow | `min_cost_flow`, `network_simplex` |
| Assignment, matching | `solve_hungarian`, `solve_assignment` |
| Exact cover, tiling | `solve_exact_cover` |

---

## Philosophy

1. **Pure Python:** no numpy, no scipy, no compiled extensions. Copy it, change it, break it, learn from it.
2. **Readable:** each solvor fits in one file you can actually read
3. **Consistent:** same Result format, same minimize/maximize convention
4. **Practical:** solves real problems (and AoC puzzles, obviously)

---

## Documentation

Full docs at **[solver.ai](https://solver.ai)** - getting started, algorithm reference, cookbook with worked examples, and troubleshooting.

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

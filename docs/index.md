# solvOR { .visually-hidden }

![solvOR logo](assets/logo.svg){ .logo-hero }

**Solvor all your optimization needs.**

solvOR is a pure Python optimization library. No numpy, no scipy, no compiled extensions. Each solver fits in one file you can actually read.

[![Build Status](https://github.com/StevenBtw/solvOR/actions/workflows/ci.yml/badge.svg)](https://github.com/StevenBtw/solvOR/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/StevenBtw/solvOR/branch/main/graph/badge.svg)](https://codecov.io/gh/StevenBtw/solvOR)
[![PyPI](https://img.shields.io/pypi/v/solvor.svg)](https://pypi.org/project/solvor/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Quick Start

```bash
uv add solvor
```

```python
from solvor import solve_lp, dijkstra, solve_hungarian

# Linear programming
result = solve_lp(c=[1, 2], A=[[1, 1]], b=[4])
print(result.solution)

# Shortest path
graph = {'A': [('B', 1), ('C', 4)], 'B': [('C', 2)], 'C': []}
result = dijkstra('A', 'C', lambda n: graph.get(n, []))
print(result.solution)  # ['A', 'B', 'C']

# Assignment
costs = [[10, 5], [3, 9]]
result = solve_hungarian(costs)
print(result.solution)  # [1, 0]
```

## What's in the box?

| Category | Solvers | Use Case |
|----------|---------|----------|
| **Linear/Integer** | `solve_lp`, `solve_milp` | Resource allocation, scheduling |
| **Constraint** | `solve_sat`, `Model` | Sudoku, puzzles, config problems |
| **Combinatorial** | `solve_knapsack`, `solve_bin_pack`, `solve_job_shop`, `solve_vrptw` | Packing, scheduling, routing |
| **Local Search** | `anneal`, `tabu_search`, `lns`, `alns` | TSP, combinatorial optimization |
| **Population** | `evolve`, `differential_evolution`, `particle_swarm` | Global search |
| **Gradient** | `gradient_descent`, `momentum`, `rmsprop`, `adam` | ML, curve fitting |
| **Quasi-Newton** | `bfgs`, `lbfgs` | Fast convergence, smooth functions |
| **Derivative-Free** | `nelder_mead`, `powell`, `bayesian_opt` | Black-box, expensive functions |
| **Pathfinding** | `bfs`, `dfs`, `dijkstra`, `astar`, `bellman_ford`, `floyd_warshall` | Shortest paths |
| **Graph** | `max_flow`, `min_cost_flow`, `kruskal`, `prim` | Flow, MST |
| **Assignment** | `solve_hungarian`, `solve_assignment` | Matching |

## When to use what?

**I need the optimal solution and...**

- My constraints are linear → `solve_lp`
- Some variables must be integers → `solve_milp`
- It's all boolean logic → `solve_sat`
- I have complex constraints → `Model` (CP-SAT)

**Good enough is fine...**

- I have a decent starting point → `tabu_search` or `anneal`
- Continuous, no gradients → `particle_swarm` or `nelder_mead`
- My function is smooth → `adam`, `bfgs`
- Each evaluation is expensive → `bayesian_opt`

**I need a path through a graph...**

- Unweighted → `bfs` or `dfs`
- Weighted, non-negative → `dijkstra` or `astar`
- Negative edge weights → `bellman_ford`

## Philosophy

1. **Pure Python** - no numpy, no scipy, runs anywhere Python runs
2. **Readable** - each solver fits in one file you can actually read
3. **Consistent** - same Result format, same conventions across all solvers
4. **Practical** - solves real problems and AoC puzzles, obviously

Working > perfect. Readable > clever. Simple > general.

[Get Started](getting-started/installation.md){ .md-button .md-button--primary }
[Examples](examples/index.md){ .md-button }

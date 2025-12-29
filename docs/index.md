---
title: solvOR - Pure Python Optimization Library
description: Learn optimization by reading the code. 40+ algorithms, zero dependencies. Each solver fits in one readable file.
---

![solvOR logo](assets/logo.svg){ .logo-hero }

**Solvor all your optimization needs.**

solvOR is a pure Python optimization library designed for learning and experimentation. No numpy, no scipy, no compiled extensions. Each algorithm fits in one readable file you can study, modify, and extend.

**Learn optimization by reading the code.**

[![Build Status](https://github.com/StevenBtw/solvOR/actions/workflows/ci.yml/badge.svg)](https://github.com/StevenBtw/solvOR/actions/workflows/ci.yml)
[![Docs](https://github.com/StevenBtw/solvOR/actions/workflows/docs.yml/badge.svg)](https://solvOR.ai)
[![codecov](https://codecov.io/gh/StevenBtw/solvOR/graph/badge.svg?token=A3H2COO119)](https://codecov.io/gh/StevenBtw/solvOR)
[![PyPI](https://img.shields.io/pypi/v/solvOR)](https://pypi.org/project/solvOR/)
[![Downloads](https://img.shields.io/pypi/dm/solvOR)](https://pypi.org/project/solvOR/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Python 3.12 | 3.13 | 3.14](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://pypi.org/project/solvOR/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://docs.astral.sh/uv/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://docs.astral.sh/ruff/)
[![Typing: ty](https://img.shields.io/badge/typing-ty-EFC621.svg)](https://docs.astral.sh/ty/)

---

## Why solvOR?

<div class="grid cards" markdown>

-   :material-book-open-variant:{ .lg .middle } **Readable Code**

    ---

    Every algorithm is implemented in clear, documented Python. No magic, no hidden complexity. Perfect for understanding how solvers actually work.

-   :material-school:{ .lg .middle } **Educational First**

    ---

    Designed for students and learners. Each solver includes Wikipedia links, complexity analysis, and practical tips. Learn the theory by reading working code.

-   :material-wrench:{ .lg .middle } **Easy to Customize**

    ---

    Want to add a custom heuristic? Modify the neighborhood function? Everything is accessible Python you can copy, tweak, and experiment with.

-   :material-lightning-bolt:{ .lg .middle } **Practical Performance**

    ---

    While not as fast as [OR-Tools](https://developers.google.com/optimization) on large instances, solvOR handles real problems effectively. Solves Sudoku instantly, TSP-100 in seconds, and scales to thousands of variables for many problem types.

</div>

---

## Quick Start

```bash
uv add solvor
```

=== "Linear Programming"

    ```python
    from solvor import solve_lp

    # Minimize 2x + 3y subject to x + y >= 4
    result = solve_lp(c=[2, 3], A=[[-1, -1]], b=[-4])
    print(result.solution)  # Optimal x, y
    print(result.objective) # Minimum cost
    ```

=== "Pathfinding"

    ```python
    from solvor import dijkstra

    graph = {'A': [('B', 1), ('C', 4)], 'B': [('C', 2)], 'C': []}
    result = dijkstra('A', 'C', lambda n: graph.get(n, []))
    print(result.solution)  # ['A', 'B', 'C']
    print(result.objective) # 3 (total distance)
    ```

=== "Constraint Programming"

    ```python
    from solvor import Model

    m = Model()
    x, y, z = [m.int_var(1, 9, name) for name in 'xyz']
    m.add(m.all_different([x, y, z]))
    m.add(x + y + z == 15)
    result = m.solve()
    print(result.solution)  # {'x': 3, 'y': 5, 'z': 7}
    ```

=== "Metaheuristics"

    ```python
    from solvor import solve_tsp

    distances = [
        [0, 29, 20, 21],
        [29, 0, 15, 17],
        [20, 15, 0, 28],
        [21, 17, 28, 0]
    ]
    result = solve_tsp(distances)
    print(result.solution)  # Best tour found
    print(result.objective) # Tour length
    ```

---

## What's in the Box?

| Category | Algorithms | Learn About |
|----------|------------|-------------|
| **Linear Programming** | `solve_lp`, `solve_milp` | [Simplex method](https://en.wikipedia.org/wiki/Simplex_algorithm), branch and bound |
| **Constraint Programming** | `solve_sat`, `Model`, `solve_exact_cover` | [SAT solving](https://en.wikipedia.org/wiki/Boolean_satisfiability_problem), [Dancing Links](https://en.wikipedia.org/wiki/Dancing_Links) |
| **Metaheuristics** | `anneal`, `tabu_search`, `lns`, `evolve` | [Local search](https://en.wikipedia.org/wiki/Local_search_(optimization)), [genetic algorithms](https://en.wikipedia.org/wiki/Genetic_algorithm) |
| **Continuous Optimization** | `adam`, `bfgs`, `nelder_mead`, `bayesian_opt` | [Gradient descent](https://en.wikipedia.org/wiki/Gradient_descent), [quasi-Newton methods](https://en.wikipedia.org/wiki/Quasi-Newton_method) |
| **Graph Algorithms** | `dijkstra`, `astar`, `bellman_ford`, `floyd_warshall` | [Shortest paths](https://en.wikipedia.org/wiki/Shortest_path_problem), [A* search](https://en.wikipedia.org/wiki/A*_search_algorithm) |
| **Graph Analysis** | `topological_sort`, `scc`, `pagerank`, `louvain`, `kcore` | [Dependency order](https://en.wikipedia.org/wiki/Topological_sorting), [centrality](https://en.wikipedia.org/wiki/PageRank), [community detection](https://en.wikipedia.org/wiki/Louvain_method) |
| **Network Flow** | `max_flow`, `min_cost_flow`, `kruskal`, `prim` | [Ford-Fulkerson](https://en.wikipedia.org/wiki/Ford%E2%80%93Fulkerson_algorithm), [MST algorithms](https://en.wikipedia.org/wiki/Minimum_spanning_tree) |
| **Combinatorial** | `solve_knapsack`, `solve_bin_pack`, `solve_job_shop` | [Dynamic programming](https://en.wikipedia.org/wiki/Dynamic_programming), scheduling |
| **Assignment** | `solve_hungarian`, `network_simplex` | [Hungarian algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm) |

---

## Choosing an Algorithm

**I need a provably optimal solution:**

- Linear constraints, continuous variables → [`solve_lp`](algorithms/linear-programming/solve-lp.md)
- Integer or binary variables → [`solve_milp`](algorithms/linear-programming/solve-milp.md)
- Boolean satisfiability → [`solve_sat`](algorithms/constraint-programming/solve-sat.md)
- Complex constraints (all-different, etc.) → [`Model`](algorithms/constraint-programming/cp.md)

**A good solution quickly is fine:**

- Have a starting point → [`tabu_search`](algorithms/metaheuristics/tabu.md) or [`anneal`](algorithms/metaheuristics/anneal.md)
- Continuous, no gradients → [`nelder_mead`](algorithms/continuous/nelder-mead.md)
- Expensive evaluations → [`bayesian_opt`](algorithms/continuous/bayesian.md)

**Graph or network problem:**

- Shortest path → [`dijkstra`](algorithms/graph/shortest-paths.md), [`astar`](algorithms/graph/shortest-paths.md), [`bellman_ford`](algorithms/graph/shortest-paths.md)
- Maximum flow → [`max_flow`](algorithms/graph/network-flow.md)
- Spanning tree → [`kruskal`](algorithms/graph/mst.md), [`prim`](algorithms/graph/mst.md)
- Dependencies → [`topological_sort`](algorithms/graph/dependency-analysis.md), [`strongly_connected_components`](algorithms/graph/dependency-analysis.md)
- Node importance → [`pagerank`](algorithms/graph/centrality.md), [`kcore`](algorithms/graph/centrality.md)
- Communities → [`louvain`](algorithms/graph/community-detection.md)

[:material-arrow-right: Full decision guide](getting-started/choosing-solver.md)

---

## Popular Examples

<div class="grid cards" markdown>

-   **[Sudoku Solver](cookbook/sudoku.md)**

    ---

    Constraint programming with `all_different`. Classic puzzle, elegant solution.

-   **[Traveling Salesman](cookbook/tsp.md)**

    ---

    Tabu search for TSP. See how local search escapes local optima.

-   **[Shortest Path Grid](cookbook/shortest-path-grid.md)**

    ---

    A* pathfinding on a 2D grid. Understand heuristics in action.

-   **[N-Queens](cookbook/n-queens.md)**

    ---

    Place N queens with no conflicts. A constraint satisfaction classic.

</div>

[:material-arrow-right: All cookbook examples](cookbook/index.md) · [:material-arrow-right: More examples](examples/index.md)

---

## Philosophy

!!! quote "Design principles"

    1. **Working > perfect** — Ship code that solves problems
    2. **Readable > clever** — Tomorrow-you needs to understand today-you's code
    3. **Simple > general** — Solve the problem at hand, not all possible problems

solvOR is not trying to compete with production solvers like OR-Tools, Gurobi, or CPLEX. It's designed to help you **understand** how these algorithms work, so you can make informed decisions about which tools to use and how to apply them.

---

[Get Started](getting-started/installation.md){ .md-button .md-button--primary }
[Browse Examples](examples/index.md){ .md-button }
[API Reference](api/index.md){ .md-button }

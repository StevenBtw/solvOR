# Welcome to the solvOR wiki

**Solvor all your optimization needs.**

solvOR is a pure Python optimization library that's readable, practical, and doesn't need numpy or scipy. Each solver fits in one file you can actually read.

## Quick Navigation

### By Category
- **[Linear & Integer Programming](Linear-&-Integer-Programming)** - Resource allocation, scheduling, production planning
- **[Constraint Programming](Constraint-Programming)** - Sudoku, N-Queens, logic puzzles
- **[Metaheuristics](Metaheuristics)** - TSP, combinatorial optimization, local search
- **[Continuous Optimization](Continuous-Optimization)** - ML training, curve fitting, gradient-based methods
- **[Pathfinding](Pathfinding)** - Shortest paths, graph traversal, mazes
- **[Network Flow & MST](Network-Flow-&-MST)** - Maximum flow, minimum cost flow, spanning trees
- **[Assignment](Assignment)** - Optimal matching, task allocation
- **[Exact Cover](Exact-Cover)** - Tiling puzzles, exact cover problems

### Quick Decision Tree

**I need the optimal solution and...**
- My constraints are linear → [`solve_lp`](solve_lp)
- Some variables must be integers → [`solve_milp`](solve_milp)
- It's all boolean logic → [`solve_sat`](solve_sat)
- I have complex constraints → [`Model`](Model) (CP-SAT)

**I'm exploring, good enough is fine...**
- I have a decent starting point → [`tabu_search`](tabu_search) or [`anneal`](anneal)
- I'm searching in the dark → [`evolve`](evolve)
- My function is smooth → [`adam`](adam) or [`gradient_descent`](gradient_descent)
- Each evaluation is expensive → [`bayesian_opt`](bayesian_opt)

**I need a path through a graph...**
- Unweighted graph → [`bfs`](bfs) or [`dfs`](dfs)
- Weighted, non-negative → [`dijkstra`](dijkstra) or [`astar`](astar)
- Negative edge weights → [`bellman_ford`](bellman_ford)
- All pairs shortest paths → [`floyd_warshall`](floyd_warshall)

**I'm working with networks...**
- Maximum throughput → [`max_flow`](max_flow)
- Minimum cost routing → [`min_cost_flow`](min_cost_flow) or [`network_simplex`](network_simplex)
- Connect all nodes minimally → [`kruskal`](kruskal) or [`prim`](prim)
- One-to-one matching → [`hungarian`](hungarian)

### Learn by Example

Check out the **[Cookbook](Cookbook)** for complete examples solving real problems:
- [Traveling Salesman Problem](Cookbook-TSP)
- [Sudoku Solver](Cookbook-Sudoku)
- [Resource Allocation](Cookbook-Resource-Allocation)
- [N-Queens Puzzle](Cookbook-N-Queens)
- And more!

## Philosophy

1. **Pure Python** - No compiled dependencies, runs anywhere Python runs
2. **Readable** - Each solver is one file you can understand
3. **Consistent** - Same `Result` format, same conventions
4. **Practical** - Solves real problems (and Advent of Code puzzles!)

## Result Format

Every solver returns the same `Result` namedtuple:

```python
Result(
    solution,     # best solution found
    objective,    # objective value
    iterations,   # solver iterations
    evaluations,  # function evaluations
    status        # OPTIMAL, FEASIBLE, INFEASIBLE, UNBOUNDED, MAX_ITER
)
```

## Getting Started

```bash
uv add solvor
```

```python
from solvor import solve_lp, dijkstra, hungarian

# Linear programming
result = solve_lp(c=[1, 2], A=[[1, 1]], b=[4])
print(result.solution)  # optimal x

# Shortest path
graph = {'A': [('B', 1), ('C', 4)], 'B': [('C', 2)], 'C': []}
result = dijkstra('A', 'C', lambda n: graph.get(n, []))
print(result.solution)  # ['A', 'B', 'C']

# Assignment
costs = [[10, 5], [3, 9]]
result = hungarian(costs)
print(result.solution)  # [1, 0]
```

Ready to dive in? Pick a category above or jump straight to the [Cookbook](Cookbook)!

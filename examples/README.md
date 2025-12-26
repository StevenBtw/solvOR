# solvOR Examples

A collection of worked examples, from minimal API demos to benchmark-grade implementations.

Most examples are self-contained and print their own explanations. Start with `quick_examples/` if you want to see the API in action, or dive into `classic/` for problems you can actually compare against published results.

## Quick Start

The `quick_examples/` directory contains minimal examples (10-20 lines) for each solver:

```bash
cd examples/quick_examples
python lp_example.py
python dijkstra_example.py
python cp_example.py
```

## Directory Structure

- **quick_examples/** - Minimal API examples for each solver
- **classic/** - Classic OR problems (TSP, knapsack, VRP, bin packing, job shop)
- **puzzles/** - Constraint programming puzzles (Sudoku, N-Queens, magic square, pentomino)
- **linear_programming/** - LP applications (diet, blending, portfolio)
- **graph_algorithms/** - Pathfinding, flow, and tree algorithms
- **machine_learning/** - ML optimization (regression, hyperparameters)
- **real_world/** - Practical applications (scheduling, timetabling)

## Examples by Category

### Linear & Integer Programming
- [lp_example.py](quick_examples/lp_example.py) - Linear programming
- [milp_example.py](quick_examples/milp_example.py) - Mixed-integer LP
- [diet_problem.py](linear_programming/diet_problem.py) - Classic Stigler diet
- [blending_problem.py](linear_programming/blending_problem.py) - Oil blending
- [portfolio_optimization.py](linear_programming/portfolio_optimization.py) - Markowitz portfolio

### Constraint Programming
- [sat_example.py](quick_examples/sat_example.py) - Boolean satisfiability
- [cp_example.py](quick_examples/cp_example.py) - CP model with all_different
- [sudoku_solver.py](puzzles/sudoku/sudoku_solver.py) - 9x9 Sudoku
- [n_queens.py](puzzles/n_queens/n_queens.py) - N-Queens problem
- [magic_square.py](puzzles/magic_square/magic_square.py) - Magic square
- [zebra_puzzle.py](puzzles/einstein_riddle/zebra_puzzle.py) - Einstein's riddle
- [pentomino_tiling.py](puzzles/pentomino/pentomino_tiling.py) - Pentomino tiling (DLX)

### Metaheuristics
- [anneal_example.py](quick_examples/anneal_example.py) - Simulated annealing
- [tabu_example.py](quick_examples/tabu_example.py) - Tabu search
- [genetic_example.py](quick_examples/genetic_example.py) - Genetic algorithm
- [differential_evolution_example.py](quick_examples/differential_evolution_example.py) - DE
- [particle_swarm_example.py](quick_examples/particle_swarm_example.py) - PSO

### Gradient-Based Optimization
- [gradient_descent_example.py](quick_examples/gradient_descent_example.py) - Basic GD
- [momentum_example.py](quick_examples/momentum_example.py) - Momentum GD
- [adam_example.py](quick_examples/adam_example.py) - Adam optimizer
- [bfgs_example.py](quick_examples/bfgs_example.py) - BFGS quasi-Newton
- [powell_example.py](quick_examples/powell_example.py) - Powell's method
- [nelder_mead_example.py](quick_examples/nelder_mead_example.py) - Nelder-Mead simplex
- [bayesian_example.py](quick_examples/bayesian_example.py) - Bayesian optimization

### Graph Algorithms
- [dijkstra_example.py](quick_examples/dijkstra_example.py) - Shortest path
- [bfs_example.py](quick_examples/bfs_example.py) - Breadth-first search
- [astar_example.py](quick_examples/astar_example.py) - A* pathfinding
- [bellman_ford_example.py](quick_examples/bellman_ford_example.py) - Negative weights
- [floyd_warshall_example.py](quick_examples/floyd_warshall_example.py) - All-pairs shortest
- [grid_pathfinding.py](graph_algorithms/shortest_path/grid_pathfinding.py) - BFS vs A* comparison
- [currency_arbitrage.py](graph_algorithms/shortest_path/currency_arbitrage.py) - Arbitrage detection

### Network Flow
- [max_flow_example.py](quick_examples/max_flow_example.py) - Maximum flow
- [network_simplex_example.py](quick_examples/network_simplex_example.py) - Min-cost flow
- [bipartite_matching.py](graph_algorithms/flow/bipartite_matching.py) - Maximum matching

### Spanning Trees
- [kruskal_example.py](quick_examples/kruskal_example.py) - MST (Kruskal)
- [prim_example.py](quick_examples/prim_example.py) - MST (Prim)

### Assignment & Covering
- [hungarian_example.py](quick_examples/hungarian_example.py) - Hungarian algorithm
- [dlx_example.py](quick_examples/dlx_example.py) - Exact cover (DLX)

### Classic OR Problems
- **TSP** (with TSPLIB att48 benchmark):
  - [tsp_tabu.py](classic/tsp/tsp_tabu.py) - Tabu search
  - [tsp_anneal.py](classic/tsp/tsp_anneal.py) - Simulated annealing
  - [tsp_genetic.py](classic/tsp/tsp_genetic.py) - Genetic algorithm
  - [att48.py](classic/tsp/att48.py) - TSPLIB benchmark data
- **Knapsack**:
  - [knapsack_01.py](classic/knapsack/knapsack_01.py) - Dynamic programming
  - [knapsack_milp.py](classic/knapsack/knapsack_milp.py) - MILP formulation
- **Assignment**:
  - [hungarian_demo.py](classic/assignment/hungarian_demo.py) - Worker-task assignment
- **Vehicle Routing** (with Solomon benchmarks):
  - [solomon_r101.py](classic/vehicle_routing/solomon_r101.py) - Benchmark data
  - [vrp_time_windows.py](classic/vehicle_routing/vrp_time_windows.py) - VRPTW solver
- **Bin Packing** (with Falkenauer benchmarks):
  - [bin_packing_benchmark.py](classic/bin_packing/bin_packing_benchmark.py) - FFD heuristics
- **Job Shop Scheduling** (with ft06 benchmark):
  - [ft06_benchmark.py](classic/job_shop_scheduling/ft06_benchmark.py) - Classic 6x6 instance

### Real-World Applications
- [nurse_scheduling.py](real_world/nurse_scheduling.py) - Hospital staff scheduling
- **School Timetabling** (ITC 2007 competition):
  - [timetabling_sat.py](real_world/school_timetabling/timetabling_sat.py) - SAT approach
  - [timetabling_cp.py](real_world/school_timetabling/timetabling_cp.py) - CP-SAT approach
  - [timetabling_milp.py](real_world/school_timetabling/timetabling_milp.py) - MILP approach
  - [timetabling_anneal.py](real_world/school_timetabling/timetabling_anneal.py) - Simulated annealing
  - [timetabling_genetic.py](real_world/school_timetabling/timetabling_genetic.py) - Genetic algorithm
  - [timetabling_tabu.py](real_world/school_timetabling/timetabling_tabu.py) - Tabu search

### Machine Learning
- [linear_regression.py](machine_learning/linear_regression.py) - Gradient descent
- [logistic_regression.py](machine_learning/logistic_regression.py) - Classification

## Benchmark Data Sources

All benchmark instances are from well-known open-source repositories:

- **TSPLIB**: http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/
- **Solomon VRPTW**: https://www.sintef.no/projectweb/top/vrptw/solomon-benchmark/
- **Job Shop (OR-Library)**: http://jobshop.jjvh.nl/
- **Bin Packing (Falkenauer)**: http://people.brunel.ac.uk/~mastjjb/jeb/orlib/binpackinfo.html
- **ITC 2007 Timetabling**: https://www.itc2007.cs.qub.ac.uk/

## Running Examples

All examples are self-contained and can be run directly:

```bash
python examples/puzzles/sudoku/sudoku_solver.py
python examples/classic/tsp/tsp_anneal.py
python examples/real_world/nurse_scheduling.py
```

Examples print their results and explain the problem being solved.

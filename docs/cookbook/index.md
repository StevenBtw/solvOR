# Cookbook

Working examples solving real problems with solvOR.

## Routing & Combinatorial

- **[Traveling Salesman](tsp.md)** - Find the shortest tour visiting all cities
- **[Knapsack Problem](knapsack.md)** - Select items to maximize value within weight limit
- **[Bin Packing](bin-packing.md)** - Pack items into minimum bins

## Puzzles & Games

- **[Sudoku Solver](sudoku.md)** - Solve 9x9 Sudoku using constraint programming
- **[N-Queens](n-queens.md)** - Place N queens on a board with no conflicts
- **[Magic Square](magic-square.md)** - Numbers 1-N² with equal row/column/diagonal sums
- **[Pentomino Tiling](pentomino.md)** - Tile a rectangle with pentomino pieces
- **[Graph Coloring](graph-coloring.md)** - Color nodes so no adjacent nodes share a color

## Scheduling & Allocation

- **[Resource Allocation](resource-allocation.md)** - Optimal task assignment with MILP
- **[Assignment Problem](assignment.md)** - Match workers to tasks optimally
- **[Nurse Scheduling](nurse-scheduling.md)** - Shift scheduling with SAT
- **[Job Shop Scheduling](job-shop.md)** - Multi-machine job scheduling

## Networks & Routing

- **[Shortest Path Grid](shortest-path-grid.md)** - Navigate a maze or grid
- **[Max Flow Network](max-flow.md)** - Find maximum throughput
- **[Currency Arbitrage](currency-arbitrage.md)** - Detect arbitrage with Bellman-Ford

## Linear Programming

- **[Diet Problem](diet.md)** - Minimum cost nutrition (LP classic)
- **[Production Planning](production-planning.md)** - Maximize profit with resource constraints
- **[Portfolio Optimization](portfolio.md)** - Asset allocation with LP

## Tips

- **Start Simple** - Run the basic example first
- **Understand the Encoding** - How is the problem represented?
- **Check Status** - Always verify `result.status` before using the solution

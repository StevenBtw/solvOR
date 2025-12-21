# Cookbook

Working examples solving real problems with solvOR.

## Problem Categories

### Routing & Combinatorial
- **[Traveling Salesman](Cookbook-TSP)** - Find the shortest tour visiting all cities
- **[Knapsack Problem](Cookbook-Knapsack)** - Select items to maximize value within weight limit

### Puzzles & Games
- **[Sudoku Solver](Cookbook-Sudoku)** - Solve 9×9 Sudoku using constraint programming
- **[N-Queens](Cookbook-N-Queens)** - Place N queens on a board with no conflicts
- **[Graph Coloring](Cookbook-Graph-Coloring)** - Color nodes so adjacent nodes differ

### Scheduling & Allocation
- **[Resource Allocation](Cookbook-Resource-Allocation)** - Optimal task assignment with MILP
- **[Assignment Problem](Cookbook-Assignment)** - Match workers to tasks optimally

### Networks & Routing
- **[Shortest Path Grid](Cookbook-Shortest-Path-Grid)** - Navigate a maze or grid
- **[Max Flow Network](Cookbook-Max-Flow-Network)** - Find maximum throughput

## Usage Pattern

Each cookbook entry follows this structure:
1. **Problem Description** - What are we solving?
2. **Why It's Interesting** - What makes this hard or important?
3. **Complete Code** - Working example you can run
4. **Explanation** - How the solution works
5. **Variations** - Extensions and related problems

## Tips for Using the Cookbook

- **Start Simple** - Run the basic example first
- **Understand the Encoding** - How is the problem represented?
- **Modify and Experiment** - Change parameters, try variations
- **Check Status** - Always verify `result.status` before using solution

## Contributing Examples

Found a great use case? The pattern is:
1. Define the problem clearly
2. Show the encoding (how you represent it)
3. Provide complete working code
4. Explain the key insights

Happy optimizing!

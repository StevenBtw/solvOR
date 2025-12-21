# Wiki Setup Instructions

Your comprehensive solvOR wiki is ready! Here's how to publish it to GitHub:

## Quick Stats
- **46 total pages** created
- **8 category overview pages**
- **27 individual solver pages**
- **10 cookbook examples with working code**
- **1 home page**

All content is in the `wiki-content/` directory.

## Setup Steps

### Option 1: GitHub Wiki (Recommended)

1. **Enable the Wiki** on your GitHub repository:
   - Go to https://github.com/StevenBtw/solvOR/settings
   - Scroll to "Features"
   - Check "Wikis"

2. **Clone the wiki repository** (after enabling):
   ```bash
   git clone https://github.com/StevenBtw/solvOR.wiki.git
   cd solvOR.wiki
   ```

3. **Copy the wiki content**:
   ```bash
   cp ../wiki-content/*.md .
   ```

4. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Add comprehensive wiki documentation"
   git push
   ```

5. **View your wiki** at:
   https://github.com/StevenBtw/solvOR/wiki

### Option 2: Automated Script

Run this script from your repository root:

```bash
#!/bin/bash
# Enable wiki on GitHub first, then run this

# Clone wiki repo
git clone https://github.com/StevenBtw/solvOR.wiki.git /tmp/wiki-repo

# Copy content
cp wiki-content/*.md /tmp/wiki-repo/

# Commit and push
cd /tmp/wiki-repo
git add .
git commit -m "Add comprehensive wiki documentation with 46 pages"
git push

echo "Wiki published! View at: https://github.com/StevenBtw/solvOR/wiki"
```

### Option 3: Manual Upload

1. Enable Wiki on GitHub (step 1 above)
2. Click "Create the first page"
3. Manually copy/paste content from each `.md` file in `wiki-content/`
4. GitHub will automatically create the navigation

## Wiki Structure

```
Home
├── Linear & Integer Programming
│   ├── solve_lp
│   └── solve_milp
├── Constraint Programming
│   ├── solve_sat
│   └── Model (CP-SAT)
├── Metaheuristics
│   ├── anneal
│   ├── tabu_search
│   ├── solve_tsp
│   └── evolve
├── Continuous Optimization
│   ├── gradient_descent
│   ├── momentum
│   ├── rmsprop
│   ├── adam
│   └── bayesian_opt
├── Pathfinding
│   ├── bfs
│   ├── dfs
│   ├── dijkstra
│   ├── astar & astar_grid
│   ├── bellman_ford
│   └── floyd_warshall
├── Network Flow & MST
│   ├── max_flow
│   ├── min_cost_flow
│   ├── network_simplex
│   ├── kruskal
│   └── prim
├── Assignment
│   ├── hungarian
│   └── solve_assignment
├── Exact Cover
│   └── solve_exact_cover
└── Cookbook
    ├── TSP
    ├── Sudoku
    ├── N-Queens
    ├── Resource Allocation
    ├── Shortest Path Grid
    ├── Max Flow Network
    ├── Assignment
    ├── Knapsack
    └── Graph Coloring
```

## What's Included

### Category Pages
Each category page includes:
- Overview of all solvers in the category
- When to use this category
- Comparison tables
- Real-world use cases

### Solver Pages
Each solver page includes:
- "At a Glance" summary
- "When to use this" with examples
- "When NOT to use this" with alternatives
- Quick code examples
- Intuitive "How it works" explanations
- Parameter guides
- Common gotchas
- Related solvers

### Cookbook Pages
Complete working examples for:
- Traveling Salesman Problem
- Sudoku solving
- N-Queens puzzle
- Resource allocation
- Shortest path in grids
- Maximum flow networks
- Assignment problems
- Knapsack optimization
- Graph coloring

## Tone & Style

All pages maintain the witty, approachable voice from your docstrings:
- "Dijkstra's negativity was legendary, just not in his algorithm"
- "The true edgelord" (Bellman-Ford)
- "When you want nature to do the work" (genetic algorithms)

Enjoy your new wiki! 🎉

# Algorithms

solvOR packs 40+ algorithms into one library. Here's how to find the right one.

## By Category

| Category | Solvers | When to Use |
|----------|---------|-------------|
| **[Linear Programming](linear-programming/solve-lp.md)** | `solve_lp`, `solve_milp` | Linear objectives, linear constraints |
| **[Constraint Programming](constraint-programming/solve-sat.md)** | `solve_sat`, `Model`, `solve_exact_cover` | Logic puzzles, satisfiability, exact cover |
| **[Metaheuristics](metaheuristics/index.md)** | `anneal`, `tabu_search`, `lns`, `evolve` | Combinatorial, no gradients |
| **[Continuous](continuous/index.md)** | `adam`, `bfgs`, `nelder_mead`, `bayesian_opt` | Smooth functions, gradients or black-box |
| **[Graph](graph/index.md)** | `dijkstra`, `astar`, `max_flow`, `kruskal` | Paths, flows, spanning trees |
| **[Combinatorial](combinatorial/index.md)** | `solve_knapsack`, `solve_bin_pack`, `solve_job_shop` | Packing, scheduling, routing |

## Quick Decision Tree

```
Is your objective linear?
├─ Yes → Do you have integer variables?
│        ├─ Yes → solve_milp
│        └─ No → solve_lp
│
└─ No → Can you compute gradients?
         ├─ Yes → Is evaluation expensive?
         │        ├─ Yes → bfgs or lbfgs
         │        └─ No → adam
         │
         └─ No → Is the problem discrete/combinatorial?
                  ├─ Yes → Is it a puzzle/exact cover?
                  │        ├─ Yes → solve_exact_cover or Model
                  │        └─ No → anneal or tabu_search
                  │
                  └─ No → Is evaluation expensive?
                           ├─ Yes → bayesian_opt
                           └─ No → nelder_mead or differential_evolution
```

## Guarantees

Some solvers prove optimality, others don't.

| Guarantee | Solvers |
|-----------|---------|
| **Optimal** | `solve_lp`, `solve_milp`, `dijkstra`, `bellman_ford`, `kruskal`, `solve_hungarian` |
| **Heuristic** | `anneal`, `tabu_search`, `evolve`, `particle_swarm` |
| **Local optimum** | `gradient_descent`, `adam`, `bfgs`, `nelder_mead` |

When you need a certificate of optimality, use the exact solvers. When you need speed and "good enough," use heuristics.

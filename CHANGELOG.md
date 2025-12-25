# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).



## [0.4.7] - 2025-12-25

Added a changelog (this file), a whole lot of solvors and some much needed optimizations, working on some more examples, but they need a bit more work still. 
This could have been a 0.5.0 release, if it wasn't for the examples, readme and wiki. Will probably add the examples in the next release, then 0.5.0 with extra tests and more polish (readme/wiki/etc.).

### Added

Santa's been busy, a lot more solvors, focussing on more real world problems, good for some examples I want to add later.

- `job_shop` job shop scheduling
- `vrp` vehicle routing problem
- `lns` large neighborhood search
- `differential_evolution` evolution strategy for continuous optimization
- `particle_swarm` swarm intelligence (just peer pressure for algorithms), also includes "velocity clamping" so particles don't yeet into infinity, which the textbook examples aparently do
- `knapsack` the classic packing problem
- `bin_pack` fit items into bins

- `CHANGELOG.md` to keep track of what was done when for future entertainment/troubleshooting

### Changed

- **`bayesian_opt` a lot of upgrades, including:**
  - Acquisition optimization now tries multiple starting points (was single-shot before)
  - Progress callbacks for monitoring long runs
  - Cholesky decomposition instead of Gaussian elimination, more stable numerically
  - Fixed status reporting when hitting iteration limit

- **`adam` added learning rate schedules:**
  - Supports constant, step, cosine, and warmup decay

- **`solve_exact_cover` added secondary columns:**
  - Optional constraints, covered at most once, but not required

- **`evolve` now with adaptive mutation:**
  - Mutation rate responds to progress and increases when stuck, decreases when improving

## [0.4.6] - 2025-12-24

Gradient-free optimizers join the party.

### Added

- **Quasi-Newton Methods:**
  - `bfgs` BFGS with optional line search and progress callbacks
  - `lbfgs` L-BFGS for when memory matters
- **Derivative-Free Optimization:**
  - `nelder_mead` simplex method with adaptive parameters
  - `powell` conjugate direction method with optional bounds

### Changed

- Renamed `hungarian` to `solve_hungarian` for naming consistency with other solve_* functions
- Upgraded `ty` to 0.0.7 (since it failed a job because I used :latest:)

## [0.4.5] - 2025-12-24

### Added

- Real world example: school timetabling with multiple solver approaches
- Utils module refactored into proper subpackage

### Changed

- Rewrote `cp.py` constraint propagation now cleaner
- Rewrote `sat.py` SAT solver improvements, some I knew, some I learned about today 
- Rewrote `utils.py` split into data_structures, helpers, validate

## [0.4.4] - 2025-12-23

### Added

- More gradient model tests, improved coverage
- Type annotations for public API's

### Changed

- Network simplex tweaks

## [0.4.3] - 2025-12-22

### Added

- Pre-commit hooks
- .envrc
- Switched to `ty` for type checking

Thanks to: [uv-cookiecutter](https://github.com/jevandezande/uv-cookiecutter)

### Changed

- CONTRIBUTING.md now has actual useful guidelines
- Ruff autoformat applied everywhere

## [0.4.2] - 2025-12-22

### Added

- GitHub Wiki with full documentation
- Codecov integration, now with badges

### Changed

- Python 3.12 minimum (3.14 still fastest, then 3.13, then 3.12)
- Fixed `assignment_cost` rejecting valid negative indices
- Fixed `is_feasible` dimension mismatch handling
- BFS now returns OPTIMAL status like DFS, consistency is nice

## [0.4.1] - 2025-12-20

### Changed

- Code clarity improvements
- Dev tooling additions

## [0.4.0] - 2025-12-20

Graph algorithms join the party.

### Added

Some extracted logic from old AoC solutions:

- **Pathfinding:**
  - `astar`, `astar_grid` A* for graphs and grids
  - `bfs`, `dfs` the classics
  - `dijkstra` shortest paths
  - `bellman_ford` when edges go negative
  - `floyd_warshall` all pairs, all the time
- **Minimum Spanning Tree:**
  - `kruskal`, `prim` two ways to connect everything cheaply
- **Assignment & Flow:**
  - `hungarian` optimal assignment in polynomial time
  - `network_simplex` min cost flow done right

## [0.3.5] - 2025-12-19

### Changed

- Python 3.13 minimum (was 3.14, but let's be reasonable)

## [0.3.4] - 2025-12-19

### Added

- Per-solver test files
- CI now tests each solver individually

### Fixed

- PyPI classifier was invalid (oops)

## [0.3.3] - 2025-12-19

### Added

- `solve_exact_cover` - Algorithm X (DLX) implementation
- Actual docstrings for public APIs

### Changed

- README explains *when* and *why*, not just *what*
- Improved names and docstrings

## [0.3.2] - 2025-12-18

### Changed

- Flattened project structure - `from solvor import solve_lp` just works now
- CI updated to match

## [0.3.1] - 2025-12-18

### Added

- GitHub Actions CI , automated tests

## [0.3.0] - 2025-12-18

First public release. Moved my solver collection from "random scripts folder(s)" to "actual package".

### Added

- **Linear Programming:**
  - `solve_lp` simplex method
  - `solve_milp` mixed-integer LP
- **Constraint Satisfaction:**
  - `solve_sat` SAT solver
  - `solve_assignment` assignment problems
  - `solve_tsp` traveling salesman
- **Metaheuristics:**
  - `anneal` simulated annealing
  - `tabu_search` tabu search
  - `evolve` genetic algorithm
- **Continuous Optimization:**
  - `gradient_descent`, `momentum`, `rmsprop`, `adam` gradient-based optimizers
  - `bayesian_opt` when you can't compute gradients
- **Network Flow:**
  - `max_flow` Ford-Fulkerson
  - `min_cost_flow` minimum cost flow
- `Result` dataclass with `.ok` property and `Status` enum
- Pure Python, no dependencies, works everywhere


[0.4.7]: https://github.com/StevenBtw/solvor/compare/v0.4.6...v0.4.7
[0.4.6]: https://github.com/StevenBtw/solvor/compare/v0.4.5...v0.4.6
[0.4.5]: https://github.com/StevenBtw/solvor/compare/v0.4.4...v0.4.5
[0.4.4]: https://github.com/StevenBtw/solvor/compare/v0.4.3...v0.4.4
[0.4.3]: https://github.com/StevenBtw/solvor/compare/v0.4.2...v0.4.3
[0.4.2]: https://github.com/StevenBtw/solvor/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/StevenBtw/solvor/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/StevenBtw/solvor/compare/v0.3.5...v0.4.0
[0.3.5]: https://github.com/StevenBtw/solvor/compare/v0.3.4...v0.3.5
[0.3.4]: https://github.com/StevenBtw/solvor/compare/v0.3.3...v0.3.4
[0.3.3]: https://github.com/StevenBtw/solvor/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/StevenBtw/solvor/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/StevenBtw/solvor/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/StevenBtw/solvor/releases/tag/v0.3.0

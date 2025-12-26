# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2025-12-26

**solvOR** is now in beta! 

Documentation finally has a proper home. Moved everything from the wiki to MkDocs, deployed at [solver.ai](https://solver.ai). The wiki now just points there.

### Added

- Documentation site at [solver.ai](https://solver.ai). MkDocs with Material theme, dark mode by default (as it should be). Getting started, algorithm reference (40+ pages), cookbook with 18 worked examples, API docs, troubleshooting.

- GitHub Actions workflow for docs, auto-deploys to GitHub Pages on push to main.

### Changed

- **BREAKING:** `evolve()` renamed `max_gen` to `max_iter` for consistency. Update your code: `evolve(..., max_gen=100)` → `evolve(..., max_iter=100)`

- `Result` is now generic (`Result[T]`), so type checkers actually know what `.solution` contains. Less red underscores in the IDE.

- Test coverage now statistically significant at 95%, added tests for adam learning rate schedules, genetic adaptive mutation, validation utilities.

- Internal cleanup: `Evaluator` class and `report_progress()` in `solvor/utils/helpers.py`, deduplicated boilerplate across 12 solver files. No API changes for this one.

- Wiki retired, all content now lives at solver.ai. The wiki just points there.

## [0.4.8] - 2025-12-25

Examples! Finally they all work (and can be used as extra tests! optional for now). Also some last changes to solvors that didn't play nice with some of the wikipedia examples. 0.5.0 will be next, which will be considered a "beta" where 0.4.x releases are "alpha". Development cycle will slow down after that, extra maintainers are welcome!

### Added

- **We need examples, lots of examples:**
  - `quick_examples/` minimal working code for every solver, copy-paste ready
  - `classic/` TSP, knapsack, bin packing, job shop, VRP benchmarks
  - `puzzles/` sudoku, n-queens, zebra puzzle, pentomino, magic square
  - `linear_programming/` diet problem, portfolio optimization, blending
  - `machine_learning/` gradient descent for regression (yes it works)
  - `real_world/` nurse scheduling that actually respects constraints

- `py.typed` marker for type checker support

### Changed

- **WIKILEAKS:** Wiki completely up to date, using quick examples from the repo now for consistency. Lots of copy-pasting.

- **Full tests overhaul:** Tests reorganized into `tests/solvers/` and `tests/examples/`, grouped example tests into folder categories. More copy-pasting.

- **CI simplified** the per-solver conditional testing was clever but unmaintainable, just run everything now, tweaked some of the slower tests.

- **Input validation added to graph algorithms and MILP:** bellman_ford, floyd_warshall, mst and milp now tell you what's wrong instead of crashing.

- **Extra parameters:** I don't want to add too many parameters and complexity, but added warm_start to milp and a solution limit. Added `allow_forest=True` for kruskals, which returns minimum spanning forest instead of giving up.

- **README & CONTRIBUTING:** Up-to-date again, were lagging behind with new solvers and parameters, CI changes and more. Should be consistent with the repo's content again.

## [0.4.7] - 2025-12-25

Added a changelog (this file), a whole lot of solvors and some much needed optimizations, working on some more examples, but they need a bit more work still. 
This could have been a 0.5.0 release, if it wasn't for the examples, readme and wiki. Will probably add the examples in the next release, then 0.5.0 with extra tests and more polish (readme/wiki/etc.).

### Added

Santa's been busy, a lot more solvors, focussing on more real world problems, good for some examples I want to add later.

- `job_shop` job shop scheduling
- `vrp` vehicle routing problem
- `lns` large neighborhood search
- `differential_evolution` evolution strategy for continuous optimization
- `particle_swarm` swarm intelligence (just peer pressure for algorithms), also includes "velocity clamping" so particles don't yeet into infinity, which the textbook examples apparently do
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

- `solve_exact_cover`, Knuths algorithm X (DLX) implementation
- Actual docstrings for public APIs

### Changed

- README explains *when* and *why*, not just *what*
- Improved names and docstrings

## [0.3.2] - 2025-12-18

### Changed

- Flattened project structure, `from solvor import solve_lp` just works now
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


[0.5.0]: https://github.com/StevenBtw/solvor/compare/v0.4.8...v0.5.0
[0.4.8]: https://github.com/StevenBtw/solvor/compare/v0.4.7...v0.4.8
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

# Changelog

What broke, what got fixed, and what's new.

## [Unreleased]

### Added

- **LP/MILP input:** `solve_lp` and `solve_milp` accept sparse rows (`{column: coefficient}` dicts, mixable with dense rows), per-row `senses` (`"<="`, `">="`, `"="`) and variable bounds `lb`/`ub`. `solve_milp` adds `binary=` (0/1 variables) and `integers` becomes optional. The 0.6.2 call form is unchanged.
- **Incremental MILP:** `MilpModel` adds rows and re-solves warm (lazy cuts, objective locks, new objectives); `solve_lexicographic` optimizes objectives in priority order on a model (a 270-variable, 4-stage downstream model: 7.4 ms). Adding a cut to a 540-variable model and re-solving takes 13% of a fresh solve.

### Fixed

- **Deep graphs:** `articulation_points`, `bridges` and `strongly_connected_components` no longer raise `RecursionError` when the DFS goes deeper than Python's recursion limit (about 1,000 nodes on one path; code dependency graphs reach that easily). All three now use an iterative DFS with the same results, result order and iteration counts; a 100,000-node path or tree takes under 0.3 s. The SCC docstring's advice to raise the recursion limit is removed.
- **Rust SCC crash:** `strongly_connected_components_edges` with the Rust backend (the default) no longer kills the Python process with a native stack overflow on deep graphs (a 100,000-node chain did). The Rust implementation is now iterative; results are identical to the Python backend, component order included.
- **Version:** `solvor.__version__` reported 0.6.1 in the 0.6.2 release. It now reads the installed package metadata, and the version lives only in `rust/Cargo.toml`.
- **Determinism:** `louvain` and `pagerank` no longer depend on set iteration order: with mutually comparable nodes (ints, strs, tuples) the same graph gives the same result under any `PYTHONHASHSEED`. Result lists and dicts follow sorted node order.
- **PageRank:** `max_iter=0`, `tol <= 0` and `damping` outside `[0, 1)` now raise `ValueError` on both backends (the Python path crashed or misbehaved). `pagerank_edges(backend="python")` ignores out-of-range edges like the Rust backend instead of raising `IndexError`, and the Rust backend now also ignores edges with negative endpoints (it raised `OverflowError`). A negative `n_nodes` raises `ValueError` on both backends.

### Result changes

Every change in solver output is listed here, float-level ones included, so downstream caches know which upgrade invalidates them.

- **Louvain now runs both phases.** Earlier releases only ran local moving and never aggregated communities, although the docs described both phases. Results change: fewer, larger communities with higher modularity (a 1,000-node path: 500 pairs at 0.50 before, above 0.9 now; a 9,091-node code graph: 1,821 communities at 0.587 before, 115 at 0.823 now). Nodes and neighbors are now processed in sorted order, so even the first level only matches earlier releases for inputs that were already sorted. If you store community labels, expect them to change. `result.iterations` now counts local-moving passes over all levels. Community list order is now by smallest member.
- **PageRank, Rust backend:** scores change at the float level (now bit-identical to the Python backend; it used an L1 convergence test before, so it ran more iterations). Rust `pagerank_edges` now reports `objective` = the last iteration's largest score change and `evaluations` = the number of nodes (it returned 0.0 and 0).
- **PageRank, Python backend:** scores change in their last few bits (up to about 4 ulp: it multiplies by the reciprocal of the out-degree instead of dividing); iteration counts are unchanged.
- **Result order:** `louvain` and `pagerank` list and dict entries now follow sorted node order (for mutually comparable nodes) instead of input order.
- **LP/MILP:** optimal objectives are unchanged (a 300-instance snapshot from 0.6.2 is part of the test suite), but where several optimal solutions exist `solve_lp` and `solve_milp` may return a different one than 0.6.2 (new LP engine and presolve).
- **Unbounded LP:** `solve_lp` reports an objective of `-inf` (minimize) or `inf` (maximize) instead of the last vertex's value (`solve_milp` already did).
- **Infeasible LP when maximizing:** `solve_lp` reports an objective of `-inf` (it reported `inf` for both directions); minimizing still gives `inf`.
- **Iteration limit before a feasible point:** `solve_lp` reports `MAX_ITER` (it reported `INFEASIBLE`), with no solution; `solve_milp` returns `MAX_ITER` when the root LP hits the limit instead of branching on an unfinished LP.
- **LP iteration counts** include bound flips (a variable jumping between its bounds without a pivot).
- **MILP solutions:** integer variables are exactly integral (`1.0`, not `0.9999999999999998`) and the objective is recomputed from them, so objectives like `79.00000000000007` become `79.0`.
- **MILP tie-breaking:** the new search order (depth-first dives) can return a different optimal solution than before where several optima tie; objectives are unchanged.
- **MILP counts:** `iterations` (nodes explored) and `evaluations` (LP iterations, now including bound flips and warm-started dual pivots) change with the new presolve and search, usually to much smaller values.

### Changed

- **PageRank speed:** the callback `pagerank` now runs on the Rust backend when installed (9,091 nodes, 33,580 arcs: 345 ms to 12 ms; the pure Python path also dropped to 183 ms). New `backend` keyword. The Rust and Python backends now return bit-identical scores and iteration counts (same max-norm convergence test, same compensated summation), so results do not depend on whether the Rust wheel is installed.
- **Wheels:** one abi3 wheel per platform (CPython 3.12+) built with pyo3 0.29, instead of one wheel per Python version.
- **Rust toolchain:** pinned to 1.99.0 for development and CI (`rust-toolchain.toml`); building from the sdist needs Rust 1.88 or newer. CI now runs rustfmt, clippy (warnings are errors), `cargo test` and a 1.88 build check.
- **LP engine:** bounded-variable simplex. Bounds no longer become rows, so binary models no longer need identity rows and branch-and-bound nodes no longer add bound rows (540 binary variables: root LP 108 ms to 11 ms).
- **MILP presolve:** single-variable rows become bounds; rows with only integer variables and integer coefficients are divided by their gcd and get an integral right-hand side. Lexicographic lock rows with a `+0.5` slack now solve at or near the root (a 270-variable model: 83.5 s and 1,391 nodes to 0.01 s and 3 nodes; 20 recorded calls from a downstream project: 105.6 s to 0.05 s, same objectives).
- **MILP heuristics:** the rounding heuristic updates row activities incrementally instead of re-checking every row after every move (a 540-variable model spent 174 s there).
- **Validation** runs once per call instead of in every branch-and-bound node.
- **LP/MILP input:** an empty `A` is accepted (only bounds), rows with an infinite right-hand side are dropped when they can never bind and make the model infeasible when they can never hold.
- **MILP branch and bound** re-solves each node with the dual simplex from its parent's basis and dives depth-first before taking the best queued node (7 branching benchmarks: 8.6 s to 2 s, 5 to 14 times fewer LP pivots).

## [0.6.2] 2026-04-07

### Fixed

- **Rust:** Dijkstra, BFS, DFS, and Bellman-Ford now raise `ValueError` on out-of-bounds source/target instead of crashing the process.
- **Rust:** Floyd-Warshall callback errors are now propagated to Python instead of being silently swallowed.
- **Rust:** `PyList` construction in bindings uses proper error propagation instead of `unwrap()`.
- **Rust:** Dijkstra and Kruskal use `total_cmp` for deterministic NaN handling in edge weights.
- **Rust:** PageRank validates `damping`, `max_iter`, and `tol` parameters at the binding boundary.

### Changed

- **Rust:** Removed `hashbrown` and `ahash` dependencies from `Cargo.toml`.

## [0.6.1] - 2026-02-01

### Fixed

- **Build:** Fixed LICENSE file not included in source distribution.
- **CI:** Updated macOS runner from deprecated macos-13 to macos-14.
- **PyPI:** Normalized package name to lowercase `solvor` for wheel compatibility.

## [0.6.0] - 2026-02-01

**Rust acceleration!** Optional Rust backend for graph algorithms with 3-60x speedup, ported from [Grafeo](https://github.com/GrafeoDB/grafeo).

### Added

- **Rust Backend:** Optional high-performance Rust implementations via PyO3. Algorithms automatically use Rust when available, with seamless Python fallback. Use `backend="rust"` or `backend="python"` to force a specific implementation.

- **Algorithms with Rust backends:**
  - `floyd_warshall` - All-pairs shortest paths (45-60x speedup)
  - `bellman_ford` - Negative weight shortest paths (10-20x speedup)
  - `dijkstra_edges` - Single-source shortest paths (5-10x speedup)
  - `bfs_edges`, `dfs_edges` - Graph traversal (3-5x speedup)
  - `pagerank_edges` - Node importance ranking (10-15x speedup)
  - `strongly_connected_components_edges`, `topological_sort_edges` - Dependency analysis (5-10x speedup)
  - `kruskal` - Minimum spanning tree (5-10x speedup)

- **Two API styles for graph algorithms:**
  - Callback-based (`dijkstra`, `bfs`, etc.) - Flexible, works with any node type, pure Python
  - Edge-list (`dijkstra_edges`, `bfs_edges`, etc.) - Integer nodes 0..n-1, Rust-accelerated

- **Performance documentation:** New [Performance page](https://solvOR.ai/getting-started/performance/) with benchmarks, backend usage guide, and API comparison.

- **Backend Infrastructure:**
  - `solvor/rust/` - Backend detection, routing, and adapters
  - `rust/` - Rust crate with PyO3 bindings
  - Multi-platform wheel builds (Linux, macOS, Windows) via GitHub Actions

- **Developer Tooling:**
  - Added [prek](https://github.com/j178/prek) for fast pre-commit hooks (Rust-based, uv-compatible)

### Changed

- **Build System:** Switched from hatchling to maturin for mixed Python/Rust builds.
- **CI:** New `publish.yml` workflow builds wheels for all platforms.
- **Code Organization:** Moved Rust backend utilities to `solvor/rust/` module.

## [0.5.5] - 2026-01-31

### Added

- **Branch-and-Price:** Added `solve_bp()` for optimal integer solutions via branch-and-price. Combines column generation with branch-and-bound to find provably optimal integer solutions. Same interface as `solve_cg()` with additional B&B parameters (`max_nodes`, `gap_tol`).

## [0.5.4] - 2026-01-24

### Added

- **Column Generation:** Added `solve_cg()` for problems with exponentially many variables. Implements Dantzig-Wolfe decomposition with LP master problem and customizable pricing. Built-in cutting stock mode with knapsack pricing, or provide your own pricing function for bin packing, vehicle routing, crew scheduling, graph coloring, etc.

- **OR-Tools Examples:** 24 converted examples showing solvOR as drop-in replacement for Google OR-Tools. Organized by category:
  - **Linear Solver (13):** LP, MIP, assignment, bin packing, knapsack problems using `solve_lp`, `solve_milp`, `solve_hungarian`, `solve_bin_pack`
  - **CP-SAT (7):** Constraint satisfaction using `Model` and `solve_job_shop` - simple CSP, N-Queens, nurse scheduling, job shop, solution enumeration
  - **Graph (2):** Max flow and linear sum assignment using `max_flow`, `solve_hungarian`
  - **Algorithms (2):** Knapsack problems using `solve_knapsack`

*All converted examples where generated using AI.

- **OR-Tools Documentation:** New index page at `docs/examples/ortools.md` with benchmark comparisons. solvOR often faster for small problems (pure Python startup vs C++ overhead), competitive on medium problems, OR-Tools wins on large problems as expected.

### Changed

- **MILP:** Greedy rounding + LNS heuristics for binary MIPs. Enable with `lns_iterations` parameter.

### Fixed

- Bugfix for `simplex.py`.

## [0.5.3] - 2025-12-29

Some small QoL improvements and ported all the graph algorithms from another project (AutoMate) which now relies on solvOR.

### Added

- **Graph analysis:** `topological_sort`, `strongly_connected_components`, `condense` for dependency ordering and cycle detection. `pagerank` for node importance. `louvain` for community detection. `articulation_points`, `bridges` for finding critical connections. `kcore_decomposition`, `kcore` for core/periphery analysis.
- **Version:** Added `__version__` to package.

### Changed

- **CI:** Parallelized lint/typecheck jobs, added publish version tag verification.
- **Exports:** Added `Progress`, `ProgressCallback` to public API.
- **Refactor:** Extracted `reconstruct_path` to utils (shared by dijkstra, bfs, a_star).

### Fixed

- **Flow solvers:** Now track iterations correctly.
- **Exports:** Fixed missing `__all__` in utils/validate.py.

## [0.5.2] - 2025-12-28

**Who let de docs out!**
solvOR went through some nice changes. CP got amputated, LP got more company, docs got a full checkup, CI got faster.

### Added

- **CI docs test:** `test_docs.py` runs `mkdocs build --strict` to catch missing type hints and broken links before they reach main.
- **Interior point:** Added `solve_lp_interior()` for linear programming. Primal-dual method with Mehrotra predictor-corrector. Alternative to simplex method.

### Fixed

- **README:** Fixed incorrect `solve_vrptw` example (wrong parameter names, missing customer IDs).

### Changed

- **CP refactor:** Extracted SAT encoding to `cp_encoder.py` (cp.py went from 861 to 428 lines, much easier to read now). `Model.solve()` defaults to `solver="auto"` which picks DFS for simple constraints and SAT for globals (circuit, no_overlap, cumulative, sum_*). Force a specific solver with `solver="dfs"` or `solver="sat"`. Also, `IntVar` now supports multiplication (`x * 3`, `3 * x`) for expressions like `timeslot * n_rooms + room`.
- **Module docstrings:** Standardized all 18 solvor docstrings with consistent format: "How it works" section, "Use this for" bullets, "Parameters" section, r"""string. Same style, so easier when scanning the different solvors.
- **solvOR.ai documentation:** Added "How It Works" sections explaining the math behind 16 algorithms (simplex, interior point, dijkstra, bellman-ford, floyd-warshall, A*, anneal, tabu, genetic, PSO, DE, BFGS, knapsack, bin-packing, SAT, DLX). Also added references, tips, and full signature sections.
- **Tests:** Removed tests with weak assertions, improved edge case coverage. No bugs found (yet).
- **CI/pre-commit:** Turned on uv cache, removed redundant type checks across Python versions, added no-commit-to-branch for main.
- **README:** Added docs badge.

## [0.5.1] - 2025-12-27

### Fixed

- **Documentation consistency pass:** Reviewed all 29 solvors and 7 extended examples against MkDocs pages. Fixed parameter name mismatches, incorrect signatures, and outdated solution formats across 15+ documentation files.

## [0.5.0] - 2025-12-26

**solvOR** is now in beta!

Documentation finally has a proper home. Moved everything from the wiki to MkDocs, deployed at [solvOR.ai](https://solvOR.ai). The wiki now just points there.

### Added

- Documentation site at [solvOR.ai](https://solvOR.ai). MkDocs with Material theme, dark mode by default (as it should be). Getting started, algorithm reference (40+ pages), cookbook with 18 worked examples, API docs, troubleshooting.

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


[0.6.0]: https://github.com/StevenBtw/solvOR/compare/v0.5.5...v0.6.0
[0.5.5]: https://github.com/StevenBtw/solvOR/compare/v0.5.4...v0.5.5
[0.5.4]: https://github.com/StevenBtw/solvOR/compare/v0.5.3...v0.5.4
[0.5.3]: https://github.com/StevenBtw/solvOR/compare/v0.5.2...v0.5.3
[0.5.2]: https://github.com/StevenBtw/solvOR/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/StevenBtw/solvOR/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/StevenBtw/solvOR/compare/v0.4.8...v0.5.0
[0.4.8]: https://github.com/StevenBtw/solvOR/compare/v0.4.7...v0.4.8
[0.4.7]: https://github.com/StevenBtw/solvOR/compare/v0.4.6...v0.4.7
[0.4.6]: https://github.com/StevenBtw/solvOR/compare/v0.4.5...v0.4.6
[0.4.5]: https://github.com/StevenBtw/solvOR/compare/v0.4.4...v0.4.5
[0.4.4]: https://github.com/StevenBtw/solvOR/compare/v0.4.3...v0.4.4
[0.4.3]: https://github.com/StevenBtw/solvOR/compare/v0.4.2...v0.4.3
[0.4.2]: https://github.com/StevenBtw/solvOR/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/StevenBtw/solvOR/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/StevenBtw/solvOR/compare/v0.3.5...v0.4.0
[0.3.5]: https://github.com/StevenBtw/solvOR/compare/v0.3.4...v0.3.5
[0.3.4]: https://github.com/StevenBtw/solvOR/compare/v0.3.3...v0.3.4
[0.3.3]: https://github.com/StevenBtw/solvOR/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/StevenBtw/solvOR/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/StevenBtw/solvOR/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/StevenBtw/solvOR/releases/tag/v0.3.0

---

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) | Versioning: [SemVer](https://semver.org/spec/v2.0.0.html)

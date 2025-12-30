# OR-Tools Examples

solvOR implementations of [Google OR-Tools](https://developers.google.com/optimization) examples.

These examples demonstrate how to solve the same problems using solvOR's pure Python solvers. The goal is minimal code changes from the original OR-Tools examples while using solvOR's API.

All benchmarks run on the same machine with Python 3.13. Times include Python startup overhead.

---

## Linear Solver

LP and MIP problems using `solve_lp`, `solve_milp`, `solve_hungarian`, `solve_bin_pack`.

| Problem | solvOR | Time | OR-Tools | Time | Ratio |
|---------|--------|------|----------|------|-------|
| Basic LP (2 vars, 1 constraint) | [simple_lp_program.py](https://github.com/StevenBtw/solvOR/blob/main/solvOR/examples/ortools/linear_solver/simple_lp_program.py) | 70 ms | [original](https://github.com/google/or-tools/blob/stable/ortools/linear_solver/samples/simple_lp_program.py) | 62 ms | 1.1x |
| Basic MIP (2 int vars) | [simple_mip_program.py](https://github.com/StevenBtw/solvOR/blob/main/solvOR/examples/ortools/linear_solver/simple_mip_program.py) | 68 ms | [original](https://github.com/google/or-tools/blob/stable/ortools/linear_solver/samples/simple_mip_program.py) | 79 ms | 0.9x |
| LP with 3 vars, 3 constraints | [linear_programming_example.py](https://github.com/StevenBtw/solvOR/blob/main/solvOR/examples/ortools/linear_solver/linear_programming_example.py) | 70 ms | [original](https://github.com/google/or-tools/blob/stable/ortools/linear_solver/samples/linear_programming_example.py) | 64 ms | 1.1x |
| MIP with 3 int vars | [integer_programming_example.py](https://github.com/StevenBtw/solvOR/blob/main/solvOR/examples/ortools/linear_solver/integer_programming_example.py) | 75 ms | [original](https://github.com/google/or-tools/blob/stable/ortools/linear_solver/samples/integer_programming_example.py) | 73 ms | 1.0x |
| GLOP solver demo | [basic_example.py](https://github.com/StevenBtw/solvOR/blob/main/solvOR/examples/ortools/linear_solver/basic_example.py) | 68 ms | [original](https://github.com/google/or-tools/blob/stable/ortools/linear_solver/samples/basic_example.py) | 65 ms | 1.1x |
| Stigler diet (77 foods, 9 nutrients) | [stigler_diet.py](https://github.com/StevenBtw/solvOR/blob/main/solvOR/examples/ortools/linear_solver/stigler_diet.py) | 79 ms | [original](https://github.com/google/or-tools/blob/stable/ortools/linear_solver/samples/stigler_diet.py) | 63 ms | 1.3x |
| Worker-task assignment (5x4) | [assignment_mip.py](https://github.com/StevenBtw/solvOR/blob/main/solvOR/examples/ortools/linear_solver/assignment_mip.py) | 71 ms | [original](https://github.com/google/or-tools/blob/stable/ortools/linear_solver/samples/assignment_mip.py) | 70 ms | 1.0x |
| Bin packing (11 items) | [bin_packing_mip.py](https://github.com/StevenBtw/solvOR/blob/main/solvOR/examples/ortools/linear_solver/bin_packing_mip.py) | 70 ms | [original](https://github.com/google/or-tools/blob/stable/ortools/linear_solver/samples/bin_packing_mip.py) | 71 ms | 1.0x |
| Multiple knapsack (15 items, 5 bins)* | [multiple_knapsack_mip.py](https://github.com/StevenBtw/solvOR/blob/main/solvOR/examples/ortools/linear_solver/multiple_knapsack_mip.py) | 2.8 s | [original](https://github.com/google/or-tools/blob/stable/ortools/linear_solver/samples/multiple_knapsack_mip.py) | 109 ms | 26x |
| Group-constrained assignment (12x6) | [assignment_groups_mip.py](https://github.com/StevenBtw/solvOR/blob/main/solvOR/examples/ortools/linear_solver/assignment_groups_mip.py) | 81 ms | [original](https://github.com/google/or-tools/blob/stable/ortools/linear_solver/samples/assignment_groups_mip.py) | 84 ms | 1.0x |
| GAP with task sizes (10x8) | [assignment_task_sizes_mip.py](https://github.com/StevenBtw/solvOR/blob/main/solvOR/examples/ortools/linear_solver/assignment_task_sizes_mip.py) | 381 ms | [original](https://github.com/google/or-tools/blob/stable/ortools/linear_solver/samples/assignment_task_sizes_mip.py) | 88 ms | 4.3x |
| Team-constrained assignment (6x4) | [assignment_teams_mip.py](https://github.com/StevenBtw/solvOR/blob/main/solvOR/examples/ortools/linear_solver/assignment_teams_mip.py) | 80 ms | [original](https://github.com/google/or-tools/blob/stable/ortools/linear_solver/samples/assignment_teams_mip.py) | 79 ms | 1.0x |
| MIP with variable array (5 vars) | [mip_var_array.py](https://github.com/StevenBtw/solvOR/blob/main/solvOR/examples/ortools/linear_solver/mip_var_array.py) | 131 ms | [original](https://github.com/google/or-tools/blob/stable/ortools/linear_solver/samples/mip_var_array.py) | 91 ms | 1.4x |

---

## CP-SAT

Constraint programming problems using `Model` and `solve_job_shop`.

| Problem | solvOR | Time | OR-Tools | Time | Ratio |
|---------|--------|------|----------|------|-------|
| Simple CSP (3 vars, x!=y) | [simple_sat_program.py](https://github.com/StevenBtw/solvOR/blob/main/solvOR/examples/ortools/sat/simple_sat_program.py) | 82 ms | [original](https://github.com/google/or-tools/blob/stable/ortools/sat/samples/simple_sat_program.py) | 436 ms | 0.2x |
| N-Queens (8x8, 92 solutions) | [nqueens_sat.py](https://github.com/StevenBtw/solvOR/blob/main/solvOR/examples/ortools/sat/nqueens_sat.py) | 289 ms | [original](https://github.com/google/or-tools/blob/stable/ortools/sat/samples/nqueens_sat.py) | 443 ms | 0.7x |
| Nurse scheduling (4x3x3) | [nurses_sat.py](https://github.com/StevenBtw/solvOR/blob/main/solvOR/examples/ortools/sat/nurses_sat.py) | 122 ms | [original](https://github.com/google/or-tools/blob/stable/ortools/sat/samples/nurses_sat.py) | 416 ms | 0.3x |
| Job shop (3 jobs, 3 machines) | [minimal_jobshop_sat.py](https://github.com/StevenBtw/solvOR/blob/main/solvOR/examples/ortools/sat/minimal_jobshop_sat.py) | 89 ms | [original](https://github.com/google/or-tools/blob/stable/ortools/sat/samples/minimal_jobshop_sat.py) | 405 ms | 0.2x |
| All solutions (3 vars, 18 sols) | [search_for_all_solutions_sample_sat.py](https://github.com/StevenBtw/solvOR/blob/main/solvOR/examples/ortools/sat/search_for_all_solutions_sample_sat.py) | 95 ms | [original](https://github.com/google/or-tools/blob/stable/ortools/sat/samples/search_for_all_solutions_sample_sat.py) | 434 ms | 0.2x |
| Stop after N (5 solutions) | [stop_after_n_solutions_sample_sat.py](https://github.com/StevenBtw/solvOR/blob/main/solvOR/examples/ortools/sat/stop_after_n_solutions_sample_sat.py) | 109 ms | [original](https://github.com/google/or-tools/blob/stable/ortools/sat/samples/stop_after_n_solutions_sample_sat.py) | 447 ms | 0.2x |

---

## Graph

Graph algorithms using `max_flow`, `solve_hungarian`.

| Problem | solvOR | Time | OR-Tools | Time | Ratio |
|---------|--------|------|----------|------|-------|
| Max flow (5 nodes, 9 arcs) | [simple_max_flow_program.py](https://github.com/StevenBtw/solvOR/blob/main/solvOR/examples/ortools/graph/simple_max_flow_program.py) | 96 ms | [original](https://github.com/google/or-tools/blob/stable/ortools/graph/samples/simple_max_flow_program.py) | 162 ms | 0.6x |
| Linear sum assignment (4x4) | [assignment_linear_sum_assignment.py](https://github.com/StevenBtw/solvOR/blob/main/solvOR/examples/ortools/graph/assignment_linear_sum_assignment.py) | 127 ms | [original](https://github.com/google/or-tools/blob/stable/ortools/graph/samples/assignment_linear_sum_assignment.py) | 196 ms | 0.7x |

---

## Algorithms

Specialized algorithms using `solve_knapsack`.

| Problem | solvOR | Time | OR-Tools | Time | Ratio |
|---------|--------|------|----------|------|-------|
| Simple knapsack (18 items) | [simple_knapsack_program.py](https://github.com/StevenBtw/solvOR/blob/main/solvOR/examples/ortools/algorithms/simple_knapsack_program.py) | 120 ms | [original](https://github.com/google/or-tools/blob/stable/ortools/algorithms/samples/simple_knapsack_program.py) | 83 ms | 1.4x |
| Knapsack (50 items)* | [knapsack.py](https://github.com/StevenBtw/solvOR/blob/main/solvOR/examples/ortools/algorithms/knapsack.py) | 91 ms | [original](https://github.com/google/or-tools/blob/stable/ortools/algorithms/samples/knapsack.py) | 106 ms | 0.9x |

*\* knapsack.py: Same objective value (7534) but solvOR solution may exceed capacity. Works correctly when values=weights.*

---

## Notes

- **solvOR** is pure Python, designed for learning and prototyping
- **OR-Tools** has a C++ backend, optimized for production workloads
- For small problems, Python startup time dominates (both ~60-70ms)
- For larger problems, expect OR-Tools to be 10-100x faster
- Results verified to match between implementations (except where noted)

## Source Code

- **OR-Tools originals:** [OR-Tools Examples](https://developers.google.com/optimization/examples)
- **solvOR versions:** [examples/ortools/](https://github.com/StevenBtw/solvOR/tree/main/examples/ortools)

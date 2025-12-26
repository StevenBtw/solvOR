# Combinatorial Solvers

Dedicated solvers for classic combinatorial optimization problems. When your problem has a well-known structure, use a specialized algorithm.

## Solvers

| Solver | Problem | Best For |
|--------|---------|----------|
| [`solve_knapsack`](knapsack.md) | 0/1 Knapsack | Select items to maximize value within capacity |
| [`solve_bin_pack`](bin-packing.md) | Bin Packing | Minimize bins needed to fit items |
| [`solve_job_shop`](job-shop.md) | Job Shop Scheduling | Minimize makespan for jobs on machines |
| [`solve_vrptw`](vrp.md) | Vehicle Routing | Route vehicles with time windows |

## When to Use

**Use specialized solvers when:**

- Your problem matches the solver's structure exactly
- You want good solutions fast without modeling effort

**Use general solvers (MILP, CP-SAT, metaheuristics) when:**

- You have additional constraints
- The problem is a variation of the classic
- You need optimality proofs

## Quick Example

```python
from solvor import solve_knapsack, solve_bin_pack, solve_job_shop

# Knapsack
values = [60, 100, 120]
weights = [10, 20, 30]
result = solve_knapsack(values, weights, capacity=50)
print(result.solution)  # [1, 1, 1]
print(result.objective)  # 220

# Bin Packing
items = [4, 8, 1, 4, 2, 1]
result = solve_bin_pack(items, bin_capacity=10)
print(result.solution)  # [[8, 2], [4, 4, 1, 1]]

# Job Shop
jobs = [[(0, 3), (1, 2)], [(1, 2), (0, 4)]]
result = solve_job_shop(jobs, n_machines=2)
print(result.objective)  # Makespan
```

## See Also

- [Metaheuristics](../metaheuristics/index.md) - For custom variants
- [Cookbook: Knapsack](../../cookbook/knapsack.md) - Full example
- [Cookbook: Bin Packing](../../cookbook/bin-packing.md) - Full example

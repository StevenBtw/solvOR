# Classic Problems

Benchmark-grade implementations of classic OR problems.

## Traveling Salesman (TSP)

Route through all cities, minimizing total distance. Uses TSPLIB `att48` benchmark (48 cities, known optimum: 10628).

### Approaches

- [tsp_tabu.py](https://github.com/StevenBtw/solvOR/blob/main/examples/classic/tsp/tsp_tabu.py) - Tabu search with 2-opt
- [tsp_anneal.py](https://github.com/StevenBtw/solvOR/blob/main/examples/classic/tsp/tsp_anneal.py) - Simulated annealing
- [tsp_genetic.py](https://github.com/StevenBtw/solvOR/blob/main/examples/classic/tsp/tsp_genetic.py) - Genetic algorithm with OX crossover

### Quick Example

```python
from solvor import solve_tsp

distances = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]
result = solve_tsp(distances)
print(result.solution)   # Tour order
print(result.objective)  # Total distance
```

## Knapsack

Select items to maximize value within weight capacity.

### Approaches

- [knapsack_01.py](https://github.com/StevenBtw/solvOR/blob/main/examples/classic/knapsack/knapsack_01.py) - Dynamic programming
- [knapsack_milp.py](https://github.com/StevenBtw/solvOR/blob/main/examples/classic/knapsack/knapsack_milp.py) - MILP formulation

### Quick Example

```python
from solvor import solve_knapsack

values = [60, 100, 120]
weights = [10, 20, 30]
result = solve_knapsack(values, weights, capacity=50)
print(result.solution)  # [1, 1, 1]
```

## Vehicle Routing (VRPTW)

Route vehicles to customers with time windows. Uses Solomon R101 benchmark.

- [vrp_time_windows.py](https://github.com/StevenBtw/solvOR/blob/main/examples/classic/vehicle_routing/vrp_time_windows.py)

```python
from solvor import Customer, solve_vrptw

customers = [
    Customer(x=0, y=0, demand=0, ready=0, due=100, service=0),  # Depot
    Customer(x=2, y=3, demand=10, ready=0, due=50, service=5),
    # ...
]
result = solve_vrptw(customers, vehicle_capacity=30, n_vehicles=5)
```

## Bin Packing

Minimize bins to fit items. Uses Falkenauer benchmarks.

- [bin_packing_benchmark.py](https://github.com/StevenBtw/solvOR/blob/main/examples/classic/bin_packing/bin_packing_benchmark.py)

```python
from solvor import solve_bin_pack

items = [4, 8, 1, 4, 2, 1]
result = solve_bin_pack(items, bin_capacity=10)
print(result.objective)  # Number of bins
```

## Job Shop Scheduling

Minimize makespan for jobs on machines. Uses Fisher-Thompson ft06 benchmark.

- [ft06_benchmark.py](https://github.com/StevenBtw/solvOR/blob/main/examples/classic/job_shop_scheduling/ft06_benchmark.py)

```python
from solvor import solve_job_shop

jobs = [[(0, 3), (1, 2)], [(1, 2), (0, 4)]]
result = solve_job_shop(jobs, n_machines=2)
print(result.objective)  # Makespan
```

## Assignment

Match workers to tasks at minimum cost.

- [hungarian_demo.py](https://github.com/StevenBtw/solvOR/blob/main/examples/classic/assignment/hungarian_demo.py)

```python
from solvor import solve_hungarian

costs = [[10, 5], [3, 9]]
result = solve_hungarian(costs)
print(result.solution)  # Column for each row
```

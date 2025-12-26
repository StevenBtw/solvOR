# Quick Examples

Minimal API demos for every solver. Each example is 10-20 lines and shows the essential pattern.

## Linear & Integer Programming

### LP

```python
from solvor import solve_lp

# Minimize 2x + 3y subject to x + y >= 4, x <= 3
result = solve_lp(
    c=[2, 3],
    A=[[-1, -1], [1, 0]],  # Ax <= b
    b=[-4, 3]
)
print(result.solution)  # Optimal x, y
```

### MILP

```python
from solvor import solve_milp

# Same, but x must be integer
result = solve_milp(c=[2, 3], A=[[-1, -1], [1, 0]], b=[-4, 3], integers=[0])
print(result.solution)
```

## Constraint Programming

### SAT

```python
from solvor import solve_sat

# (x1 OR x2) AND (NOT x1 OR x3) AND (NOT x2 OR NOT x3)
result = solve_sat([[1, 2], [-1, 3], [-2, -3]])
print(result.solution)  # {1: True, 2: False, 3: True}
```

### CP-SAT Model

```python
from solvor import Model

m = Model()
x = m.int_var(1, 9, 'x')
y = m.int_var(1, 9, 'y')
z = m.int_var(1, 9, 'z')
m.add(m.all_different([x, y, z]))
m.add(x + y + z == 15)
result = m.solve()
print(result.solution)
```

## Metaheuristics

### Simulated Annealing

```python
from solvor import anneal
import random

def objective(x): return sum(xi**2 for xi in x)
def neighbor(x):
    x = list(x)
    x[random.randint(0, len(x)-1)] += random.uniform(-0.5, 0.5)
    return x

result = anneal([5, 5, 5], objective, neighbor)
print(result.solution)
```

### Tabu Search

```python
from solvor import tabu_search

def neighbors(perm):
    for i in range(len(perm)):
        for j in range(i+2, len(perm)):
            new = list(perm)
            new[i], new[j] = new[j], new[i]
            yield (i, j), tuple(new)

result = tabu_search([0, 3, 1, 4, 2], objective, neighbors)
```

## Gradient-Based

### Adam

```python
from solvor import adam

def grad(x): return [2*x[0], 2*x[1]]

result = adam(grad, x0=[5.0, 5.0])
print(result.solution)  # Close to [0, 0]
```

### Bayesian Optimization

```python
from solvor import bayesian_opt

def expensive(x): return (x[0]-2)**2 + (x[1]+1)**2

result = bayesian_opt(expensive, bounds=[(-5, 5), (-5, 5)], max_iter=30)
print(result.solution)
```

## Graph Algorithms

### Dijkstra

```python
from solvor import dijkstra

graph = {'A': [('B', 1), ('C', 4)], 'B': [('C', 2)], 'C': []}
result = dijkstra('A', 'C', lambda n: graph[n])
print(result.solution)  # ['A', 'B', 'C']
```

### A* Grid

```python
from solvor import astar_grid

maze = [[0, 0, 1, 0], [0, 0, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0]]
result = astar_grid(maze, start=(0, 0), goal=(3, 3), blocked=1)
print(result.solution)
```

### Max Flow

```python
from solvor import max_flow

graph = {
    's': [('a', 10, 0), ('b', 5, 0)],
    'a': [('t', 5, 0), ('b', 15, 0)],
    'b': [('t', 10, 0)],
    't': []
}
result = max_flow(graph, source='s', sink='t')
print(result.objective)  # 15
```

### MST

```python
from solvor import kruskal

edges = [(0, 1, 4), (0, 2, 3), (1, 2, 2), (1, 3, 5)]
result = kruskal(n_nodes=4, edges=edges)
print(result.objective)  # Total weight
```

## Assignment

### Hungarian

```python
from solvor import solve_hungarian

costs = [[10, 5, 13], [3, 9, 18], [10, 6, 12]]
result = solve_hungarian(costs)
print(result.solution)  # [1, 0, 2]
```

## Combinatorial

### Knapsack

```python
from solvor import solve_knapsack

values = [60, 100, 120]
weights = [10, 20, 30]
result = solve_knapsack(values, weights, capacity=50)
print(result.solution)  # [1, 1, 1]
```

### Exact Cover

```python
from solvor import solve_exact_cover

matrix = [[1, 0, 1], [0, 1, 1], [1, 1, 0]]
result = solve_exact_cover(matrix)
print(result.solution)
```

## Full Examples

See the [examples/ folder](https://github.com/StevenBtw/solvOR/tree/main/examples) for complete, runnable files.

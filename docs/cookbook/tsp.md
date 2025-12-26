# Traveling Salesman Problem

Find the shortest tour visiting all cities exactly once and returning to start.

## The Problem

Given N cities and distances between them, find the shortest possible route that visits each city exactly once and returns to the starting city.

This is the classic TSP, NP-hard, but small instances (< 100 cities) are very solvable with tabu search.

## Why It's Hard

With N cities, there are (N-1)!/2 possible tours. For 10 cities that's 181,440 tours. For 20 cities it's around 60 quintillion. Brute force isn't an option.

## Example

```python
from solvor import solve_tsp

# Distance matrix for 5 cities
distances = [
    [0, 29, 20, 21, 16],
    [29, 0, 15, 29, 28],
    [20, 15, 0, 15, 14],
    [21, 29, 15, 0, 4],
    [16, 28, 14, 4, 0]
]

result = solve_tsp(distances, max_iter=1000)

print(f"Best tour: {result.solution}")
print(f"Total distance: {result.objective}")
print(f"Iterations: {result.iterations}")

# Example output:
# Best tour: [0, 2, 4, 3, 1]
# Total distance: 73
```

## How It Works

`solve_tsp` uses tabu search with 2-opt moves:

1. **Initialize:** Start with nearest-neighbor heuristic
2. **2-opt neighborhood:** For each pair of edges, try reversing the segment between them
3. **Tabu list:** Remember recent moves to prevent cycling
4. **Iterate:** Always pick best non-tabu move
5. **Stop:** After max iterations or no improvement

## Variations

### Asymmetric TSP

```python
# Different distances each direction
distances = [
    [0, 10, 20],
    [15, 0, 25],  # dist[0][1] != dist[1][0]
    [30, 35, 0]
]
result = solve_tsp(distances)
```

### Multiple Salesmen (Vehicle Routing)

For VRP, see `solve_vrptw` or use MILP formulation.

## Real-World Applications

- **Logistics:** Delivery routes, warehouse picking
- **Manufacturing:** PCB drilling, robotic arm movement
- **DNA Sequencing:** Fragment assembly

## Performance Tips

1. **Tune max_iter:** More iterations = better solution, but diminishing returns
2. **Multiple runs:** Run with different seeds, take best result
3. **Preprocessing:** Remove impossible edges, use symmetry

## See Also

- [Metaheuristics](../algorithms/metaheuristics/index.md)

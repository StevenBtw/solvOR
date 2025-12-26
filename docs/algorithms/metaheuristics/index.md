# Metaheuristics

When you're climbing a mountain in the fog with no map, metaheuristics are your guide. These are the "good enough, fast enough" algorithms that explore solution spaces without guarantees, but with impressive practical results.

## Solvers

| Solver | Strategy | Best For |
|--------|----------|----------|
| [`anneal`](anneal.md) | Accepts worse solutions probabilistically | Quick setup, escaping local optima |
| [`tabu_search`](tabu.md) | Greedy + memory to prevent cycling | Reproducible results, debugging |
| [`evolve`](genetic.md) | Population-based crossover/mutation | Multi-objective, high diversity |
| [`lns`/`alns`](lns.md) | Destroy-and-repair | Large problems, structured solutions |

## When to Use

**Perfect for:**

- No gradient information available
- Objective function is black-box or noisy
- Many local optima (non-convex landscapes)
- Combinatorial optimization (TSP, job shop scheduling)
- Fast prototyping: implement a neighbor function, you're done

## When NOT to Use

- **Need provable optimality** - Use MILP, CP-SAT, or exact algorithms
- **Have gradient information** - Use gradient descent or Adam
- **Convex problems** - Use simplex or gradient methods

## Quick Example

```python
from solvor import anneal, solve_tsp

# Anneal: Minimize a function
def objective(x):
    return sum(xi**2 for xi in x)

def neighbor(x):
    import random
    i = random.randint(0, len(x)-1)
    x_new = list(x)
    x_new[i] += random.uniform(-0.5, 0.5)
    return x_new

result = anneal([1, 2, 3], objective, neighbor, max_iter=10000)
print(result.solution)  # Close to [0, 0, 0]

# TSP: Solve traveling salesman
distances = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]
result = solve_tsp(distances)
print(result.solution)  # Tour like [0, 1, 3, 2]
```

## Decision Tree

```
Do you have gradient info? → Use gradient descent, not metaheuristics

Is each evaluation expensive (>1 second)?
  Yes → bayesian_opt
  No → Continue

Do you need population diversity?
  Yes → evolve
  No → Continue

Do you value determinism and debugging?
  Yes → tabu_search
  No → anneal
```

## See Also

- [Continuous Optimization](../continuous/index.md) - When you have gradients
- [Cookbook: TSP](../../cookbook/tsp.md) - Full traveling salesman example

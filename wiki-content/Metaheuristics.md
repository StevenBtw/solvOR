# Metaheuristics

When you're climbing a mountain in the fog with no map, metaheuristics are your guide. These are the "good enough, fast enough" algorithms that explore solution spaces without guarantees, but with impressive practical results. Trading optimality certificates for speed and flexibility.

## Solvers in this Category

### [anneal](anneal)
Simulated annealing. Accepts worse solutions probabilistically, cooling down over time. Like a ball rolling on a landscape, energetic enough early on to escape local valleys, settling into the best valley it finds.

**Complexity:** O(iterations × neighbor_cost)
**Guarantees:** None, but often finds excellent solutions

### [tabu_search](tabu_search)
Greedy local search with memory. Always picks the best neighbor, but maintains a "tabu list" of recent moves to prevent cycling. More deterministic than annealing, easier to debug.

**Complexity:** O(iterations × neighbors × evaluation_cost)
**Guarantees:** None, deterministic for fixed seed

### [solve_tsp](solve_tsp)
Traveling salesman with tabu search. Built-in 2-opt neighborhood and nearest-neighbor initialization. Just pass a distance matrix.

**Complexity:** O(n² iterations) for n cities
**Guarantees:** Heuristic solution, no optimality bound

### [evolve](evolve)
Genetic algorithm. Maintains a population, combines solutions via crossover, occasionally mutates. Slower than single-solution methods but explores more diversity. Excels at multi-objective problems.

**Complexity:** O(generations × population × evaluation_cost)
**Guarantees:** None

### [bayesian_opt](bayesian_opt)
For expensive function evaluations. Builds a surrogate model (Gaussian process) to guess where to sample next. Use when each evaluation costs real time/money.

**Complexity:** O(iterations × dimensions²)
**Guarantees:** None, but sample-efficient

## When to Use This Category

**Perfect for:**
- No gradient information available
- Objective function is black-box or noisy
- Many local optima (non-convex landscapes)
- Combinatorial optimization (TSP, job shop scheduling)
- Fast prototyping, implement a neighbor function, you're done
- Multi-objective optimization (Pareto fronts)

**The magic:** Flexible, fast to implement, handle arbitrary objectives.

## When NOT to Use This Category

- **Need provable optimality** → Use MILP, CP-SAT, or exact algorithms
- **Have gradient information** → Use gradient descent or Adam
- **Convex problems** → Use simplex or gradient methods
- **Tiny search spaces** → Just enumerate

## Comparing Solvers

| Solver | Strategy | Deterministic? | Best For |
|--------|----------|----------------|----------|
| `anneal` | Single solution, probabilistic | No | Quick setup, escaping local optima |
| `tabu_search` | Single solution, greedy + memory | Yes (given seed) | Reproducible results, debugging |
| `solve_tsp` | Tabu with TSP-specific neighbors | Yes (given seed) | Traveling salesman specifically |
| `evolve` | Population-based | No | Multi-objective, high diversity |
| `bayesian_opt` | Surrogate model | No | Expensive evaluations (10-100 total) |

## Quick Example

```python
from solvor import anneal, tabu_search, evolve, solve_tsp

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
print(result.objective)  # Total distance
```

## Tips & Tricks

### For Annealing
- **Higher temperature = more exploration** - Start hot to escape local optima
- **Slower cooling = better solutions** - But takes longer
- **Small neighbor moves** - Don't teleport randomly, make local perturbations

### For Tabu Search
- **Cooldown length matters**, to short: cycles. Too long: missed opportunities
- **Diversification**, if stuck, restart or add random moves
- **Aspiration criteria**, override tabu if you find a new best

### For Genetic Algorithms
- **Crossover is critical**, bad crossover = expensive random search
- **Balance exploration/exploitation** - High mutation rate = more exploration
- **Elite preservation**, keep the best solutions across generations

### For Bayesian Optimization
- **Use for 10-100 evaluations**, below that, random search is fine. Above that, other methods are faster
- **Low dimensions work best**, struggles above 15-20 dimensions
- **Initial random sampling**, start with 5-10 random points to seed the model

## Real-World Applications

- **Traveling Salesman:** Vehicle routing, PCB drilling, warehouse picking
- **Job Shop Scheduling:** Minimize makespan with complex machine/task constraints
- **Hyperparameter Tuning:** Optimize ML model parameters (use bayesian_opt)
- **Portfolio Optimization:** Non-convex objectives with transaction costs
- **Game AI:** Tune parameters when evaluation = playing full games

## Decision Tree: Which Metaheuristic?

```
Do you have gradient info? → Use gradient descent, not metaheuristics

Is each evaluation expensive (>1 second)?
  Yes → bayesian_opt
  No → Continue

Do you need population diversity? (multi-objective, broad exploration)
  Yes → evolve
  No → Continue

Do you value determinism and debugging?
  Yes → tabu_search
  No → anneal

Is it specifically TSP?
  Yes → solve_tsp
```

## See Also

- [Continuous Optimization](Continuous-Optimization) - When you have gradients
- [Linear & Integer Programming](Linear-&-Integer-Programming) - When you need optimality proofs
- [Cookbook: TSP](Cookbook-TSP) - Full traveling salesman example

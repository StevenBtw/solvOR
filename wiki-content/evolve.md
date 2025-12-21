# evolve

**Genetic algorithm, population-based search with crossover and mutation.**

## At a Glance

- **Category:** [Metaheuristics](Metaheuristics)
- **Problem Type:** Multi-objective, exploration-heavy
- **Complexity:** O(generations × population × eval_cost)
- **Guarantees:** None
- **Status:** Returns `FEASIBLE`

## When to Use This

Best for multi-objective optimization or when searching in the dark. More overhead than single-solution methods but better diversity.

## Quick Example

```python
from solvor import evolve

def objective(genome):
    return sum(genome)

def crossover(p1, p2):
    mid = len(p1) // 2
    return p1[:mid] + p2[mid:]

def mutate(genome):
    import random
    i = random.randint(0, len(genome)-1)
    genome = list(genome)
    genome[i] = random.random()
    return genome

population = [[random.random() for _ in range(10)] for _ in range(50)]
result = evolve(objective, population, crossover, mutate)
```

## See Also
- [Metaheuristics](Metaheuristics)

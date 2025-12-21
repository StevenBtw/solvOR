# bayesian_opt

**Bayesian optimization for expensive function evaluations.**

## Summary 

- **Category:** [Metaheuristics](Metaheuristics)
- **Problem Type:** Expensive black-box optimization
- **Complexity:** O(iterations × dimensions²)
- **Guarantees:** None (sample-efficient)
- **Status:** Returns `FEASIBLE`

## When to Use This

Use when each evaluation costs real time or money. 10-100 evaluations total. Hyperparameter tuning, A/B testing, simulation optimization.

## Quick Example

```python
from solvor import bayesian_opt

def expensive_objective(x):
    # Imagine this takes 10 seconds
    return (x[0]-2)**2 + (x[1]+1)**2

result = bayesian_opt(expensive_objective, bounds=[(-5,5), (-5,5)], max_iter=30)
print(result.solution)  # Close to [2, -1] with only 30 evaluations
```

## See Also
- [Continuous Optimization](Continuous-Optimization)

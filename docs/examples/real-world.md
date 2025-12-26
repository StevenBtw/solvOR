# Real World

Practical applications with realistic constraints.

## Nurse Scheduling

Assign nurses to shifts while respecting regulations and preferences.

[nurse_scheduling.py](https://github.com/StevenBtw/solvOR/blob/main/examples/real_world/nurse_scheduling.py)

### Constraints

- Each shift must be covered
- Maximum shifts per nurse per week
- Minimum rest between shifts
- No more than N consecutive days
- Skill requirements

### Approach

Uses CP-SAT model with `all_different` and sum constraints.

```python
from solvor import Model

def schedule_nurses(nurses, shifts, days):
    m = Model()

    # x[nurse][day][shift] = 1 if nurse works shift on day
    x = {}
    for n in nurses:
        for d in range(days):
            for s in shifts:
                x[n, d, s] = m.bool_var(f'{n}_{d}_{s}')

    # Each shift covered by exactly one nurse
    for d in range(days):
        for s in shifts:
            m.add(sum(x[n, d, s] for n in nurses) == 1)

    # Max 5 shifts per nurse per week
    for n in nurses:
        m.add(sum(x[n, d, s] for d in range(days) for s in shifts) <= 5)

    return m.solve()
```

## School Timetabling

Assign classes to rooms and time slots without conflicts.

Uses ITC 2007 competition format.

### Approaches

Multiple solver approaches compared:

- [timetabling_sat.py](https://github.com/StevenBtw/solvOR/blob/main/examples/real_world/school_timetabling/timetabling_sat.py) - SAT
- [timetabling_cp.py](https://github.com/StevenBtw/solvOR/blob/main/examples/real_world/school_timetabling/timetabling_cp.py) - CP-SAT
- [timetabling_milp.py](https://github.com/StevenBtw/solvOR/blob/main/examples/real_world/school_timetabling/timetabling_milp.py) - MILP
- [timetabling_anneal.py](https://github.com/StevenBtw/solvOR/blob/main/examples/real_world/school_timetabling/timetabling_anneal.py) - Simulated annealing
- [timetabling_genetic.py](https://github.com/StevenBtw/solvOR/blob/main/examples/real_world/school_timetabling/timetabling_genetic.py) - Genetic algorithm
- [timetabling_tabu.py](https://github.com/StevenBtw/solvOR/blob/main/examples/real_world/school_timetabling/timetabling_tabu.py) - Tabu search

### Constraints

- No teacher teaches two classes at once
- No room hosts two classes at once
- Class-room compatibility (capacity, equipment)
- Teacher availability
- Curriculum requirements

## Machine Learning

### Linear Regression

Gradient descent for least squares regression.

[linear_regression.py](https://github.com/StevenBtw/solvOR/blob/main/examples/machine_learning/linear_regression.py)

```python
from solvor import adam

def grad(weights):
    # Compute gradient of MSE loss
    return gradient

result = adam(grad, x0=initial_weights, lr=0.01)
```

### Logistic Regression

Binary classification with Adam optimizer.

[logistic_regression.py](https://github.com/StevenBtw/solvOR/blob/main/examples/machine_learning/logistic_regression.py)

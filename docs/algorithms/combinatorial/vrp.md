# solve_vrptw

Vehicle Routing Problem with Time Windows. Route vehicles to serve customers within capacity and time constraints.

## Example

```python
from solvor import Customer, solve_vrptw

# Depot at (0,0), customers at various locations
customers = [
    Customer(x=0, y=0, demand=0, ready=0, due=100, service=0),  # Depot
    Customer(x=2, y=3, demand=10, ready=0, due=50, service=5),
    Customer(x=5, y=1, demand=15, ready=10, due=60, service=5),
    Customer(x=3, y=4, demand=20, ready=20, due=70, service=5),
]

result = solve_vrptw(
    customers,
    vehicle_capacity=30,
    n_vehicles=2
)
print(result.solution)   # Routes for each vehicle
print(result.objective)  # Total distance
```

## Customer

```python
@dataclass
class Customer:
    x: float            # X coordinate
    y: float            # Y coordinate
    demand: float       # Demand to serve
    ready: float        # Earliest service time
    due: float          # Latest service time
    service: float      # Service duration
```

## The Problem

- Each vehicle starts and ends at depot
- Visit all customers within time windows
- Don't exceed vehicle capacity
- Minimize total distance

## See Also

- [Cookbook: TSP](../../cookbook/tsp.md) - Single vehicle, no constraints
- [Metaheuristics](../metaheuristics/index.md) - For larger instances

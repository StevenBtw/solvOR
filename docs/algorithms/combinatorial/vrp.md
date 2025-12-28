# solve_vrptw

Vehicle Routing Problem with Time Windows. Route vehicles to serve customers within capacity and time constraints. Uses Adaptive Large Neighborhood Search (ALNS).

## Signature

```python
def solve_vrptw(
    customers: list[Customer] | list[tuple],
    vehicles: int | list[Vehicle],
    depot: tuple[float, float] = (0.0, 0.0),
    *,
    vehicle_capacity: float = float("inf"),
    distance_weight: float = 1.0,
    vehicle_weight: float = 0.0,
    tw_penalty: float = 1000.0,
    capacity_penalty: float = 1000.0,
    sync_penalty: float = 10000.0,
    max_iter: int = 10000,
    max_no_improve: int = 500,
    seed: int | None = None,
    on_progress: ProgressCallback | None = None,
    progress_interval: int = 0,
) -> Result[VRPState]
```

## Parameters

| Parameter | Description |
|-----------|-------------|
| `customers` | List of Customer objects or tuples (id, x, y, demand, tw_start, tw_end, service_time, required_vehicles) |
| `vehicles` | Number of vehicles (int) or list of Vehicle objects |
| `depot` | Depot coordinates (x, y), default (0, 0) |
| `vehicle_capacity` | Capacity per vehicle (when using int for vehicles) |
| `distance_weight` | Weight for total distance in objective |
| `vehicle_weight` | Weight for number of vehicles used |
| `tw_penalty` | Penalty per unit of time window violation |
| `capacity_penalty` | Penalty per unit of capacity violation |
| `sync_penalty` | Penalty for multi-resource sync violations |
| `max_iter` | Maximum ALNS iterations |
| `max_no_improve` | Stop after this many iterations without improvement |
| `seed` | Random seed for reproducibility |
| `on_progress` | Progress callback (return True to stop early) |
| `progress_interval` | Call progress every N iterations (0 = disabled) |

## Example

```python
from solvor import Customer, solve_vrptw

# Customers at various locations (depot is separate)
customers = [
    Customer(id=1, x=2, y=3, demand=10, tw_start=0, tw_end=50, service_time=5),
    Customer(id=2, x=5, y=1, demand=15, tw_start=10, tw_end=60, service_time=5),
    Customer(id=3, x=3, y=4, demand=20, tw_start=20, tw_end=70, service_time=5),
]

result = solve_vrptw(
    customers,
    vehicles=2,
    depot=(0, 0),
    vehicle_capacity=30
)
print(result.solution.routes)  # Routes for each vehicle
print(result.objective)        # Total distance
```

## Customer

```python
@dataclass
class Customer:
    id: int                        # Unique customer ID
    x: float                       # X coordinate
    y: float                       # Y coordinate
    demand: float = 0.0            # Demand to serve
    tw_start: float = 0.0          # Earliest service time
    tw_end: float = inf            # Latest service time
    service_time: float = 0.0      # Service duration
    required_vehicles: int = 1     # For multi-resource visits
```

## The Problem

- Each vehicle starts and ends at depot
- Visit all customers within time windows
- Don't exceed vehicle capacity
- Minimize total distance

## See Also

- [Cookbook: TSP](../../cookbook/tsp.md) - Single vehicle, no constraints
- [Metaheuristics](../metaheuristics/index.md) - For larger instances

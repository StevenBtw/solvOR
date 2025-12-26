"""
Vehicle Routing Problem (VRP) Example

Route vehicles to serve customers with time windows and capacity constraints.
"""

from solvor import Customer, solve_vrptw

# Customers with locations, demands, and time windows
customers = [
    Customer(id=1, x=10, y=20, demand=5, tw_start=0, tw_end=50),
    Customer(id=2, x=30, y=10, demand=8, tw_start=10, tw_end=60),
    Customer(id=3, x=20, y=30, demand=3, tw_start=5, tw_end=40),
    Customer(id=4, x=40, y=25, demand=6, tw_start=20, tw_end=70),
]

result = solve_vrptw(
    customers,
    vehicles=2,
    depot=(0, 0),
    vehicle_capacity=15,
    max_iter=100,
    seed=42,
)
print(f"Routes: {result.solution.routes}")
print(f"Total distance: {result.objective:.2f}")

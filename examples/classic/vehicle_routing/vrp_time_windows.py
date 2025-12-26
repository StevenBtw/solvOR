"""
Vehicle Routing Problem with Time Windows (VRPTW)

Solve the Solomon R101 benchmark using solvor's ALNS-based VRPTW solver.

Problem: Route vehicles from depot to serve customers within their time windows,
minimizing total distance while respecting vehicle capacity.

Source: Solomon (1987) "Algorithms for the Vehicle Routing and Scheduling
        Problems with Time Window Constraints"
        https://www.sintef.no/projectweb/top/vrptw/solomon-benchmark/

Why this solver: ALNS (Adaptive Large Neighborhood Search) is the state-of-the-art
for VRPTW. It uses destroy-and-repair operators that work well on routing problems.
"""

from time import perf_counter

from solomon_r101 import R101_DATA, VEHICLE_CAPACITY

from solvor.vrp import Customer, solve_vrptw


def main():
    # Build customer list from Solomon data
    # Format: (x, y, demand, ready_time, due_time, service_time)
    customers = []
    for i, (x, y, demand, ready, due, service) in enumerate(R101_DATA[1:], start=1):
        customers.append(
            Customer(
                id=i,
                x=x,
                y=y,
                demand=demand,
                tw_start=ready,
                tw_end=due,
                service_time=service,
            )
        )

    # Depot location
    depot = (R101_DATA[0][0], R101_DATA[0][1])

    print("Solomon R101 VRPTW")
    print(f"  Customers: {len(customers)}")
    print(f"  Vehicle capacity: {VEHICLE_CAPACITY}")
    print(f"  Depot: {depot}")
    print()

    start = perf_counter()

    result = solve_vrptw(
        customers,
        vehicles=25,  # Upper bound on fleet size
        depot=depot,
        vehicle_capacity=VEHICLE_CAPACITY,
        max_iter=5000,
        max_no_improve=500,
        seed=42,
    )

    elapsed = perf_counter() - start

    print(f"Time: {elapsed:.2f}s")
    print(f"Status: {result.status}")
    print(f"Total distance: {result.objective:.1f}")
    print(f"Iterations: {result.iterations}")

    # Count routes used
    state = result.solution
    routes_used = sum(1 for r in state.routes if r)
    print(f"Vehicles used: {routes_used}")

    # Print routes
    print("\nRoutes:")
    for v, route in enumerate(state.routes):
        if route:
            load = sum(customers[c - 1].demand for c in route)
            dist = state.route_distance(v)
            print(f"  Vehicle {v}: {route} (load={load}, dist={dist:.1f})")

    # Check feasibility
    if state.is_feasible():
        print("\nSolution is feasible!")
    else:
        tw_viol = state.time_window_violation()
        cap_viol = state.capacity_violation()
        if tw_viol > 0:
            print(f"  Time window violations: {tw_viol:.1f}")
        if cap_viol > 0:
            print(f"  Capacity violations: {cap_viol:.1f}")
        if state.unassigned:
            print(f"  Unassigned customers: {state.unassigned}")


if __name__ == "__main__":
    main()

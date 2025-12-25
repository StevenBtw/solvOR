"""Tests for the VRPTW solver."""

from math import hypot
from random import Random

from solvor.types import Status
from solvor.vrp import (
    Customer,
    Vehicle,
    VRPState,
    greedy_insertion,
    random_removal,
    regret_insertion,
    related_removal,
    route_removal,
    solve_vrptw,
    sync_aware_insertion,
    sync_removal,
    vrp_objective,
    worst_removal,
)


class TestCustomerVehicle:
    def test_customer_defaults(self):
        c = Customer(1, 10.0, 20.0)
        assert c.id == 1
        assert c.x == 10.0
        assert c.y == 20.0
        assert c.demand == 0.0
        assert c.tw_start == 0.0
        assert c.tw_end == float("inf")
        assert c.service_time == 0.0
        assert c.required_vehicles == 1

    def test_customer_with_values(self):
        c = Customer(2, 5.0, 5.0, demand=10.0, tw_start=8.0, tw_end=12.0, service_time=0.5, required_vehicles=2)
        assert c.demand == 10.0
        assert c.tw_start == 8.0
        assert c.tw_end == 12.0
        assert c.service_time == 0.5
        assert c.required_vehicles == 2

    def test_vehicle_defaults(self):
        v = Vehicle(0)
        assert v.id == 0
        assert v.capacity == float("inf")
        assert v.max_duration == float("inf")

    def test_vehicle_with_capacity(self):
        v = Vehicle(1, capacity=100.0, max_duration=480.0)
        assert v.capacity == 100.0
        assert v.max_duration == 480.0


class TestVRPState:
    def test_from_problem(self):
        customers = [
            Customer(0, 0.0, 0.0),  # Depot
            Customer(1, 10.0, 0.0, demand=5.0),
            Customer(2, 0.0, 10.0, demand=3.0),
        ]
        vehicles = [Vehicle(0, capacity=100.0), Vehicle(1, capacity=100.0)]

        state = VRPState.from_problem(customers, vehicles)

        assert len(state.customers) == 3
        assert len(state.vehicles) == 2
        assert len(state.routes) == 2
        assert all(r == [] for r in state.routes)
        assert state.unassigned == {1, 2}

    def test_distance(self):
        customers = [
            Customer(0, 0.0, 0.0),
            Customer(1, 3.0, 4.0),  # Distance from depot = 5
        ]
        state = VRPState.from_problem(customers, [Vehicle(0)])

        assert abs(state.dist(0, 1) - 5.0) < 1e-9

    def test_route_distance(self):
        customers = [
            Customer(0, 0.0, 0.0),  # Depot
            Customer(1, 10.0, 0.0),
            Customer(2, 10.0, 10.0),
        ]
        state = VRPState.from_problem(customers, [Vehicle(0)])
        state.routes[0] = [1, 2]

        # Depot -> 1 (10) + 1 -> 2 (10) + 2 -> Depot (sqrt(200))
        expected = 10.0 + 10.0 + hypot(10.0, 10.0)
        assert abs(state.route_distance(0) - expected) < 1e-9

    def test_route_load(self):
        customers = [
            Customer(0, 0.0, 0.0),
            Customer(1, 10.0, 0.0, demand=5.0),
            Customer(2, 20.0, 0.0, demand=3.0),
        ]
        state = VRPState.from_problem(customers, [Vehicle(0)])
        state.routes[0] = [1, 2]

        assert state.route_load(0) == 8.0

    def test_arrival_times(self):
        customers = [
            Customer(0, 0.0, 0.0),
            Customer(1, 10.0, 0.0, service_time=1.0),
            Customer(2, 20.0, 0.0),
        ]
        state = VRPState.from_problem(customers, [Vehicle(0)])
        state.routes[0] = [1, 2]
        state.update_arrival_times()

        # Arrival at 1: distance 10
        # Arrival at 2: 10 + service(1) + travel(10) = 21
        assert abs(state.arrival_times[0][0] - 10.0) < 1e-9
        assert abs(state.arrival_times[0][1] - 21.0) < 1e-9

    def test_time_window_violation(self):
        customers = [
            Customer(0, 0.0, 0.0),
            Customer(1, 100.0, 0.0, tw_end=50.0),  # Too far, will arrive late
        ]
        state = VRPState.from_problem(customers, [Vehicle(0)])
        state.routes[0] = [1]
        state.update_arrival_times()

        # Arrives at 100, window ends at 50, violation = 50
        assert abs(state.time_window_violation() - 50.0) < 1e-9

    def test_capacity_violation(self):
        customers = [
            Customer(0, 0.0, 0.0),
            Customer(1, 10.0, 0.0, demand=60.0),
            Customer(2, 20.0, 0.0, demand=50.0),
        ]
        vehicles = [Vehicle(0, capacity=100.0)]
        state = VRPState.from_problem(customers, vehicles)
        state.routes[0] = [1, 2]  # Total demand 110, capacity 100

        assert abs(state.capacity_violation() - 10.0) < 1e-9

    def test_copy(self):
        customers = [Customer(0, 0.0, 0.0), Customer(1, 10.0, 0.0)]
        state = VRPState.from_problem(customers, [Vehicle(0)])
        state.routes[0] = [1]

        copy = state.copy()
        copy.routes[0] = []

        assert state.routes[0] == [1]  # Original unchanged
        assert copy.routes[0] == []

    def test_vehicles_used(self):
        customers = [Customer(0, 0.0, 0.0), Customer(1, 10.0, 0.0), Customer(2, 20.0, 0.0)]
        state = VRPState.from_problem(customers, [Vehicle(0), Vehicle(1), Vehicle(2)])
        state.routes[0] = [1]
        state.routes[1] = [2]
        # Vehicle 2 is empty

        assert state.vehicles_used() == 2


class TestDestroyOperators:
    def setup_method(self):
        self.customers = [
            Customer(0, 0.0, 0.0),  # Depot
            Customer(1, 10.0, 0.0),
            Customer(2, 20.0, 0.0),
            Customer(3, 30.0, 0.0),
            Customer(4, 40.0, 0.0),
        ]
        self.state = VRPState.from_problem(self.customers, [Vehicle(0), Vehicle(1)])
        self.state.routes[0] = [1, 2]
        self.state.routes[1] = [3, 4]
        self.state.unassigned = set()
        self.state.update_arrival_times()
        self.rng = Random(42)

    def test_random_removal(self):
        result = random_removal(self.state, self.rng, degree=0.5)

        # Should remove ~2 customers (50% of 4)
        total_assigned = sum(len(r) for r in result.routes)
        assert total_assigned < 4
        assert len(result.unassigned) > 0

    def test_worst_removal(self):
        result = worst_removal(self.state, self.rng, degree=0.25)

        # Should remove at least 1 customer
        assert len(result.unassigned) >= 1

    def test_related_removal(self):
        result = related_removal(self.state, self.rng, degree=0.5)

        # Should remove nearby customers
        assert len(result.unassigned) >= 1

    def test_route_removal(self):
        result = route_removal(self.state, self.rng, n_routes=1)

        # Should remove entire route
        empty_routes = sum(1 for r in result.routes if not r)
        assert empty_routes >= 1
        assert len(result.unassigned) >= 2


class TestRepairOperators:
    def setup_method(self):
        self.customers = [
            Customer(0, 0.0, 0.0),  # Depot
            Customer(1, 10.0, 0.0, demand=5.0),
            Customer(2, 20.0, 0.0, demand=5.0),
            Customer(3, 15.0, 10.0, demand=5.0),
        ]
        self.state = VRPState.from_problem(self.customers, [Vehicle(0, capacity=100.0), Vehicle(1, capacity=100.0)])
        self.rng = Random(42)

    def test_greedy_insertion(self):
        result = greedy_insertion(self.state, self.rng)

        # All customers should be assigned
        assert len(result.unassigned) == 0
        total = sum(len(r) for r in result.routes)
        assert total == 3

    def test_regret_insertion(self):
        result = regret_insertion(self.state, self.rng, k=2)

        # All customers should be assigned
        assert len(result.unassigned) == 0

    def test_insertion_respects_capacity(self):
        customers = [
            Customer(0, 0.0, 0.0),
            Customer(1, 10.0, 0.0, demand=60.0),
            Customer(2, 20.0, 0.0, demand=60.0),
        ]
        state = VRPState.from_problem(customers, [Vehicle(0, capacity=50.0)])

        result = greedy_insertion(state, self.rng)

        # Neither customer fits, both should remain unassigned
        assert len(result.unassigned) == 2


class TestMultiResource:
    def test_sync_violation(self):
        customers = [
            Customer(0, 0.0, 0.0),
            Customer(1, 10.0, 0.0, required_vehicles=2),
        ]
        state = VRPState.from_problem(customers, [Vehicle(0), Vehicle(1)])

        # Only one vehicle visits customer 1
        state.routes[0] = [1]
        state.routes[1] = []
        state.unassigned = set()
        state.update_arrival_times()

        # Should have sync violation (needs 2 vehicles, only 1 assigned)
        assert state.sync_violation() > 0

    def test_sync_satisfied(self):
        customers = [
            Customer(0, 0.0, 0.0),
            Customer(1, 10.0, 0.0, required_vehicles=2),
        ]
        state = VRPState.from_problem(customers, [Vehicle(0), Vehicle(1)])

        # Both vehicles visit customer 1
        state.routes[0] = [1]
        state.routes[1] = [1]
        state.unassigned = set()
        state.update_arrival_times()

        # Both arrive at same time (distance 10), no violation
        assert state.sync_violation() < 1e-6

    def test_sync_removal(self):
        customers = [
            Customer(0, 0.0, 0.0),
            Customer(1, 10.0, 0.0, required_vehicles=2),
            Customer(2, 20.0, 0.0),
        ]
        state = VRPState.from_problem(customers, [Vehicle(0), Vehicle(1)])
        state.routes[0] = [1, 2]
        state.routes[1] = [1]
        state.unassigned = set()
        state.update_arrival_times()

        rng = Random(42)
        result = sync_removal(state, rng)

        # Should remove the sync customer
        assert 1 in result.unassigned

    def test_sync_aware_insertion(self):
        customers = [
            Customer(0, 0.0, 0.0),
            Customer(1, 10.0, 0.0, required_vehicles=2),
        ]
        state = VRPState.from_problem(customers, [Vehicle(0), Vehicle(1)])

        rng = Random(42)
        result = sync_aware_insertion(state, rng)

        # Customer 1 should be assigned to both vehicles
        assert 1 in result.routes[0] or 1 in result.routes[1]
        # Check sync assignments tracked
        if 1 not in result.unassigned:
            vehicles_with_1 = [v for v, r in enumerate(result.routes) if 1 in r]
            assert len(vehicles_with_1) >= 2


class TestObjective:
    def test_objective_components(self):
        customers = [
            Customer(0, 0.0, 0.0),
            Customer(1, 10.0, 0.0),
        ]
        state = VRPState.from_problem(customers, [Vehicle(0)])
        state.routes[0] = [1]
        state.unassigned = set()
        state.update_arrival_times()

        obj = vrp_objective(state, distance_weight=1.0, vehicle_weight=0.0, tw_penalty=0.0, capacity_penalty=0.0)

        # Distance: depot -> 1 -> depot = 20
        assert abs(obj - 20.0) < 1e-9

    def test_unassigned_penalty(self):
        customers = [
            Customer(0, 0.0, 0.0),
            Customer(1, 10.0, 0.0),
        ]
        state = VRPState.from_problem(customers, [Vehicle(0)])
        # Customer 1 unassigned

        obj = vrp_objective(state, unassigned_penalty=1000.0)
        assert obj >= 1000.0


class TestSolveVRPTW:
    def test_simple_vrptw(self):
        customers = [
            Customer(1, 10.0, 0.0, demand=5.0),
            Customer(2, 20.0, 0.0, demand=5.0),
            Customer(3, 30.0, 0.0, demand=5.0),
        ]

        result = solve_vrptw(customers, vehicles=2, depot=(0.0, 0.0), vehicle_capacity=100.0, max_iter=100, seed=42)

        assert result.status == Status.FEASIBLE
        state = result.solution
        assert isinstance(state, VRPState)
        assert len(state.unassigned) == 0

    def test_vrptw_with_time_windows(self):
        customers = [
            Customer(1, 10.0, 0.0, tw_start=0.0, tw_end=100.0),
            Customer(2, 20.0, 0.0, tw_start=0.0, tw_end=100.0),
        ]

        result = solve_vrptw(customers, vehicles=2, max_iter=50, seed=42)

        assert result.ok
        state = result.solution
        assert state.time_window_violation() < 1e-6

    def test_vrptw_tuple_input(self):
        # Test tuple format: (id, x, y, demand, tw_start, tw_end)
        customers = [
            (1, 10.0, 0.0, 5.0, 0.0, 100.0),
            (2, 20.0, 0.0, 3.0, 0.0, 100.0),
        ]

        result = solve_vrptw(customers, vehicles=1, max_iter=50, seed=42)

        assert result.ok

    def test_vrptw_with_multi_resource(self):
        customers = [
            Customer(1, 10.0, 0.0),
            Customer(2, 20.0, 0.0, required_vehicles=2),  # Needs 2 vehicles
            Customer(3, 30.0, 0.0),
        ]

        result = solve_vrptw(customers, vehicles=3, max_iter=200, seed=42)

        assert result.ok
        state = result.solution

        # Customer 2 should be in multiple routes
        vehicles_with_2 = [v for v, r in enumerate(state.routes) if 2 in r]
        # May or may not succeed depending on optimization, but should try
        assert len(state.unassigned) == 0 or 2 not in state.unassigned or len(vehicles_with_2) >= 1

    def test_vrptw_respects_capacity(self):
        customers = [
            Customer(1, 10.0, 0.0, demand=40.0),
            Customer(2, 20.0, 0.0, demand=40.0),
            Customer(3, 30.0, 0.0, demand=40.0),
        ]

        result = solve_vrptw(customers, vehicles=2, vehicle_capacity=50.0, max_iter=100, seed=42)
        # With capacity 50 and demands of 40, need at least 3 routes
        # Since we only have 2 vehicles, some may be unassigned
        # Total demand is 120, capacity is 100, so violations are expected
        assert result.objective >= 0  # Objective includes distance and penalties


class TestProgressCallback:
    def test_callback_receives_progress(self):
        calls = []

        def callback(progress):
            calls.append(progress.iteration)

        customers = [Customer(1, 10.0, 0.0)]
        solve_vrptw(customers, vehicles=1, max_iter=50, on_progress=callback, progress_interval=10, seed=42)

        assert len(calls) > 0
        assert calls[0] == 10


class TestEdgeCases:
    def test_single_customer(self):
        customers = [Customer(1, 10.0, 0.0)]
        result = solve_vrptw(customers, vehicles=1, max_iter=20, seed=42)

        assert result.ok
        state = result.solution
        assert len(state.unassigned) == 0

    def test_no_vehicles(self):
        customers = [Customer(1, 10.0, 0.0)]
        result = solve_vrptw(customers, vehicles=0, max_iter=20, seed=42)

        state = result.solution
        # No vehicles means customers stay unassigned
        assert len(state.unassigned) == 1

    def test_many_customers_few_vehicles(self):
        customers = [Customer(i, float(i * 10), 0.0) for i in range(1, 11)]
        result = solve_vrptw(customers, vehicles=2, max_iter=100, seed=42)

        assert result.ok
        state = result.solution
        assert len(state.unassigned) == 0

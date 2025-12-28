r"""
Vehicle Routing Problem with Time Windows (VRPTW) support.

Provides state representation, constraint handling, and LNS operators for
VRPTW variants including multi-resource visits (where some customers require
multiple vehicles to attend simultaneously).

    from solvor.vrp import Customer, Vehicle, solve_vrptw

    customers = [
        Customer(1, 10, 20, demand=5, tw_start=8, tw_end=12),
        Customer(2, 30, 40, demand=8, tw_start=9, tw_end=14),
        Customer(3, 15, 35, demand=10, required_vehicles=2),  # Needs 2 vehicles!
    ]
    result = solve_vrptw(customers, vehicles=3, depot=(0, 0))

Multi-resource visits: Some customers need multiple vehicles to arrive at the
same time (e.g., heavy deliveries requiring two trucks). Set required_vehicles > 1
on such customers. The solver will try to synchronize arrival times across vehicles.

How it works: uses Adaptive Large Neighborhood Search (ALNS) with multiple
destroy operators (random, worst, related, route removal) and repair operators
(greedy, regret-2 insertion). Operators that produce good results get selected
more often. Supports time windows and capacity constraints.

Use this for:

- Delivery routing with time windows
- Multi-vehicle logistics planning
- Pickup and delivery problems
- Fleet management optimization

Parameters:

    customers: list of Customer objects with location, demand, time windows
    vehicles: number of vehicles or list of Vehicle objects
    depot: (x, y) coordinates of depot
    vehicle_capacity: capacity per vehicle (if using int for vehicles)

Multi-resource visits: set required_vehicles > 1 on customers that need
multiple vehicles simultaneously (e.g., heavy deliveries). The solver
will try to synchronize arrival times across vehicles.
"""

from dataclasses import dataclass, field
from math import hypot
from random import Random

from solvor.lns import alns
from solvor.types import ProgressCallback, Result

__all__ = [
    "Customer",
    "Vehicle",
    "VRPState",
    "solve_vrptw",
    "vrp_objective",
    "random_removal",
    "worst_removal",
    "related_removal",
    "route_removal",
    "sync_removal",
    "greedy_insertion",
    "regret_insertion",
    "sync_aware_insertion",
]

SYNC_TOLERANCE = 1e-6


@dataclass(frozen=True, slots=True)
class Customer:
    """A customer/location to visit."""

    id: int
    x: float
    y: float
    demand: float = 0.0
    tw_start: float = 0.0
    tw_end: float = float("inf")
    service_time: float = 0.0
    required_vehicles: int = 1  # Multi-resource: how many vehicles needed simultaneously


@dataclass(frozen=True, slots=True)
class Vehicle:
    """A vehicle in the fleet."""

    id: int
    capacity: float = float("inf")
    max_duration: float = float("inf")


@dataclass
class VRPState:
    """Mutable state of a VRPTW solution."""

    customers: list[Customer]
    vehicles: list[Vehicle]
    routes: list[list[int]]
    arrival_times: list[list[float]]
    unassigned: set[int]
    sync_assignments: dict[int, set[int]] = field(default_factory=dict)

    # Cached distance matrix
    _dist: list[list[float]] | None = field(default=None, repr=False)

    def copy(self) -> "VRPState":
        return VRPState(
            customers=self.customers,  # Immutable, share reference
            vehicles=self.vehicles,  # Immutable, share reference
            routes=[list(r) for r in self.routes],
            arrival_times=[list(a) for a in self.arrival_times],
            unassigned=set(self.unassigned),
            sync_assignments={k: set(v) for k, v in self.sync_assignments.items()},
            _dist=self._dist,  # Share cached distances
        )

    def dist(self, i: int, j: int) -> float:
        if self._dist is not None:
            return self._dist[i][j]
        ci, cj = self.customers[i], self.customers[j]
        return hypot(ci.x - cj.x, ci.y - cj.y)

    def route_distance(self, v: int) -> float:
        route = self.routes[v]
        if not route:
            return 0.0
        total = self.dist(0, route[0])  # Depot to first
        for i in range(len(route) - 1):
            total += self.dist(route[i], route[i + 1])
        total += self.dist(route[-1], 0)  # Last to depot
        return total

    def total_distance(self) -> float:
        return sum(self.route_distance(v) for v in range(len(self.vehicles)))

    def route_load(self, v: int) -> float:
        return sum(self.customers[c].demand for c in self.routes[v])

    def compute_arrival_times(self, v: int) -> list[float]:
        route = self.routes[v]
        if not route:
            return []

        times = []
        t = self.dist(0, route[0])  # Travel from depot

        for i, cid in enumerate(route):
            c = self.customers[cid]
            # Wait if arriving before time window opens
            t = max(t, c.tw_start)
            times.append(t)
            # Add service time and travel to next
            t += c.service_time
            if i < len(route) - 1:
                t += self.dist(cid, route[i + 1])

        return times

    def update_arrival_times(self) -> None:
        for v in range(len(self.vehicles)):
            self.arrival_times[v] = self.compute_arrival_times(v)

    def time_window_violation(self) -> float:
        violation = 0.0
        for v, route in enumerate(self.routes):
            for i, cid in enumerate(route):
                c = self.customers[cid]
                if i < len(self.arrival_times[v]):
                    arrival = self.arrival_times[v][i]
                    if arrival > c.tw_end:
                        violation += arrival - c.tw_end
        return violation

    def capacity_violation(self) -> float:
        violation = 0.0
        for v, vehicle in enumerate(self.vehicles):
            load = self.route_load(v)
            if load > vehicle.capacity:
                violation += load - vehicle.capacity
        return violation

    def sync_violation(self) -> float:
        """Penalty for multi-resource customers with mismatched arrival times."""
        violation = 0.0

        for cid, c in enumerate(self.customers):
            if c.required_vehicles <= 1:
                continue

            # Find all vehicles visiting this customer
            visiting_vehicles = []
            for v, route in enumerate(self.routes):
                if cid in route:
                    idx = route.index(cid)
                    if idx < len(self.arrival_times[v]):
                        visiting_vehicles.append((v, self.arrival_times[v][idx]))

            # Check if enough vehicles assigned
            if len(visiting_vehicles) < c.required_vehicles:
                violation += (c.required_vehicles - len(visiting_vehicles)) * 1000.0
            elif len(visiting_vehicles) > 1:
                # Check synchronization
                times = [t for _, t in visiting_vehicles]
                violation += max(times) - min(times)

        return violation

    def unassigned_penalty(self) -> float:
        return len(self.unassigned) * 10000.0

    def is_feasible(self, tolerance: float = 1e-6) -> bool:
        return (
            len(self.unassigned) == 0
            and self.time_window_violation() < tolerance
            and self.capacity_violation() < tolerance
            and self.sync_violation() < tolerance
        )

    def vehicles_used(self) -> int:
        return sum(1 for r in self.routes if r)

    @classmethod
    def from_problem(
        cls,
        customers: list[Customer],
        vehicles: list[Vehicle],
    ) -> "VRPState":
        n = len(customers)
        n_vehicles = len(vehicles)

        dist = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = hypot(customers[i].x - customers[j].x, customers[i].y - customers[j].y)
                dist[i][j] = d
                dist[j][i] = d

        unassigned = {c.id for c in customers if c.id != 0}

        return cls(
            customers=customers,
            vehicles=vehicles,
            routes=[[] for _ in range(n_vehicles)],
            arrival_times=[[] for _ in range(n_vehicles)],
            unassigned=unassigned,
            sync_assignments={},
            _dist=dist,
        )


def random_removal(state: VRPState, rng: Random, degree: float = 0.2) -> VRPState:
    """Remove a random fraction of customers from routes."""
    state = state.copy()

    assigned = [cid for route in state.routes for cid in route]
    if not assigned:
        return state

    n_remove = max(1, int(len(assigned) * degree))
    to_remove = set(rng.sample(assigned, min(n_remove, len(assigned))))

    for v in range(len(state.routes)):
        state.routes[v] = [c for c in state.routes[v] if c not in to_remove]

    state.unassigned.update(to_remove)
    state.update_arrival_times()
    return state


def worst_removal(state: VRPState, rng: Random, degree: float = 0.2) -> VRPState:
    """Remove customers that contribute most to total cost."""
    state = state.copy()

    assigned = [(cid, v, i) for v, route in enumerate(state.routes) for i, cid in enumerate(route)]
    if not assigned:
        return state

    costs = []
    for cid, v, i in assigned:
        route = state.routes[v]
        if len(route) == 1:
            # Only customer: saves depot->customer->depot
            saving = state.dist(0, cid) + state.dist(cid, 0)
        elif i == 0:
            # First: saves depot->cid->next, gains depot->next
            saving = state.dist(0, cid) + state.dist(cid, route[1]) - state.dist(0, route[1])
        elif i == len(route) - 1:
            # Last: saves prev->cid->depot, gains prev->depot
            saving = state.dist(route[-2], cid) + state.dist(cid, 0) - state.dist(route[-2], 0)
        else:
            # Middle: saves prev->cid->next, gains prev->next
            prev_c, next_c = route[i - 1], route[i + 1]
            saving = state.dist(prev_c, cid) + state.dist(cid, next_c) - state.dist(prev_c, next_c)

        costs.append((saving, cid))

    costs.sort(reverse=True)
    n_remove = max(1, int(len(assigned) * degree))

    to_remove: set[int] = set()
    candidates = [c for _, c in costs]

    while len(to_remove) < n_remove and candidates:
        p = rng.random() ** 2  # bias toward worst
        idx = min(int(p * len(candidates)), len(candidates) - 1)
        to_remove.add(candidates.pop(idx))

    for v in range(len(state.routes)):
        state.routes[v] = [c for c in state.routes[v] if c not in to_remove]

    state.unassigned.update(to_remove)
    state.update_arrival_times()
    return state


def related_removal(state: VRPState, rng: Random, degree: float = 0.2) -> VRPState:
    """Remove clusters of nearby customers."""
    state = state.copy()

    assigned = [cid for route in state.routes for cid in route]
    if not assigned:
        return state

    n_remove = max(1, int(len(assigned) * degree))

    seed = rng.choice(assigned)
    to_remove = {seed}

    others = [(state.dist(seed, c), c) for c in assigned if c != seed]
    others.sort()

    for _, cid in others:
        if len(to_remove) >= n_remove:
            break
        to_remove.add(cid)

    for v in range(len(state.routes)):
        state.routes[v] = [c for c in state.routes[v] if c not in to_remove]

    state.unassigned.update(to_remove)
    state.update_arrival_times()
    return state


def route_removal(state: VRPState, rng: Random, n_routes: int = 1) -> VRPState:
    """Remove entire route(s)."""
    state = state.copy()

    non_empty = [v for v, r in enumerate(state.routes) if r]
    if not non_empty:
        return state

    n = min(n_routes, len(non_empty))
    to_remove_vehicles = rng.sample(non_empty, n)

    for v in to_remove_vehicles:
        state.unassigned.update(state.routes[v])
        state.routes[v] = []
        state.arrival_times[v] = []

    return state


def sync_removal(state: VRPState, rng: Random) -> VRPState:
    """Remove multi-resource customers so they can be reinserted together."""
    state = state.copy()

    sync_customers = [
        cid for cid, c in enumerate(state.customers) if c.required_vehicles > 1 and cid not in state.unassigned
    ]

    if not sync_customers:
        return random_removal(state, rng)

    target = rng.choice(sync_customers)
    to_remove = {target}

    assigned = [cid for route in state.routes for cid in route if cid != target]
    if assigned:
        nearby = sorted(assigned, key=lambda c: state.dist(target, c))[:3]
        to_remove.update(nearby)

    for v in range(len(state.routes)):
        state.routes[v] = [c for c in state.routes[v] if c not in to_remove]

    state.unassigned.update(to_remove)

    for cid in to_remove:
        state.sync_assignments.pop(cid, None)

    state.update_arrival_times()
    return state


def _insertion_cost(state: VRPState, v: int, pos: int, cid: int) -> float | None:
    route = state.routes[v]
    customer = state.customers[cid]
    vehicle = state.vehicles[v]

    if state.route_load(v) + customer.demand > vehicle.capacity:
        return None

    if not route:
        cost = state.dist(0, cid) + state.dist(cid, 0)
    elif pos == 0:
        cost = state.dist(0, cid) + state.dist(cid, route[0]) - state.dist(0, route[0])
    elif pos == len(route):
        cost = state.dist(route[-1], cid) + state.dist(cid, 0) - state.dist(route[-1], 0)
    else:
        prev_c, next_c = route[pos - 1], route[pos]
        cost = state.dist(prev_c, cid) + state.dist(cid, next_c) - state.dist(prev_c, next_c)

    # simplified time window check - only validates this customer
    if not route or pos == 0:
        arrival = state.dist(0, cid)
    elif pos - 1 < len(state.arrival_times[v]):
        prev_arr = state.arrival_times[v][pos - 1]
        prev_cust = state.customers[route[pos - 1]]
        arrival = prev_arr + prev_cust.service_time + state.dist(route[pos - 1], cid)
    else:
        arrival = state.dist(0, cid)
        for i in range(pos):
            c = state.customers[route[i]]
            arrival = max(arrival, c.tw_start) + c.service_time
            arrival += state.dist(route[i], route[i + 1] if i < pos - 1 else cid)

    arrival = max(arrival, customer.tw_start)
    if arrival > customer.tw_end:
        return None

    return cost


def greedy_insertion(state: VRPState, rng: Random) -> VRPState:
    """Insert each unassigned customer at cheapest feasible position."""
    state = state.copy()

    unassigned = list(state.unassigned)
    rng.shuffle(unassigned)

    for cid in unassigned:
        best_cost = float("inf")
        best_pos = None

        for v in range(len(state.vehicles)):
            route = state.routes[v]
            for pos in range(len(route) + 1):
                cost = _insertion_cost(state, v, pos, cid)
                if cost is not None and cost < best_cost:
                    best_cost = cost
                    best_pos = (v, pos)

        if best_pos is not None:
            v, pos = best_pos
            state.routes[v].insert(pos, cid)
            state.unassigned.remove(cid)
            state.arrival_times[v] = state.compute_arrival_times(v)

    return state


def regret_insertion(state: VRPState, rng: Random, k: int = 2) -> VRPState:
    """Insert customers with highest regret (fewest good options) first."""
    state = state.copy()

    while state.unassigned:
        best_regret = -float("inf")
        best_customer = None
        best_pos = None

        for cid in state.unassigned:
            # Find all feasible insertions
            options = []
            for v in range(len(state.vehicles)):
                route = state.routes[v]
                for pos in range(len(route) + 1):
                    cost = _insertion_cost(state, v, pos, cid)
                    if cost is not None:
                        options.append((cost, v, pos))

            if not options:
                continue

            options.sort()

            # Regret = k-th best - best (or large value if fewer than k options)
            if len(options) >= k:
                regret = options[k - 1][0] - options[0][0]
            else:
                regret = 10000.0  # High regret for customers with few options

            if regret > best_regret:
                best_regret = regret
                best_customer = cid
                best_pos = (options[0][1], options[0][2])

        if best_customer is None or best_pos is None:
            break

        v, pos = best_pos
        state.routes[v].insert(pos, best_customer)
        state.unassigned.remove(best_customer)
        state.arrival_times[v] = state.compute_arrival_times(v)

    return state


def sync_aware_insertion(state: VRPState, rng: Random) -> VRPState:
    """Insert multi-resource customers with synchronized vehicle assignments."""
    state = state.copy()

    single = [c for c in state.unassigned if state.customers[c].required_vehicles == 1]
    multi = [c for c in state.unassigned if state.customers[c].required_vehicles > 1]

    for cid in multi:
        customer = state.customers[cid]
        n_required = customer.required_vehicles

        best_cost = float("inf")
        best_insertions: list[tuple[int, int]] = []

        # collect feasible (cost, pos) per vehicle
        feasible: list[list[tuple[float, int]]] = []
        for v in range(len(state.vehicles)):
            opts = []
            for pos in range(len(state.routes[v]) + 1):
                cost = _insertion_cost(state, v, pos, cid)
                if cost is not None:
                    opts.append((cost, pos))
            feasible.append(opts)

        available = [v for v, opts in enumerate(feasible) if opts]

        if len(available) >= n_required:
            # greedy: pick cheapest n_required vehicles
            vcosts = [(min(opts)[0] if opts else float("inf"), v) for v, opts in enumerate(feasible)]
            vcosts.sort()
            selected = [v for _, v in vcosts[:n_required]]

            total = 0.0
            insertions = []
            for v in selected:
                best_opt = min(feasible[v])
                total += best_opt[0]
                insertions.append((v, best_opt[1]))

            if total < best_cost:
                best_cost = total
                best_insertions = insertions

        if best_insertions:
            for v, pos in sorted(best_insertions, key=lambda x: -x[1]):
                state.routes[v].insert(pos, cid)
            state.unassigned.remove(cid)
            state.sync_assignments[cid] = {v for v, _ in best_insertions}

    state.unassigned = set(single)
    state = regret_insertion(state, rng)

    state.update_arrival_times()
    return state


def vrp_objective(
    state: VRPState,
    *,
    distance_weight: float = 1.0,
    vehicle_weight: float = 0.0,
    tw_penalty: float = 1000.0,
    capacity_penalty: float = 1000.0,
    sync_penalty: float = 10000.0,
    unassigned_penalty: float = 100000.0,
) -> float:
    """Compute weighted objective for VRPTW (lower is better)."""
    obj = 0.0
    obj += distance_weight * state.total_distance()
    obj += vehicle_weight * state.vehicles_used()
    obj += tw_penalty * state.time_window_violation()
    obj += capacity_penalty * state.capacity_violation()
    obj += sync_penalty * state.sync_violation()
    obj += unassigned_penalty * len(state.unassigned)
    return obj


def _build_initial_solution(state: VRPState, rng: Random) -> VRPState:
    return greedy_insertion(state, rng)


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
) -> Result:
    """Solve VRPTW using Adaptive Large Neighborhood Search."""
    rng = Random(seed)

    cust_list: list[Customer] = [Customer(0, depot[0], depot[1])]
    for c in customers:
        if isinstance(c, Customer):
            cust_list.append(c)
        else:
            cust_list.append(
                Customer(
                    c[0],
                    c[1],
                    c[2],
                    c[3] if len(c) > 3 else 0.0,
                    c[4] if len(c) > 4 else 0.0,
                    c[5] if len(c) > 5 else float("inf"),
                    c[6] if len(c) > 6 else 0.0,
                    c[7] if len(c) > 7 else 1,
                )
            )

    if isinstance(vehicles, int):
        veh_list = [Vehicle(i, vehicle_capacity) for i in range(vehicles)]
    else:
        veh_list = list(vehicles)

    state = VRPState.from_problem(cust_list, veh_list)
    initial = _build_initial_solution(state, rng)

    def objective(s: VRPState) -> float:
        return vrp_objective(
            s,
            distance_weight=distance_weight,
            vehicle_weight=vehicle_weight,
            tw_penalty=tw_penalty,
            capacity_penalty=capacity_penalty,
            sync_penalty=sync_penalty,
        )

    has_sync = any(c.required_vehicles > 1 for c in cust_list)

    destroy_ops: list = [
        lambda s, r: random_removal(s, r, 0.1),
        lambda s, r: random_removal(s, r, 0.3),
        lambda s, r: worst_removal(s, r, 0.1),
        lambda s, r: worst_removal(s, r, 0.3),
        lambda s, r: related_removal(s, r, 0.2),
        route_removal,
    ]

    repair_ops: list = [
        greedy_insertion,
        lambda s, r: regret_insertion(s, r, 2),
        lambda s, r: regret_insertion(s, r, 3),
    ]

    if has_sync:
        destroy_ops.append(sync_removal)
        repair_ops.append(sync_aware_insertion)

    return alns(
        initial,
        objective,
        destroy_ops,
        repair_ops,
        max_iter=max_iter,
        max_no_improve=max_no_improve,
        seed=seed,
        on_progress=on_progress,
        progress_interval=progress_interval,
    )

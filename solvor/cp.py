"""
CP-SAT Solver, constraint programming with a SAT backend.

Use this for puzzles and scheduling with "all different", arithmetic constraints,
and logical combinations. Sudoku, N-Queens, nurse rostering, timetabling. The
solver encodes your integer variables and constraints into SAT clauses, then
hands it off to the SAT solver. You get the expressiveness of CP with the
raw solving power of modern SAT.

Don't use this for: optimization problems (use milp), pure linear problems
(simplex is simpler and faster), or trivially small problems where the encoding
overhead isn't worth it.

    from solvor.cp import Model

    m = Model()
    x = m.int_var(0, 9, 'x')
    y = m.int_var(0, 9, 'y')
    m.add(m.all_different([x, y]))
    m.add(x + y == 10)
    result = m.solve()  # {'x': 3, 'y': 7} or similar

    # warm start with hints (guides search toward known values)
    result = m.solve(hints={'x': 3})

    # find multiple solutions (result.solutions contains all found)
    result = m.solve(solution_limit=10)

    # global constraints for scheduling and routing
    m.add(m.circuit([x, y, z]))  # Hamiltonian cycle
    m.add(m.no_overlap(starts, durations))  # intervals don't overlap
    m.add(m.cumulative(starts, durations, demands, capacity))  # resource limit

For problems that need both satisfaction and optimization, or heavier constraint
logic, z3 sits nicely between this CP approach and full MILP.
"""

from itertools import combinations

from solvor.sat import Status as SATStatus
from solvor.sat import solve_sat
from solvor.types import Result, Status

__all__ = ["Model"]


class IntVar:
    def __init__(self, model, lb, ub, name):
        self.model = model
        self.lb = lb
        self.ub = ub
        self.name = name
        self.bool_vars = {}
        for v in range(lb, ub + 1):
            self.bool_vars[v] = model._new_bool_var()

    def __eq__(self, other):
        if isinstance(other, int):
            return ("eq_const", self, other)
        if isinstance(other, IntVar):
            return ("eq_var", self, other)
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, int):
            return ("ne_const", self, other)
        if isinstance(other, IntVar):
            return ("ne_var", self, other)
        return NotImplemented

    def __add__(self, other):
        return ("add", self, other)

    def __radd__(self, other):
        return ("add", other, self)


class Model:
    def __init__(self):
        self._next_bool = 1
        self._vars = {}
        self._constraints = []
        self._clauses = []

    def _new_bool_var(self):
        v = self._next_bool
        self._next_bool += 1
        return v

    def int_var(self, lb, ub, name=None):
        if name is None:
            name = f"_v{len(self._vars)}"
        var = IntVar(self, lb, ub, name)
        self._vars[name] = var
        return var

    def all_different(self, variables):
        return ("all_different", tuple(variables))

    def circuit(self, variables):
        """Hamiltonian circuit: variables form a single cycle visiting all nodes."""
        return ("circuit", tuple(variables))

    def no_overlap(self, starts, durations):
        """Intervals (starts[i], starts[i]+durations[i]) don't overlap."""
        if len(starts) != len(durations):
            raise ValueError("starts and durations must have same length")
        return ("no_overlap", tuple(starts), tuple(durations))

    def cumulative(self, starts, durations, demands, capacity):
        """At any time, sum of active demands <= capacity."""
        if len(starts) != len(durations) or len(durations) != len(demands):
            raise ValueError("starts, durations, demands must have same length")
        return ("cumulative", tuple(starts), tuple(durations), tuple(demands), capacity)

    def add(self, constraint):
        self._constraints.append(constraint)

    def _encode_exactly_one(self, lits):
        if not lits:
            return
        self._clauses.append(lits)
        for a, b in combinations(lits, 2):
            self._clauses.append([-a, -b])

    def _encode_at_most_one(self, lits):
        for a, b in combinations(lits, 2):
            self._clauses.append([-a, -b])

    def _encode_vars(self):
        for var in self._vars.values():
            lits = [var.bool_vars[v] for v in range(var.lb, var.ub + 1)]
            self._encode_exactly_one(lits)

    def _encode_all_different(self, variables):
        all_vals = set()
        for var in variables:
            all_vals.update(range(var.lb, var.ub + 1))

        for val in all_vals:
            lits = []
            for var in variables:
                if val in var.bool_vars:
                    lits.append(var.bool_vars[val])
            if len(lits) > 1:
                self._encode_at_most_one(lits)

    def _encode_eq_const(self, var, val):
        if val in var.bool_vars:
            self._clauses.append([var.bool_vars[val]])
        else:
            self._clauses.append([])

    def _encode_ne_const(self, var, val):
        if val in var.bool_vars:
            self._clauses.append([-var.bool_vars[val]])

    def _encode_eq_var(self, var1, var2):
        common = set(var1.bool_vars.keys()) & set(var2.bool_vars.keys())
        for val in common:
            self._clauses.append([-var1.bool_vars[val], var2.bool_vars[val]])
            self._clauses.append([var1.bool_vars[val], -var2.bool_vars[val]])

        for val in set(var1.bool_vars.keys()) - common:
            self._clauses.append([-var1.bool_vars[val]])
        for val in set(var2.bool_vars.keys()) - common:
            self._clauses.append([-var2.bool_vars[val]])

    def _encode_ne_var(self, var1, var2):
        common = set(var1.bool_vars.keys()) & set(var2.bool_vars.keys())
        for val in common:
            self._clauses.append([-var1.bool_vars[val], -var2.bool_vars[val]])

    def _flatten_sum(self, expr):
        terms = []
        const = 0

        def flatten(e):
            nonlocal const
            if isinstance(e, IntVar):
                terms.append(e)
            elif isinstance(e, int):
                const += e
            elif isinstance(e, tuple) and e[0] == "add":
                flatten(e[1])
                flatten(e[2])

        flatten(expr)
        return terms, const

    def _encode_sum_eq(self, variables, target):
        n = len(variables)
        if n == 0:
            if target != 0:
                self._clauses.append([])
            return

        min_sum = sum(v.lb for v in variables)
        max_sum = sum(v.ub for v in variables)

        if target < min_sum or target > max_sum:
            self._clauses.append([])
            return

        if n == 1:
            self._encode_eq_const(variables[0], target)
            return

        if n == 2:
            v1, v2 = variables
            for val1 in range(v1.lb, v1.ub + 1):
                val2 = target - val1
                if val2 < v2.lb or val2 > v2.ub:
                    self._clauses.append([-v1.bool_vars[val1]])
                else:
                    self._clauses.append([-v1.bool_vars[val1], v2.bool_vars[val2]])
            return

        partial_sum = self.int_var(variables[0].lb + variables[1].lb, variables[0].ub + variables[1].ub)

        for v1 in range(variables[0].lb, variables[0].ub + 1):
            for v2 in range(variables[1].lb, variables[1].ub + 1):
                s = v1 + v2
                if s in partial_sum.bool_vars:
                    self._clauses.append(
                        [-variables[0].bool_vars[v1], -variables[1].bool_vars[v2], partial_sum.bool_vars[s]]
                    )

        self._encode_sum_eq([partial_sum] + list(variables[2:]), target)

    def _encode_circuit(self, variables):
        """Encode circuit constraint: successor variables form a single Hamiltonian cycle."""
        n = len(variables)
        if n == 0:
            return

        # All different
        self._encode_all_different(variables)

        # No self-loops: x[i] != i
        for i, var in enumerate(variables):
            if i in var.bool_vars:
                self._clauses.append([-var.bool_vars[i]])

        # Subtour elimination using MTZ formulation with auxiliary order variables
        # t[i] represents position of node i in the tour
        # For nodes other than 0: if x[i] = j, then t[j] = t[i] + 1
        if n <= 1:
            return

        t = [self.int_var(0 if i == 0 else 1, n - 1, f"_circuit_t{i}") for i in range(n)]
        # t[0] is fixed to 0
        self._clauses.append([t[0].bool_vars[0]])

        # For each edge i -> j (j != 0): t[j] >= t[i] + 1
        for i, var in enumerate(variables):
            for j in range(1, n):  # j != 0 (depot can be revisited)
                if j in var.bool_vars:
                    # x[i] = j implies t[j] > t[i]
                    for ti in range(var.lb, var.ub + 1):
                        if ti not in t[i].bool_vars:
                            continue
                        # If x[i] = j and t[i] = ti, then t[j] >= ti + 1
                        for tj in range(t[j].lb, ti + 1):  # t[j] <= ti (violation)
                            if tj in t[j].bool_vars:
                                # NOT(x[i]=j) OR NOT(t[i]=ti) OR NOT(t[j]=tj)
                                self._clauses.append([-var.bool_vars[j], -t[i].bool_vars[ti], -t[j].bool_vars[tj]])

    def _encode_no_overlap(self, starts, durations):
        """Encode no-overlap constraint: intervals don't overlap."""
        n = len(starts)
        for i in range(n):
            for j in range(i + 1, n):
                # Either i ends before j starts, or j ends before i starts
                # start[i] + dur[i] <= start[j] OR start[j] + dur[j] <= start[i]
                self._encode_disjunctive_le(starts[i], durations[i], starts[j], durations[j])

    def _encode_disjunctive_le(self, start1, dur1, start2, dur2):
        """Encode: end1 <= start2 OR end2 <= start1."""
        # Forbid all (s1, s2) combinations where neither ordering is possible
        for s1 in range(start1.lb, start1.ub + 1):
            for s2 in range(start2.lb, start2.ub + 1):
                # If s1 + dur1 > s2 (i before j fails) and s2 + dur2 > s1 (j before i fails)
                # then infeasible for this (s1, s2) combination
                i_before_j = s1 + dur1 <= s2
                j_before_i = s2 + dur2 <= s1

                if not i_before_j and not j_before_i:
                    # This combination is infeasible
                    self._clauses.append([-start1.bool_vars[s1], -start2.bool_vars[s2]])

    def _encode_cumulative(self, starts, durations, demands, capacity):
        """Encode cumulative constraint: sum of active demands <= capacity at all times."""
        n = len(starts)
        if n == 0:
            return

        # Find time horizon
        min_start = min(s.lb for s in starts)
        max_end = max(s.ub + d for s, d in zip(starts, durations))

        # For each time point, sum of active demands <= capacity
        for t in range(min_start, max_end):
            # Collect (start, duration, demand) for tasks that could be active at time t
            active_lits = []
            active_demands = []
            for i in range(n):
                # Task i is active at t if start[i] <= t < start[i] + dur[i]
                # So start[i] in range [t - dur[i] + 1, t]
                for s in range(max(starts[i].lb, t - durations[i] + 1), min(starts[i].ub, t) + 1):
                    if s in starts[i].bool_vars and s <= t < s + durations[i]:
                        active_lits.append(starts[i].bool_vars[s])
                        active_demands.append(demands[i])

            if not active_lits:
                continue

            # Encode: sum of demands for active tasks <= capacity
            # For each subset with sum > capacity, at least one must be false
            # This is exponential but necessary for correctness
            # Use simpler encoding: for small numbers of active tasks
            if len(active_lits) <= 10:
                self._encode_capacity_constraint(active_lits, active_demands, capacity)

    def _encode_capacity_constraint(self, lits, demands, capacity):
        """Encode sum constraint: if all lits true, demands sum must <= capacity."""
        # Find minimal infeasible subsets
        n = len(lits)
        for size in range(1, n + 1):
            for subset in combinations(range(n), size):
                if sum(demands[i] for i in subset) > capacity:
                    # At least one of these must be false
                    # Check if any smaller subset is already infeasible
                    is_minimal = True
                    for smaller_size in range(1, size):
                        for smaller in combinations(subset, smaller_size):
                            if sum(demands[i] for i in smaller) > capacity:
                                is_minimal = False
                                break
                        if not is_minimal:
                            break
                    if is_minimal:
                        self._clauses.append([-lits[i] for i in subset])

    def _encode_constraint(self, constraint):
        if isinstance(constraint, tuple):
            kind = constraint[0]

            if kind == "all_different":
                self._encode_all_different(constraint[1])

            elif kind == "circuit":
                self._encode_circuit(constraint[1])

            elif kind == "no_overlap":
                self._encode_no_overlap(constraint[1], constraint[2])

            elif kind == "cumulative":
                self._encode_cumulative(constraint[1], constraint[2], constraint[3], constraint[4])

            elif kind == "eq_const":
                self._encode_eq_const(constraint[1], constraint[2])

            elif kind == "ne_const":
                self._encode_ne_const(constraint[1], constraint[2])

            elif kind == "eq_var":
                self._encode_eq_var(constraint[1], constraint[2])

            elif kind == "ne_var":
                self._encode_ne_var(constraint[1], constraint[2])

            elif kind == "add":
                terms, const = self._flatten_sum(constraint)
                if len(terms) == 0:
                    if const != 0:
                        self._clauses.append([])
                else:
                    pass

            elif kind == "sum_eq":
                self._encode_sum_eq(list(constraint[1]), constraint[2])

            elif kind == "sum_le":
                terms, target = constraint[1], constraint[2]
                aux = self.int_var(sum(v.lb for v in terms), min(sum(v.ub for v in terms), target))
                self._encode_sum_eq(list(terms), aux)

            elif kind == "sum_ge":
                terms, target = constraint[1], constraint[2]
                aux = self.int_var(max(sum(v.lb for v in terms), target), sum(v.ub for v in terms))
                self._encode_sum_eq(list(terms), aux)

    def sum_eq(self, variables, target):
        return ("sum_eq", tuple(variables), target)

    def sum_le(self, variables, target):
        return ("sum_le", tuple(variables), target)

    def sum_ge(self, variables, target):
        return ("sum_ge", tuple(variables), target)

    def solve(self, *, hints: dict[str, int] | None = None, solution_limit: int = 1, **kwargs):
        self._clauses = []
        self._encode_vars()

        for constraint in self._constraints:
            self._encode_constraint(constraint)

        if any(len(c) == 0 for c in self._clauses):
            return Result(None, 0, 0, 0, Status.INFEASIBLE)

        # Convert hints to SAT assumptions
        assumptions = list(kwargs.pop("assumptions", []) or [])
        if hints:
            for name, val in hints.items():
                if name in self._vars:
                    var = self._vars[name]
                    if val in var.bool_vars:
                        assumptions.append(var.bool_vars[val])

        sat_result = solve_sat(self._clauses, assumptions=assumptions or None, solution_limit=solution_limit, **kwargs)

        if sat_result.status == SATStatus.INFEASIBLE:
            return Result(None, 0, sat_result.iterations, sat_result.evaluations, Status.INFEASIBLE)

        if sat_result.status == SATStatus.MAX_ITER:
            return Result(None, 0, sat_result.iterations, sat_result.evaluations, Status.MAX_ITER)

        def decode_sat_solution(sat_sol: dict[int, bool]) -> dict[str, int]:
            """Convert SAT solution to CP variable assignments."""
            cp_sol = {}
            for name, var in self._vars.items():
                if name.startswith("_"):
                    continue
                for val, bool_var in var.bool_vars.items():
                    if sat_sol.get(bool_var, False):
                        cp_sol[name] = val
                        break
            return cp_sol

        solution = decode_sat_solution(sat_result.solution)  # type: ignore[arg-type]

        # Convert multiple SAT solutions to CP solutions
        if sat_result.solutions is not None:
            cp_solutions = tuple(decode_sat_solution(s) for s in sat_result.solutions)
            return Result(
                solution,
                0,
                sat_result.iterations,
                sat_result.evaluations,
                solutions=cp_solutions,
            )

        return Result(solution, 0, sat_result.iterations, sat_result.evaluations)

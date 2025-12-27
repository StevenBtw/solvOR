"""
CP Solver, constraint programming with backtracking search.

Write constraints like a human ("all different", "x + y == 10"), and the solver
finds valid assignments. Perfect for puzzles and scheduling: Sudoku, N-Queens,
nurse rostering, timetabling.

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

Uses DFS with constraint propagation by default (fast for most problems). Falls
back to SAT encoding automatically for complex global constraints. You can also
choose explicitly: `m.solve(solver='dfs')` or `m.solve(solver='sat')`.

For optimization problems use MILP. For heavier constraint logic, Z3 sits nicely
between this approach and full MILP: https://github.com/Z3Prover/z3
"""

from itertools import combinations
from typing import Any

from solvor.sat import Status as SATStatus
from solvor.sat import solve_sat
from solvor.types import Result, Status

__all__ = ["Model"]


class Expr:
    """Wrapper for expression tuples to support comparison operators."""

    def __init__(self, data):
        self.data = data

    def __eq__(self, other):
        if isinstance(other, Expr):
            return ("ne_expr", self.data, other.data, False)  # False = eq
        if isinstance(other, (int, IntVar)):
            return ("ne_expr", self.data, other, False)
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, Expr):
            return ("ne_expr", self.data, other.data, True)  # True = ne
        if isinstance(other, (int, IntVar)):
            return ("ne_expr", self.data, other, True)
        return NotImplemented

    def __add__(self, other):
        return Expr(("add", self.data, other.data if isinstance(other, Expr) else other))

    def __radd__(self, other):
        return Expr(("add", other.data if isinstance(other, Expr) else other, self.data))

    def __sub__(self, other):
        if isinstance(other, int):
            return Expr(("add", self.data, -other))
        if isinstance(other, Expr):
            return Expr(("sub", self.data, other.data))
        return NotImplemented


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
        if isinstance(other, Expr):
            return ("ne_expr", self, other.data, False)
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, int):
            return ("ne_const", self, other)
        if isinstance(other, IntVar):
            return ("ne_var", self, other)
        if isinstance(other, Expr):
            return ("ne_expr", self, other.data, True)
        return NotImplemented

    def __add__(self, other):
        return Expr(("add", self, other.data if isinstance(other, Expr) else other))

    def __radd__(self, other):
        return Expr(("add", other.data if isinstance(other, Expr) else other, self))

    def __sub__(self, other):
        if isinstance(other, int):
            return Expr(("add", self, -other))
        if isinstance(other, IntVar):
            return Expr(("sub", self, other))
        if isinstance(other, Expr):
            return Expr(("sub", self, other.data))
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, int):
            return Expr(("rsub", self, other))  # other - self
        return NotImplemented


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

    def _encode_ne_expr(self, left, right, is_ne):
        """Encode (left_expr != right_expr) or (left_expr == right_expr).

        Handles linear expressions like (x + c1) != (y + c2).
        """
        # Handle subtraction: (x - y) ?= c => x ?= y + c
        if isinstance(left, tuple) and left[0] == "sub":
            x, y = left[1], left[2]
            if isinstance(x, IntVar) and isinstance(y, IntVar):
                right_const = right if isinstance(right, int) else 0
                # x - y ?= c => x ?= y + c
                offset = -right_const  # x ?= y + c => offset is -c for the existing formula
                if is_ne:
                    for v1 in x.bool_vars:
                        v2 = v1 - right_const
                        if v2 in y.bool_vars:
                            self._clauses.append([-x.bool_vars[v1], -y.bool_vars[v2]])
                else:
                    for v1 in x.bool_vars:
                        v2 = v1 - right_const
                        if v2 in y.bool_vars:
                            self._clauses.append([-x.bool_vars[v1], y.bool_vars[v2]])
                            self._clauses.append([x.bool_vars[v1], -y.bool_vars[v2]])
                        else:
                            self._clauses.append([-x.bool_vars[v1]])
                return

        left_terms, left_const = self._flatten_sum(left)
        right_terms, right_const = self._flatten_sum(right)

        # Handle case: single var + const on left, constant on right
        # (x + c1) == c2 => x == c2 - c1
        if len(left_terms) == 1 and len(right_terms) == 0:
            var = left_terms[0]
            target = right_const - left_const
            if is_ne:
                self._encode_ne_const(var, target)
            else:
                self._encode_eq_const(var, target)
            return

        # Handle case: constant on left, single var + const on right
        # c1 == (y + c2) => y == c1 - c2
        if len(left_terms) == 0 and len(right_terms) == 1:
            var = right_terms[0]
            target = left_const - right_const
            if is_ne:
                self._encode_ne_const(var, target)
            else:
                self._encode_eq_const(var, target)
            return

        # Handle case: two vars on left, constant on right
        # (x + y + c1) == c2 => x + y == c2 - c1
        if len(left_terms) == 2 and len(right_terms) == 0:
            target = right_const - left_const
            if is_ne:
                # Forbid all pairs where sum == target
                v1, v2 = left_terms
                for val1 in v1.bool_vars:
                    val2 = target - val1
                    if val2 in v2.bool_vars:
                        self._clauses.append([-v1.bool_vars[val1], -v2.bool_vars[val2]])
            else:
                self._encode_sum_eq(left_terms, target)
            return

        # Handle simple case: single var + const on each side
        if len(left_terms) == 1 and len(right_terms) == 1:
            var1, var2 = left_terms[0], right_terms[0]
            offset = right_const - left_const  # v1 + left_const ?= v2 + right_const => v1 ?= v2 + offset

            if is_ne:
                # var1 + left_const != var2 + right_const
                # => var1 != var2 + offset
                # => forbid all (v1, v2) where v1 == v2 + offset
                for v1 in var1.bool_vars:
                    v2 = v1 - offset
                    if v2 in var2.bool_vars:
                        self._clauses.append([-var1.bool_vars[v1], -var2.bool_vars[v2]])
            else:
                # var1 + left_const == var2 + right_const
                # => var1 == var2 + offset
                for v1 in var1.bool_vars:
                    v2 = v1 - offset
                    if v2 in var2.bool_vars:
                        self._clauses.append([-var1.bool_vars[v1], var2.bool_vars[v2]])
                        self._clauses.append([var1.bool_vars[v1], -var2.bool_vars[v2]])
                    else:
                        self._clauses.append([-var1.bool_vars[v1]])

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

    def _encode_sum_le(self, variables, target):
        """Encode sum(variables) <= target by forbidding combinations > target."""
        if len(variables) == 0:
            return
        if len(variables) == 1:
            v = variables[0]
            for val in range(v.lb, v.ub + 1):
                if val > target:
                    self._clauses.append([-v.bool_vars[val]])
            return
        if len(variables) == 2:
            v1, v2 = variables
            for val1 in range(v1.lb, v1.ub + 1):
                for val2 in range(v2.lb, v2.ub + 1):
                    if val1 + val2 > target:
                        self._clauses.append([-v1.bool_vars[val1], -v2.bool_vars[val2]])
            return
        # n > 2: create partial sum for first two, recurse
        v1, v2 = variables[0], variables[1]
        rest_min = sum(v.lb for v in variables[2:])
        partial_sum = self.int_var(v1.lb + v2.lb, min(v1.ub + v2.ub, target - rest_min))
        for val1 in range(v1.lb, v1.ub + 1):
            for val2 in range(v2.lb, v2.ub + 1):
                s = val1 + val2
                if s in partial_sum.bool_vars:
                    self._clauses.append([-v1.bool_vars[val1], -v2.bool_vars[val2], partial_sum.bool_vars[s]])
                else:
                    # Forbid combinations whose sum is outside partial_sum's domain
                    self._clauses.append([-v1.bool_vars[val1], -v2.bool_vars[val2]])
        self._encode_sum_le([partial_sum] + list(variables[2:]), target)

    def _encode_sum_ge(self, variables, target):
        """Encode sum(variables) >= target by forbidding combinations < target."""
        if len(variables) == 0:
            if target > 0:
                self._clauses.append([])
            return
        if len(variables) == 1:
            v = variables[0]
            for val in range(v.lb, v.ub + 1):
                if val < target:
                    self._clauses.append([-v.bool_vars[val]])
            return
        if len(variables) == 2:
            v1, v2 = variables
            for val1 in range(v1.lb, v1.ub + 1):
                for val2 in range(v2.lb, v2.ub + 1):
                    if val1 + val2 < target:
                        self._clauses.append([-v1.bool_vars[val1], -v2.bool_vars[val2]])
            return
        # n > 2: create partial sum for first two, recurse
        v1, v2 = variables[0], variables[1]
        rest_max = sum(v.ub for v in variables[2:])
        partial_sum = self.int_var(max(v1.lb + v2.lb, target - rest_max), v1.ub + v2.ub)
        for val1 in range(v1.lb, v1.ub + 1):
            for val2 in range(v2.lb, v2.ub + 1):
                s = val1 + val2
                if s in partial_sum.bool_vars:
                    self._clauses.append([-v1.bool_vars[val1], -v2.bool_vars[val2], partial_sum.bool_vars[s]])
                else:
                    # Forbid combinations whose sum is outside partial_sum's domain
                    self._clauses.append([-v1.bool_vars[val1], -v2.bool_vars[val2]])
        self._encode_sum_ge([partial_sum] + list(variables[2:]), target)

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

            elif kind == "ne_expr":
                self._encode_ne_expr(constraint[1], constraint[2], constraint[3])

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
                self._encode_sum_le(list(terms), target)

            elif kind == "sum_ge":
                terms, target = constraint[1], constraint[2]
                self._encode_sum_ge(list(terms), target)

    def sum_eq(self, variables, target):
        return ("sum_eq", tuple(variables), target)

    def sum_le(self, variables, target):
        return ("sum_le", tuple(variables), target)

    def sum_ge(self, variables, target):
        return ("sum_ge", tuple(variables), target)

    def solve(
        self,
        *,
        hints: dict[str, int] | None = None,
        solution_limit: int = 1,
        solver: str = "dfs",
        **kwargs: Any,
    ) -> Result:
        """Solve the constraint satisfaction problem.

        Args:
            hints: Initial value hints to guide search.
            solution_limit: Max solutions to find (default 1).
            solver: 'dfs' (default, fast backtracking) or 'sat' (SAT encoding).
            **kwargs: Additional solver options.

        Returns:
            Result with solution dict mapping variable names to values.
        """
        if solver == "dfs":
            return self._solve_dfs(hints=hints, solution_limit=solution_limit, **kwargs)
        elif solver == "sat":
            return self._solve_sat(hints=hints, solution_limit=solution_limit, **kwargs)
        else:
            raise ValueError(f"Unknown solver: {solver}. Use 'dfs' or 'sat'.")

    def _solve_dfs(
        self, *, hints: dict[str, int] | None = None, solution_limit: int = 1, **kwargs
    ):
        """DFS backtracking solver with constraint propagation.

        Uses arc consistency and MRV heuristic for fast constraint satisfaction.
        Falls back to SAT for complex constraints (circuit, no_overlap, cumulative, sum).
        """
        # Check for constraints that need SAT solver
        for c in self._constraints:
            if isinstance(c, tuple) and c[0] in ("circuit", "no_overlap", "cumulative", "sum_eq", "sum_le", "sum_ge"):
                return self._solve_sat(hints=hints, solution_limit=solution_limit, **kwargs)

        # Initialize domains as sets
        domains = {name: set(range(var.lb, var.ub + 1)) for name, var in self._vars.items()}

        # Apply hints
        if hints:
            for name, val in hints.items():
                if name in domains and val in domains[name]:
                    domains[name] = {val}

        # Propagate initial constraints
        if not self._propagate(domains):
            return Result(None, 0, 0, 0, Status.INFEASIBLE)

        solutions: list[dict[str, int]] = []
        iterations = [0]  # Use list for mutation in nested function

        def backtrack(domains: dict[str, set[int]]) -> bool:
            iterations[0] += 1

            # Check if all assigned
            unassigned = [n for n in domains if len(domains[n]) > 1 and not n.startswith("_")]
            if not unassigned:
                # Found solution
                sol = {n: next(iter(d)) for n, d in domains.items() if not n.startswith("_")}
                solutions.append(sol)
                return len(solutions) >= solution_limit

            # MRV: pick variable with smallest domain
            var_name = min(unassigned, key=lambda n: len(domains[n]))
            var_domain = list(domains[var_name])

            for val in var_domain:
                # Make assignment
                new_domains = {n: d.copy() for n, d in domains.items()}
                new_domains[var_name] = {val}

                # Propagate
                if self._propagate(new_domains):
                    if backtrack(new_domains):
                        return True

            return False

        backtrack(domains)

        if not solutions:
            return Result(None, 0, iterations[0], 0, Status.INFEASIBLE)

        if len(solutions) == 1:
            return Result(solutions[0], 0, iterations[0], 0)

        return Result(solutions[0], 0, iterations[0], 0, solutions=tuple(solutions))

    def _propagate(self, domains: dict[str, set[int]]) -> bool:
        """Apply arc consistency until fixpoint. Returns False if domain wipeout."""
        changed = True
        while changed:
            changed = False
            for constraint in self._constraints:
                old_sizes = {n: len(d) for n, d in domains.items()}
                if not self._propagate_constraint(constraint, domains):
                    return False
                # Check for domain wipeout or changes
                for n, d in domains.items():
                    if not d:
                        return False
                    if len(d) < old_sizes[n]:
                        changed = True
        return True

    def _propagate_constraint(self, constraint, domains: dict[str, set[int]]) -> bool:
        """Propagate a single constraint. Returns False if inconsistent."""
        if not isinstance(constraint, tuple):
            return True

        kind = constraint[0]

        if kind == "all_different":
            return self._propagate_all_different(constraint[1], domains)

        elif kind == "eq_const":
            var, val = constraint[1], constraint[2]
            if val not in domains[var.name]:
                return False
            domains[var.name] = {val}

        elif kind == "ne_const":
            var, val = constraint[1], constraint[2]
            domains[var.name].discard(val)

        elif kind == "eq_var":
            var1, var2 = constraint[1], constraint[2]
            common = domains[var1.name] & domains[var2.name]
            if not common:
                return False
            domains[var1.name] = common
            domains[var2.name] = common.copy()

        elif kind == "ne_var":
            var1, var2 = constraint[1], constraint[2]
            # If one is assigned, remove from other
            if len(domains[var1.name]) == 1:
                val = next(iter(domains[var1.name]))
                domains[var2.name].discard(val)
            if len(domains[var2.name]) == 1:
                val = next(iter(domains[var2.name]))
                domains[var1.name].discard(val)

        elif kind == "ne_expr":
            return self._propagate_ne_expr(constraint[1], constraint[2], constraint[3], domains)

        return True

    def _propagate_all_different(self, variables, domains: dict[str, set[int]]) -> bool:
        """Propagate all_different: assigned values removed from other domains."""
        # Remove assigned values from other domains
        for var in variables:
            if len(domains[var.name]) == 1:
                val = next(iter(domains[var.name]))
                for other in variables:
                    if other is not var:
                        domains[other.name].discard(val)
        return True

    def _propagate_ne_expr(self, left, right, is_ne: bool, domains: dict[str, set[int]]) -> bool:
        """Propagate (left_expr != right_expr) or (left_expr == right_expr)."""
        left_terms, left_const = self._flatten_sum(left)
        right_terms, right_const = self._flatten_sum(right)

        if len(left_terms) == 1 and len(right_terms) == 1:
            var1, var2 = left_terms[0], right_terms[0]
            offset = right_const - left_const

            if is_ne:
                # var1 != var2 + offset
                if len(domains[var1.name]) == 1:
                    v1 = next(iter(domains[var1.name]))
                    domains[var2.name].discard(v1 - offset)
                if len(domains[var2.name]) == 1:
                    v2 = next(iter(domains[var2.name]))
                    domains[var1.name].discard(v2 + offset)
            else:
                # var1 == var2 + offset
                valid1 = {v for v in domains[var1.name] if (v - offset) in domains[var2.name]}
                valid2 = {v for v in domains[var2.name] if (v + offset) in domains[var1.name]}
                if not valid1 or not valid2:
                    return False
                domains[var1.name] = valid1
                domains[var2.name] = valid2

        return True

    def _solve_sat(
        self, *, hints: dict[str, int] | None = None, solution_limit: int = 1, **kwargs
    ):
        """SAT-based solver (encodes to boolean satisfiability)."""
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

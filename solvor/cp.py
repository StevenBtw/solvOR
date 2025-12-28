r"""
CP Solver, constraint programming with backtracking search.

Write constraints like a human ("all different", "x + y == 10"), and the solver
finds valid assignments. Perfect for puzzles and scheduling.

    from solvor.cp import Model

    m = Model()
    x = m.int_var(0, 9, 'x')
    y = m.int_var(0, 9, 'y')
    m.add(m.all_different([x, y]))
    m.add(x + y == 10)
    result = m.solve()  # {'x': 3, 'y': 7} or similar

    # global constraints for scheduling and routing
    m.add(m.circuit([x, y, z]))  # Hamiltonian cycle
    m.add(m.no_overlap(starts, durations))  # intervals don't overlap
    m.add(m.cumulative(starts, durations, demands, capacity))  # resource limit

How it works: DFS with constraint propagation and arc consistency. Falls back
to SAT encoding automatically for complex global constraints. MRV (minimum
remaining values) heuristic picks the most constrained variable first.

Use this for:

- Sudoku, N-Queens, logic puzzles
- Nurse rostering and timetabling
- Scheduling with complex constraints
- When "all different" or other global constraints fit naturally

Parameters:

    Model.int_var(lb, ub, name): create integer variable
    Model.add(constraint): add constraint
    Model.solve(hints, solution_limit, solver): find solutions

    hints: initial value hints to guide search
    solution_limit: find multiple solutions
    solver: 'auto' (default), 'dfs', or 'sat'

For optimization use MILP. For heavier constraint logic, see Z3.
"""

from typing import Any

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

    def __mul__(self, other):
        if isinstance(other, int):
            return Expr(("mul", self.data, other))
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, int):
            return Expr(("mul", self.data, other))
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

    def __mul__(self, other):
        if isinstance(other, int):
            return Expr(("mul", self, other))
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, int):
            return Expr(("mul", self, other))
        return NotImplemented


class Model:
    def __init__(self):
        self._next_bool = 1
        self._vars = {}
        self._constraints = []

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

    def sum_eq(self, variables, target):
        return ("sum_eq", tuple(variables), target)

    def sum_le(self, variables, target):
        return ("sum_le", tuple(variables), target)

    def sum_ge(self, variables, target):
        return ("sum_ge", tuple(variables), target)

    def _choose_solver(self) -> str:
        """Pick best solver based on constraints.

        Returns 'dfs' for simple constraints (all_different, eq, ne).
        Returns 'sat' for global constraints (circuit, no_overlap, cumulative, sum_*).
        """
        SAT_REQUIRED = {"circuit", "no_overlap", "cumulative", "sum_eq", "sum_le", "sum_ge"}
        for c in self._constraints:
            if isinstance(c, tuple) and c[0] in SAT_REQUIRED:
                return "sat"
        return "dfs"

    def solve(
        self,
        *,
        hints: dict[str, int] | None = None,
        solution_limit: int = 1,
        solver: str = "auto",
        **kwargs: Any,
    ) -> Result:
        """Solve the constraint satisfaction problem.

        Args:
            hints: Initial value hints to guide search.
            solution_limit: Max solutions to find (default 1).
            solver: 'auto' (default), 'dfs' (fast backtracking), or 'sat' (SAT encoding).
                    'auto' picks DFS for simple constraints, SAT for global constraints.
            **kwargs: Additional solver options.

        Returns:
            Result with solution dict mapping variable names to values.
        """
        if solver == "auto":
            solver = self._choose_solver()

        if solver == "dfs":
            return self._solve_dfs(hints=hints, solution_limit=solution_limit, **kwargs)
        elif solver == "sat":
            from solvor.cp_encoder import SATEncoder

            encoder = SATEncoder(self)
            return encoder.solve(hints=hints, solution_limit=solution_limit, **kwargs)
        else:
            raise ValueError(f"Unknown solver: {solver}. Use 'auto', 'dfs', or 'sat'.")

    def _solve_dfs(
        self, *, hints: dict[str, int] | None = None, solution_limit: int = 1, **kwargs
    ):
        """DFS backtracking solver with constraint propagation.

        Uses arc consistency and MRV heuristic for fast constraint satisfaction.
        Falls back to SAT for global constraints (circuit, no_overlap, cumulative, sum_*).
        """
        # Fall back to SAT for constraints that DFS can't handle
        sat_required = {"circuit", "no_overlap", "cumulative", "sum_eq", "sum_le", "sum_ge"}
        for c in self._constraints:
            if isinstance(c, tuple) and c[0] in sat_required:
                from solvor.cp_encoder import SATEncoder

                encoder = SATEncoder(self)
                return encoder.solve(hints=hints, solution_limit=solution_limit, **kwargs)

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

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

    def _encode_constraint(self, constraint):
        if isinstance(constraint, tuple):
            kind = constraint[0]

            if kind == "all_different":
                self._encode_all_different(constraint[1])

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

    def solve(self, **kwargs):
        self._clauses = []
        self._encode_vars()

        for constraint in self._constraints:
            self._encode_constraint(constraint)

        if any(len(c) == 0 for c in self._clauses):
            return Result(None, 0, 0, 0, Status.INFEASIBLE)

        sat_result = solve_sat(self._clauses, **kwargs)

        if sat_result.status == SATStatus.INFEASIBLE:
            return Result(None, 0, sat_result.iterations, sat_result.evaluations, Status.INFEASIBLE)

        if sat_result.status == SATStatus.MAX_ITER:
            return Result(None, 0, sat_result.iterations, sat_result.evaluations, Status.MAX_ITER)

        solution = {}
        sat_solution: dict[int, bool] = sat_result.solution  # type: ignore[assignment]
        for name, var in self._vars.items():
            if name.startswith("_"):
                continue
            for val, bool_var in var.bool_vars.items():
                if sat_solution.get(bool_var, False):
                    solution[name] = val
                    break

        return Result(solution, 0, sat_result.iterations, sat_result.evaluations)

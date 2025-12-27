"""
SAT encoding for CP constraints.

Internal module used by cp.py when SAT-based solving is needed. Handles encoding
of integer variables and constraints to boolean satisfiability clauses.

Not part of the public API - use Model from cp.py instead.
"""

from itertools import combinations
from typing import TYPE_CHECKING, Any

from solvor.sat import Status as SATStatus
from solvor.sat import solve_sat
from solvor.types import Result, Status

if TYPE_CHECKING:
    from solvor.cp import IntVar, Model

__all__ = ["SATEncoder"]


class SATEncoder:
    """Encodes CP model to SAT and solves it."""

    def __init__(self, model: "Model"):
        self.model = model
        self._clauses: list[list[int]] = []
        self._next_bool = model._next_bool

    def _new_bool_var(self) -> int:
        v = self._next_bool
        self._next_bool += 1
        return v

    # -------------------------------------------------------------------------
    # Basic encoding helpers
    # -------------------------------------------------------------------------

    def _encode_exactly_one(self, lits: list[int]) -> None:
        """Encode exactly-one constraint: exactly one literal must be true."""
        if not lits:
            return
        self._clauses.append(lits)
        for a, b in combinations(lits, 2):
            self._clauses.append([-a, -b])

    def _encode_at_most_one(self, lits: list[int]) -> None:
        """Encode at-most-one constraint: at most one literal can be true."""
        for a, b in combinations(lits, 2):
            self._clauses.append([-a, -b])

    # -------------------------------------------------------------------------
    # Variable encoding
    # -------------------------------------------------------------------------

    def _encode_vars(self) -> None:
        """Encode all integer variables as exactly-one boolean constraints."""
        for var in self.model._vars.values():
            lits = [var.bool_vars[v] for v in range(var.lb, var.ub + 1)]
            self._encode_exactly_one(lits)

    # -------------------------------------------------------------------------
    # Simple constraint encoding
    # -------------------------------------------------------------------------

    def _encode_all_different(self, variables: tuple["IntVar", ...]) -> None:
        """Encode all-different constraint: no two variables share a value."""
        all_vals: set[int] = set()
        for var in variables:
            all_vals.update(range(var.lb, var.ub + 1))

        for val in all_vals:
            lits = []
            for var in variables:
                if val in var.bool_vars:
                    lits.append(var.bool_vars[val])
            if len(lits) > 1:
                self._encode_at_most_one(lits)

    def _encode_eq_const(self, var: "IntVar", val: int) -> None:
        """Encode var == constant."""
        if val in var.bool_vars:
            self._clauses.append([var.bool_vars[val]])
        else:
            self._clauses.append([])  # Unsatisfiable

    def _encode_ne_const(self, var: "IntVar", val: int) -> None:
        """Encode var != constant."""
        if val in var.bool_vars:
            self._clauses.append([-var.bool_vars[val]])

    def _encode_eq_var(self, var1: "IntVar", var2: "IntVar") -> None:
        """Encode var1 == var2."""
        common = set(var1.bool_vars.keys()) & set(var2.bool_vars.keys())
        for val in common:
            self._clauses.append([-var1.bool_vars[val], var2.bool_vars[val]])
            self._clauses.append([var1.bool_vars[val], -var2.bool_vars[val]])

        for val in set(var1.bool_vars.keys()) - common:
            self._clauses.append([-var1.bool_vars[val]])
        for val in set(var2.bool_vars.keys()) - common:
            self._clauses.append([-var2.bool_vars[val]])

    def _encode_ne_var(self, var1: "IntVar", var2: "IntVar") -> None:
        """Encode var1 != var2."""
        common = set(var1.bool_vars.keys()) & set(var2.bool_vars.keys())
        for val in common:
            self._clauses.append([-var1.bool_vars[val], -var2.bool_vars[val]])

    # -------------------------------------------------------------------------
    # Expression handling
    # -------------------------------------------------------------------------

    def _flatten_sum(self, expr: Any) -> tuple[list["IntVar"], int]:
        """Flatten a sum expression into (list of variables, constant offset)."""
        from solvor.cp import IntVar

        terms: list[IntVar] = []
        const = 0

        def flatten(e: Any) -> None:
            nonlocal const
            if isinstance(e, IntVar):
                terms.append(e)
            elif isinstance(e, int):
                const += e
            elif isinstance(e, tuple) and e[0] == "add":
                flatten(e[1])
                flatten(e[2])
            elif isinstance(e, tuple) and e[0] == "mul":
                # Handle multiplication: (mul, var_or_expr, int_coef)
                _, operand, coef = e
                if isinstance(operand, IntVar) and isinstance(coef, int):
                    # For now, we can only handle coef == 1 directly
                    # More complex cases need auxiliary variables
                    for _ in range(coef):
                        terms.append(operand)
                elif isinstance(coef, IntVar) and isinstance(operand, int):
                    for _ in range(operand):
                        terms.append(coef)

        flatten(expr)
        return terms, const

    def _encode_ne_expr(
        self, left: Any, right: Any, is_ne: bool
    ) -> None:
        """Encode (left_expr != right_expr) or (left_expr == right_expr).

        Handles linear expressions like (x + c1) != (y + c2).
        """
        from solvor.cp import IntVar

        # Handle subtraction: (x - y) ?= c => x ?= y + c
        if isinstance(left, tuple) and left[0] == "sub":
            x, y = left[1], left[2]
            if isinstance(x, IntVar) and isinstance(y, IntVar):
                right_const = right if isinstance(right, int) else 0
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
        if len(left_terms) == 1 and len(right_terms) == 0:
            var = left_terms[0]
            target = right_const - left_const
            if is_ne:
                self._encode_ne_const(var, target)
            else:
                self._encode_eq_const(var, target)
            return

        # Handle case: constant on left, single var + const on right
        if len(left_terms) == 0 and len(right_terms) == 1:
            var = right_terms[0]
            target = left_const - right_const
            if is_ne:
                self._encode_ne_const(var, target)
            else:
                self._encode_eq_const(var, target)
            return

        # Handle case: two vars on left, constant on right
        if len(left_terms) == 2 and len(right_terms) == 0:
            target = right_const - left_const
            if is_ne:
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
            offset = right_const - left_const

            if is_ne:
                for v1 in var1.bool_vars:
                    v2 = v1 - offset
                    if v2 in var2.bool_vars:
                        self._clauses.append([-var1.bool_vars[v1], -var2.bool_vars[v2]])
            else:
                for v1 in var1.bool_vars:
                    v2 = v1 - offset
                    if v2 in var2.bool_vars:
                        self._clauses.append([-var1.bool_vars[v1], var2.bool_vars[v2]])
                        self._clauses.append([var1.bool_vars[v1], -var2.bool_vars[v2]])
                    else:
                        self._clauses.append([-var1.bool_vars[v1]])

    # -------------------------------------------------------------------------
    # Sum constraints
    # -------------------------------------------------------------------------

    def _encode_sum_eq(self, variables: list["IntVar"], target: int) -> None:
        """Encode sum(variables) == target."""
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

        # n > 2: create partial sum variable and recurse
        partial_sum = self._create_int_var(
            variables[0].lb + variables[1].lb,
            variables[0].ub + variables[1].ub,
        )

        for v1 in range(variables[0].lb, variables[0].ub + 1):
            for v2 in range(variables[1].lb, variables[1].ub + 1):
                s = v1 + v2
                if s in partial_sum.bool_vars:
                    self._clauses.append(
                        [-variables[0].bool_vars[v1], -variables[1].bool_vars[v2], partial_sum.bool_vars[s]]
                    )

        self._encode_sum_eq([partial_sum] + list(variables[2:]), target)

    def _encode_sum_le(self, variables: list["IntVar"], target: int) -> None:
        """Encode sum(variables) <= target."""
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
        partial_sum = self._create_int_var(v1.lb + v2.lb, min(v1.ub + v2.ub, target - rest_min))

        for val1 in range(v1.lb, v1.ub + 1):
            for val2 in range(v2.lb, v2.ub + 1):
                s = val1 + val2
                if s in partial_sum.bool_vars:
                    self._clauses.append([-v1.bool_vars[val1], -v2.bool_vars[val2], partial_sum.bool_vars[s]])
                else:
                    self._clauses.append([-v1.bool_vars[val1], -v2.bool_vars[val2]])

        self._encode_sum_le([partial_sum] + list(variables[2:]), target)

    def _encode_sum_ge(self, variables: list["IntVar"], target: int) -> None:
        """Encode sum(variables) >= target."""
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
        partial_sum = self._create_int_var(max(v1.lb + v2.lb, target - rest_max), v1.ub + v2.ub)

        for val1 in range(v1.lb, v1.ub + 1):
            for val2 in range(v2.lb, v2.ub + 1):
                s = val1 + val2
                if s in partial_sum.bool_vars:
                    self._clauses.append([-v1.bool_vars[val1], -v2.bool_vars[val2], partial_sum.bool_vars[s]])
                else:
                    self._clauses.append([-v1.bool_vars[val1], -v2.bool_vars[val2]])

        self._encode_sum_ge([partial_sum] + list(variables[2:]), target)

    def _create_int_var(self, lb: int, ub: int) -> "IntVar":
        """Create auxiliary integer variable for encoding."""
        from solvor.cp import IntVar

        name = f"_aux{self._next_bool}"
        var = IntVar(self.model, lb, ub, name)
        # Manually create bool vars using our counter
        var.bool_vars = {}
        for v in range(lb, ub + 1):
            var.bool_vars[v] = self._new_bool_var()
        self.model._vars[name] = var
        return var

    # -------------------------------------------------------------------------
    # Global constraints
    # -------------------------------------------------------------------------

    def _encode_circuit(self, variables: tuple["IntVar", ...]) -> None:
        """Encode circuit constraint: successor variables form a Hamiltonian cycle."""
        n = len(variables)
        if n == 0:
            return

        # All different
        self._encode_all_different(variables)

        # No self-loops: x[i] != i
        for i, var in enumerate(variables):
            if i in var.bool_vars:
                self._clauses.append([-var.bool_vars[i]])

        if n <= 1:
            return

        # Subtour elimination using MTZ formulation
        t = [self._create_int_var(0 if i == 0 else 1, n - 1) for i in range(n)]
        # t[0] is fixed to 0
        self._clauses.append([t[0].bool_vars[0]])

        # For each edge i -> j (j != 0): t[j] >= t[i] + 1
        for i, var in enumerate(variables):
            for j in range(1, n):
                if j in var.bool_vars:
                    for ti in range(var.lb, var.ub + 1):
                        if ti not in t[i].bool_vars:
                            continue
                        for tj in range(t[j].lb, ti + 1):
                            if tj in t[j].bool_vars:
                                self._clauses.append([-var.bool_vars[j], -t[i].bool_vars[ti], -t[j].bool_vars[tj]])

    def _encode_no_overlap(
        self, starts: tuple["IntVar", ...], durations: tuple[int, ...]
    ) -> None:
        """Encode no-overlap constraint: intervals don't overlap."""
        n = len(starts)
        for i in range(n):
            for j in range(i + 1, n):
                self._encode_disjunctive_le(starts[i], durations[i], starts[j], durations[j])

    def _encode_disjunctive_le(
        self,
        start1: "IntVar",
        dur1: int,
        start2: "IntVar",
        dur2: int,
    ) -> None:
        """Encode: end1 <= start2 OR end2 <= start1."""
        for s1 in range(start1.lb, start1.ub + 1):
            for s2 in range(start2.lb, start2.ub + 1):
                i_before_j = s1 + dur1 <= s2
                j_before_i = s2 + dur2 <= s1

                if not i_before_j and not j_before_i:
                    self._clauses.append([-start1.bool_vars[s1], -start2.bool_vars[s2]])

    def _encode_cumulative(
        self,
        starts: tuple["IntVar", ...],
        durations: tuple[int, ...],
        demands: tuple[int, ...],
        capacity: int,
    ) -> None:
        """Encode cumulative constraint: sum of active demands <= capacity."""
        n = len(starts)
        if n == 0:
            return

        min_start = min(s.lb for s in starts)
        max_end = max(s.ub + d for s, d in zip(starts, durations))

        for t in range(min_start, max_end):
            active_lits = []
            active_demands = []
            for i in range(n):
                for s in range(max(starts[i].lb, t - durations[i] + 1), min(starts[i].ub, t) + 1):
                    if s in starts[i].bool_vars and s <= t < s + durations[i]:
                        active_lits.append(starts[i].bool_vars[s])
                        active_demands.append(demands[i])

            if not active_lits:
                continue

            if len(active_lits) <= 10:
                self._encode_capacity_constraint(active_lits, active_demands, capacity)

    def _encode_capacity_constraint(
        self, lits: list[int], demands: list[int], capacity: int
    ) -> None:
        """Encode sum constraint: if all lits true, demands sum must <= capacity."""
        n = len(lits)
        for size in range(1, n + 1):
            for subset in combinations(range(n), size):
                if sum(demands[i] for i in subset) > capacity:
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

    # -------------------------------------------------------------------------
    # Constraint dispatcher
    # -------------------------------------------------------------------------

    def _encode_constraint(self, constraint: Any) -> None:
        """Encode a single constraint to SAT clauses."""
        if not isinstance(constraint, tuple):
            return

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
            if len(terms) == 0 and const != 0:
                self._clauses.append([])
        elif kind == "sum_eq":
            self._encode_sum_eq(list(constraint[1]), constraint[2])
        elif kind == "sum_le":
            self._encode_sum_le(list(constraint[1]), constraint[2])
        elif kind == "sum_ge":
            self._encode_sum_ge(list(constraint[1]), constraint[2])

    # -------------------------------------------------------------------------
    # Main solve method
    # -------------------------------------------------------------------------

    def solve(
        self,
        *,
        hints: dict[str, int] | None = None,
        solution_limit: int = 1,
        **kwargs: Any,
    ) -> Result[dict[str, int] | None]:
        """Solve the model using SAT encoding."""
        self._clauses = []
        self._encode_vars()

        for constraint in self.model._constraints:
            self._encode_constraint(constraint)

        if any(len(c) == 0 for c in self._clauses):
            return Result(None, 0, 0, 0, Status.INFEASIBLE)

        # Convert hints to SAT assumptions
        assumptions: list[int] = list(kwargs.pop("assumptions", []) or [])
        if hints:
            for name, val in hints.items():
                if name in self.model._vars:
                    var = self.model._vars[name]
                    if val in var.bool_vars:
                        assumptions.append(var.bool_vars[val])

        sat_result = solve_sat(
            self._clauses,
            assumptions=assumptions or None,
            solution_limit=solution_limit,
            **kwargs,
        )

        if sat_result.status == SATStatus.INFEASIBLE:
            return Result(None, 0, sat_result.iterations, sat_result.evaluations, Status.INFEASIBLE)

        if sat_result.status == SATStatus.MAX_ITER:
            return Result(None, 0, sat_result.iterations, sat_result.evaluations, Status.MAX_ITER)

        def decode_sat_solution(sat_sol: dict[int, bool]) -> dict[str, int]:
            """Convert SAT solution to CP variable assignments."""
            cp_sol: dict[str, int] = {}
            for name, var in self.model._vars.items():
                if name.startswith("_"):
                    continue
                for val, bool_var in var.bool_vars.items():
                    if sat_sol.get(bool_var, False):
                        cp_sol[name] = val
                        break
            return cp_sol

        solution = decode_sat_solution(sat_result.solution)  # type: ignore[arg-type]

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

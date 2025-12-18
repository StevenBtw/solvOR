"""
SAT Solver - DPLL with Unit Propagation and Clause Learning

Solves boolean satisfiability: find assignment where all clauses are true.
Uses DPLL (Davis-Putnam-Logemann-Loveland) with unit propagation and basic
conflict-driven clause learning (CDCL).

Usage:
    from solvor.sat import solve_sat, Status
    result = solve_sat([[1, 2], [-1, 3], [-2, -3]])  # (x1 OR x2) AND (NOT x1 OR x3) AND ...
    result = solve_sat(clauses, assumptions=[1, -3])  # with assumed literals

Parameters:
    clauses     : List of clauses, each clause is list of integers (positive = var, negative = NOT var)
    assumptions : Literals that must be true (default: [])
    max_conflicts: Maximum conflicts before restart (default: 100)
    max_restarts : Maximum restarts (default: 100)

Returns Result(solution, objective, iterations, evaluations, status)
    solution = dict mapping variable -> bool, or None if UNSAT
    objective = number of variables (SAT) or 0 (UNSAT)
    iterations = number of decisions made
    evaluations = number of propagations

Status values:
    OPTIMAL    - Satisfying assignment found
    INFEASIBLE - Proven unsatisfiable
    MAX_ITER   - Resource limit reached
"""

from collections import namedtuple, defaultdict
from collections.abc import Sequence
from enum import IntEnum, auto

__all__ = ["solve_sat", "Status", "Result"]

class Status(IntEnum):
    OPTIMAL = auto()
    FEASIBLE = auto()
    INFEASIBLE = auto()
    UNBOUNDED = auto()
    MAX_ITER = auto()

Result = namedtuple('Result', ['solution', 'objective', 'iterations', 'evaluations', 'status'])

def solve_sat(
    clauses: Sequence[Sequence[int]],
    *,
    assumptions: Sequence[int] | None = None,
    max_conflicts: int = 100,
    max_restarts: int = 100,
) -> Result:
    """(clauses, opts) -> Result with satisfying assignment or UNSAT."""
    if not clauses:
        return Result({}, 0, 0, 0, Status.OPTIMAL)

    assumptions = assumptions or []
    clauses = [list(c) for c in clauses]

    variables = set()
    for clause in clauses:
        for lit in clause:
            variables.add(abs(lit))
    n_vars = max(variables) if variables else 0

    watch = defaultdict(list)
    for i, clause in enumerate(clauses):
        if len(clause) >= 1:
            watch[clause[0]].append(i)
        if len(clause) >= 2:
            watch[clause[1]].append(i)

    assignment = {}
    trail = []
    level = {}
    reason = {}
    decisions = 0
    propagations = 0
    conflicts = 0
    restarts = 0
    learned = []

    def value(lit):
        var = abs(lit)
        if var not in assignment:
            return None
        return assignment[var] if lit > 0 else not assignment[var]

    def assign(var, val, dec_level, clause_idx=None):
        nonlocal propagations
        propagations += 1
        assignment[var] = val
        trail.append(var)
        level[var] = dec_level
        reason[var] = clause_idx

    def unassign_to(target_level):
        while trail and level[trail[-1]] > target_level:
            var = trail.pop()
            del assignment[var]
            del level[var]
            del reason[var]

    def propagate(dec_level):
        nonlocal conflicts
        queue = list(trail[-1:]) if trail else []
        head = 0

        while head < len(queue) or any(value(lit) is False for lit in assumptions if abs(lit) not in assignment):
            for lit in assumptions:
                var = abs(lit)
                if var not in assignment:
                    val = lit > 0
                    assign(var, val, dec_level)
                    queue.append(var)

            if head >= len(queue):
                break

            var = queue[head]
            head += 1
            false_lit = -var if assignment[var] else var

            i = 0
            watches = watch[false_lit]
            while i < len(watches):
                ci = watches[i]
                clause = clauses[ci] if ci < len(clauses) else learned[ci - len(clauses)]

                # Unit clauses: if the only literal is false, it's a conflict
                if len(clause) == 1:
                    if value(clause[0]) is False:
                        conflicts += 1
                        return ci
                    i += 1
                    continue

                other = clause[1] if clause[0] == false_lit else clause[0]
                if value(other) is True:
                    i += 1
                    continue

                found = False
                for j in range(2, len(clause)):
                    lit = clause[j]
                    if value(lit) is not False:
                        clause[0], clause[j] = clause[j], clause[0]
                        if clause[0] != false_lit:
                            watches[i] = watches[-1]
                            watches.pop()
                            watch[clause[0]].append(ci)
                            found = True
                            break
                        else:
                            clause[1], clause[j] = clause[j], clause[1]
                            i += 1
                            found = True
                            break

                if found:
                    continue

                if clause[0] == false_lit:
                    clause[0], clause[1] = clause[1], clause[0]

                if value(clause[0]) is False:
                    conflicts += 1
                    return ci
                elif value(clause[0]) is None:
                    assign(abs(clause[0]), clause[0] > 0, dec_level, ci)
                    queue.append(abs(clause[0]))

                i += 1

        return None

    def analyze(conflict_clause_idx):
        if conflict_clause_idx < len(clauses):
            clause = clauses[conflict_clause_idx]
        else:
            clause = learned[conflict_clause_idx - len(clauses)]

        current_level = max(level.get(abs(lit), 0) for lit in clause) if clause else 0
        if current_level == 0:
            return None, -1

        learned_clause = set(clause)
        count = sum(1 for lit in learned_clause if level.get(abs(lit), 0) == current_level)

        trail_idx = len(trail) - 1
        while count > 1 and trail_idx >= 0:
            var = trail[trail_idx]
            trail_idx -= 1

            if var not in [abs(lit) for lit in learned_clause]:
                continue
            if level.get(var, 0) != current_level:
                continue
            if reason.get(var) is None:
                continue

            reason_idx = reason[var]
            if reason_idx < len(clauses):
                reason_clause = clauses[reason_idx]
            else:
                reason_clause = learned[reason_idx - len(clauses)]

            lit = var if var in [abs(x) for x in learned_clause if x > 0 and abs(x) == var] else -var
            if lit in learned_clause:
                learned_clause.remove(lit)
            elif -lit in learned_clause:
                learned_clause.remove(-lit)

            for reason_lit in reason_clause:
                if abs(reason_lit) != var:
                    learned_clause.add(reason_lit)

            count = sum(1 for lit in learned_clause if level.get(abs(lit), 0) == current_level)

        learned_list = list(learned_clause)
        if not learned_list:
            return None, -1

        levels = [level.get(abs(lit), 0) for lit in learned_list]
        if len(set(levels)) <= 1:
            backtrack = 0
        else:
            backtrack = sorted(set(levels))[-2]

        return learned_list, backtrack

    def decide():
        for v in range(1, n_vars + 1):
            if v not in assignment:
                return v
        return None

    dec_level = 0
    conflict = propagate(dec_level)

    while True:
        if conflict is not None:
            if dec_level == 0:
                return Result(None, 0, decisions, propagations, Status.INFEASIBLE)

            learned_clause, backtrack_level = analyze(conflict)
            if learned_clause is None:
                return Result(None, 0, decisions, propagations, Status.INFEASIBLE)

            unassign_to(backtrack_level)
            dec_level = backtrack_level

            clause_idx = len(clauses) + len(learned)
            learned.append(learned_clause)

            if len(learned_clause) >= 1:
                watch[learned_clause[0]].append(clause_idx)
            if len(learned_clause) >= 2:
                watch[learned_clause[1]].append(clause_idx)

            if len(learned_clause) == 1:
                lit = learned_clause[0]
                assign(abs(lit), lit > 0, dec_level, clause_idx)

            if conflicts >= max_conflicts:
                if restarts >= max_restarts:
                    return Result(None, 0, decisions, propagations, Status.MAX_ITER)
                restarts += 1
                conflicts = 0
                unassign_to(0)
                dec_level = 0

            conflict = propagate(dec_level)
            continue

        var = decide()
        if var is None:
            sol = {v: assignment[v] for v in range(1, n_vars + 1) if v in assignment}
            return Result(sol, len(sol), decisions, propagations, Status.OPTIMAL)

        decisions += 1
        dec_level += 1
        assign(var, True, dec_level)
        conflict = propagate(dec_level)

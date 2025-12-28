r"""
SAT solver, boolean satisfiability with clause learning.

You're navigating a maze where every dead end teaches you which turns to avoid.
Hit a contradiction? Learn a new rule that prevents the same mistake. The solver
gets smarter with each conflict, cutting through exponential search space.

    from solvor.sat import solve_sat

    # Clauses in CNF: each clause is OR'd literals, all clauses AND'd
    # Positive int = true, negative = false
    # (x1 OR x2) AND (NOT x1 OR x3) AND (NOT x2 OR NOT x3)
    result = solve_sat([[1, 2], [-1, 3], [-2, -3]])
    result = solve_sat(clauses, solution_limit=10)  # Find multiple solutions

How it works: CDCL (Conflict-Driven Clause Learning) with VSIDS variable ordering.
Makes decisions, propagates unit clauses, and when conflicts occur, learns new
clauses that prevent the same conflict. Restarts periodically with Luby sequence.

Use this for:

- Boolean satisfiability problems
- Logic puzzles (Sudoku encoded as SAT)
- Scheduling conflicts and dependency resolution
- Hardware/software verification

Parameters:

    clauses: list of clauses, each clause is list of literals (int)
    assumptions: literals that must be true
    max_conflicts: conflict limit before giving up
    solution_limit: find multiple solutions

For integer domains use CP. For exact cover use DLX.
Don't use for: optimization (MILP), continuous variables (simplex/gradient).
"""

from collections.abc import Sequence
from heapq import heapify, heappop, heappush

from solvor.types import Result, Status

__all__ = ["solve_sat"]

UNDEF = 2


def lit_var(lit: int) -> int:
    return abs(lit)


def lit_sign(lit: int) -> int:
    return 1 if lit > 0 else 0


def lit_neg(lit: int) -> int:
    return -lit


def luby(i: int) -> int:
    """Luby restart sequence: 1,1,2,1,1,2,4,1,1,2,1,1,2,4,8,..."""
    k = 1
    while True:
        if i == (1 << k) - 1:
            return 1 << (k - 1)
        if i >= (1 << (k - 1)):
            i -= (1 << (k - 1)) - 1
            k = 1
        else:
            k += 1


class BinaryImplications:
    """Binary clause (a ∨ b) means ¬a → b and ¬b → a. Stores clause index for conflict analysis."""

    __slots__ = ("pos", "neg")

    def __init__(self, n_vars: int):
        self.pos: list[list[tuple[int, int]]] = [[] for _ in range(n_vars + 1)]
        self.neg: list[list[tuple[int, int]]] = [[] for _ in range(n_vars + 1)]

    def add(self, lit_a: int, lit_b: int, clause_idx: int) -> None:
        if lit_a > 0:
            self.neg[lit_a].append((lit_b, clause_idx))
        else:
            self.pos[-lit_a].append((lit_b, clause_idx))
        if lit_b > 0:
            self.neg[lit_b].append((lit_a, clause_idx))
        else:
            self.pos[-lit_b].append((lit_a, clause_idx))

    def implications(self, false_lit: int) -> list[tuple[int, int]]:
        return self.neg[false_lit] if false_lit > 0 else self.pos[-false_lit]

    def clear_learned(self, original_count: int) -> None:
        for lst in self.pos:
            lst[:] = [(lit, idx) for lit, idx in lst if idx < original_count]
        for lst in self.neg:
            lst[:] = [(lit, idx) for lit, idx in lst if idx < original_count]


def solve_sat(
    clauses: Sequence[Sequence[int]],
    *,
    assumptions: Sequence[int] | None = None,
    max_conflicts: int = 100_000,
    max_restarts: int = 10_000,
    solution_limit: int = 1,
    luby_factor: int = 100,
) -> Result:
    if not clauses:
        return Result({}, 0, 0, 0)

    all_solutions: list[dict[int, bool]] = []
    assumptions = list(assumptions) if assumptions else []
    clauses = [list(c) for c in clauses]

    # Find all variables
    n_vars = 0
    for clause in clauses:
        for lit in clause:
            n_vars = max(n_vars, lit_var(lit))

    if n_vars == 0:
        return Result({}, 0, 0, 0)

    vals = [UNDEF] * (n_vars + 1)
    levels = [0] * (n_vars + 1)
    reasons = [-1] * (n_vars + 1)

    trail = []
    trail_lim = []
    prop_head = 0

    watch_pos = [[] for _ in range(n_vars + 1)]
    watch_neg = [[] for _ in range(n_vars + 1)]
    big = BinaryImplications(n_vars)

    learned = []
    lbd_scores = []

    # VSIDS: negative activity for max-heap behavior
    activity = [0.0] * (n_vars + 1)
    activity_inc = 1.0
    var_heap = [(-activity[v], v) for v in range(1, n_vars + 1)]
    heapify(var_heap)
    in_heap = [True] * (n_vars + 1)

    phase = [True] * (n_vars + 1)
    decisions = 0
    propagations = 0
    conflicts = 0
    restarts = 0

    def lit_value(lit):
        v = vals[lit_var(lit)]
        if v == UNDEF:
            return None
        return (v == 1) == (lit > 0)

    def watch_list(lit):
        return watch_pos[lit] if lit > 0 else watch_neg[-lit]

    def add_watch(lit, idx):
        if lit > 0:
            watch_pos[lit].append(idx)
        else:
            watch_neg[-lit].append(idx)

    def get_clause(idx):
        return clauses[idx] if idx < len(clauses) else learned[idx - len(clauses)]

    def bump_activity(var):
        nonlocal activity_inc
        activity[var] += activity_inc
        if in_heap[var]:
            heappush(var_heap, (-activity[var], var))

    def decay_activity():
        nonlocal activity_inc
        activity_inc /= 0.95

    def assign(var, val, reason_idx):
        nonlocal propagations
        propagations += 1
        vals[var] = 1 if val else 0
        levels[var] = len(trail_lim)
        reasons[var] = reason_idx
        trail.append(var)

    def unassign_to(level):
        nonlocal prop_head
        while len(trail_lim) > level:
            trail_lim.pop()
        target = trail_lim[-1] if trail_lim else 0
        while len(trail) > target:
            var = trail.pop()
            phase[var] = vals[var] == 1
            vals[var] = UNDEF
            if not in_heap[var]:
                heappush(var_heap, (-activity[var], var))
                in_heap[var] = True
        prop_head = len(trail)

    def find_pure_literals():
        pos_count = [0] * (n_vars + 1)
        neg_count = [0] * (n_vars + 1)
        for clause in clauses:
            for lit in clause:
                if lit > 0:
                    pos_count[lit] += 1
                else:
                    neg_count[-lit] += 1
        pure = []
        for v in range(1, n_vars + 1):
            if pos_count[v] > 0 and neg_count[v] == 0:
                pure.append((v, True))
            elif neg_count[v] > 0 and pos_count[v] == 0:
                pure.append((v, False))
        return pure

    def propagate():
        nonlocal conflicts, prop_head

        if len(trail_lim) == 0:
            for lit in assumptions:
                var = lit_var(lit)
                v = vals[var]
                if v == UNDEF:
                    assign(var, lit > 0, -1)
                elif (v == 1) != (lit > 0):
                    conflicts += 1
                    return -2

        while prop_head < len(trail):
            var = trail[prop_head]
            prop_head += 1
            false_lit = var if vals[var] == 0 else -var

            for implied, clause_idx in big.implications(false_lit):
                impl_var = lit_var(implied)
                if vals[impl_var] == UNDEF:
                    assign(impl_var, implied > 0, clause_idx)
                elif (vals[impl_var] == 1) != (implied > 0):
                    conflicts += 1
                    return clause_idx

            watches = watch_list(false_lit)
            i = 0
            while i < len(watches):
                clause_idx = watches[i]
                clause = get_clause(clause_idx)

                if len(clause) == 1:
                    conflicts += 1
                    return clause_idx

                if clause[0] == false_lit:
                    clause[0], clause[1] = clause[1], clause[0]

                first_val = lit_value(clause[0])
                if first_val is True:
                    i += 1
                    continue

                found = False
                for k in range(2, len(clause)):
                    if lit_value(clause[k]) is not False:
                        clause[1], clause[k] = clause[k], clause[1]
                        watches[i] = watches[-1]
                        watches.pop()
                        add_watch(clause[1], clause_idx)
                        found = True
                        break

                if found:
                    continue

                if first_val is False:
                    conflicts += 1
                    return clause_idx
                else:
                    assign(lit_var(clause[0]), clause[0] > 0, clause_idx)

                i += 1

        return -1

    def analyze(conflict_idx):
        if conflict_idx == -2:
            return None, -1, 0

        clause = get_clause(conflict_idx) if conflict_idx >= 0 else []
        current_level = len(trail_lim)

        if current_level == 0:
            return None, -1, 0

        seen = [False] * (n_vars + 1)
        learned_lits = []
        counter = 0

        def add_lit(lit):
            nonlocal counter
            var = lit_var(lit)
            if seen[var] or vals[var] == UNDEF:
                return
            seen[var] = True
            bump_activity(var)
            if levels[var] == current_level:
                counter += 1
            else:
                learned_lits.append(lit_neg(lit) if (vals[var] == 1) == (lit > 0) else lit)

        for lit in clause:
            add_lit(lit)

        trail_idx = len(trail) - 1
        while counter > 0:
            while trail_idx >= 0 and not seen[trail[trail_idx]]:
                trail_idx -= 1
            if trail_idx < 0:
                break

            var = trail[trail_idx]
            trail_idx -= 1

            if levels[var] == current_level:
                counter -= 1
                if counter == 0:
                    uip_lit = var if vals[var] == 0 else -var
                    learned_lits.insert(0, uip_lit)
                    break

                reason_idx = reasons[var]
                if reason_idx >= 0:
                    for lit in get_clause(reason_idx):
                        if lit_var(lit) != var:
                            add_lit(lit)

        decay_activity()

        if not learned_lits:
            return None, -1, 0

        lvl_set = set(levels[lit_var(lit)] for lit in learned_lits if vals[lit_var(lit)] != UNDEF)
        lvls = sorted(lvl_set, reverse=True)
        bt_level = lvls[1] if len(lvls) > 1 else 0
        lbd = len(lvl_set)

        return learned_lits, bt_level, lbd

    def pick_var():
        while var_heap:
            _, var = heappop(var_heap)
            in_heap[var] = False
            if vals[var] == UNDEF:
                return var
        return 0

    def reduce_db():
        nonlocal learned, lbd_scores
        if len(learned) < 2000:
            return

        indexed = sorted(enumerate(learned), key=lambda x: (lbd_scores[x[0]], len(x[1])))
        keep, keep_lbd = [], []
        for i, (orig_idx, clause) in enumerate(indexed):
            if i < len(indexed) // 2 or lbd_scores[orig_idx] <= 3:
                keep.append(clause)
                keep_lbd.append(lbd_scores[orig_idx])

        learned, lbd_scores = keep, keep_lbd

        for v in range(1, n_vars + 1):
            watch_pos[v] = [c for c in watch_pos[v] if c < len(clauses)]
            watch_neg[v] = [c for c in watch_neg[v] if c < len(clauses)]

        big.clear_learned(len(clauses))

        for i, clause in enumerate(learned):
            idx = len(clauses) + i
            if len(clause) == 2:
                big.add(clause[0], clause[1], idx)
            elif len(clause) > 2:
                add_watch(clause[0], idx)
                add_watch(clause[1], idx)

    unit_clauses = []
    for i, clause in enumerate(clauses):
        if len(clause) == 0:
            return Result(None, 0, 0, 0, Status.INFEASIBLE)
        elif len(clause) == 1:
            unit_clauses.append((clause[0], i))
        elif len(clause) == 2:
            big.add(clause[0], clause[1], i)
        else:
            add_watch(clause[0], i)
            add_watch(clause[1], i)

    for var, val in find_pure_literals():
        if vals[var] == UNDEF:
            assign(var, val, -1)

    for lit, idx in unit_clauses:
        var = lit_var(lit)
        val = lit > 0
        if vals[var] == UNDEF:
            assign(var, val, idx)
        elif (vals[var] == 1) != val:
            return Result(None, 0, 0, 0, Status.INFEASIBLE)

    conflict = propagate()
    if conflict >= 0:
        return Result(None, 0, decisions, propagations, Status.INFEASIBLE)

    dec_level = 0
    conflicts_since_restart = 0
    luby_idx = 1
    next_restart = luby_factor * luby(luby_idx)

    while True:
        if conflict >= 0 or conflict == -2:
            if dec_level == 0 or conflict == -2:
                if all_solutions:
                    return Result(
                        all_solutions[0], len(all_solutions[0]), decisions, propagations, solutions=tuple(all_solutions)
                    )
                return Result(None, 0, decisions, propagations, Status.INFEASIBLE)

            learned_clause, bt_level, lbd = analyze(conflict)

            if learned_clause is None:
                if all_solutions:
                    return Result(
                        all_solutions[0], len(all_solutions[0]), decisions, propagations, solutions=tuple(all_solutions)
                    )
                return Result(None, 0, decisions, propagations, Status.INFEASIBLE)

            unassign_to(bt_level)
            dec_level = bt_level

            clause_idx = len(clauses) + len(learned)
            learned.append(learned_clause)
            lbd_scores.append(lbd)

            if len(learned_clause) == 2:
                big.add(learned_clause[0], learned_clause[1], clause_idx)
            elif len(learned_clause) > 2:
                add_watch(learned_clause[0], clause_idx)
                add_watch(learned_clause[1], clause_idx)

            if learned_clause:
                assign(lit_var(learned_clause[0]), learned_clause[0] > 0, clause_idx)

            conflicts_since_restart += 1

            if conflicts_since_restart >= next_restart:
                if restarts >= max_restarts:
                    if all_solutions:
                        return Result(
                            all_solutions[0],
                            len(all_solutions[0]),
                            decisions,
                            propagations,
                            Status.MAX_ITER,
                            solutions=tuple(all_solutions),
                        )
                    return Result(None, 0, decisions, propagations, Status.MAX_ITER)

                restarts += 1
                luby_idx += 1
                next_restart = luby_factor * luby(luby_idx)
                conflicts_since_restart = 0
                unassign_to(0)
                dec_level = 0
                reduce_db()

            conflict = propagate()
            continue

        var = pick_var()

        if var == 0:
            sol = {v: vals[v] == 1 for v in range(1, n_vars + 1) if vals[v] != UNDEF}
            all_solutions.append(sol)

            if len(all_solutions) >= solution_limit:
                if solution_limit == 1:
                    return Result(sol, len(sol), decisions, propagations)
                return Result(sol, len(sol), decisions, propagations, solutions=tuple(all_solutions))

            blocking = [(-v if vals[v] == 1 else v) for v in range(1, n_vars + 1) if vals[v] != UNDEF]
            clause_idx = len(clauses) + len(learned)
            learned.append(blocking)
            lbd_scores.append(n_vars)

            if len(blocking) >= 2:
                add_watch(blocking[0], clause_idx)
                add_watch(blocking[1], clause_idx)
            elif len(blocking) == 1:
                add_watch(blocking[0], clause_idx)

            unassign_to(0)
            dec_level = 0
            conflict = propagate()
            continue

        decisions += 1
        dec_level += 1
        trail_lim.append(len(trail))
        assign(var, phase[var], -1)
        conflict = propagate()

        if conflicts >= max_conflicts:
            if all_solutions:
                return Result(
                    all_solutions[0],
                    len(all_solutions[0]),
                    decisions,
                    propagations,
                    Status.MAX_ITER,
                    solutions=tuple(all_solutions),
                )
            return Result(None, 0, decisions, propagations, Status.MAX_ITER)

"""
Nurse Scheduling Problem - SAT Encoding

Schedule nurses to shifts over a week while satisfying:
- Coverage requirements (minimum nurses per shift)
- Labor regulations (max hours, rest between shifts)
- Fairness (balanced workload)
- Preferences (requested days off)

This is a simplified version of the NSP (Nurse Scheduling Problem),
a well-studied problem in healthcare operations research.

Source: INRC-II Nurse Rostering Competition benchmarks
        http://www.schedulingbenchmarks.org/nrp/
        Burke et al. (2004) "The State of the Art of Nurse Rostering"

Applications:
- Hospital staff scheduling
- Call center staffing
- Police/fire department rosters
- Any 24/7 service operation

Why this solver: We use direct SAT encoding with sequential counter
cardinality constraints for coverage requirements and workload limits.
"""

from itertools import combinations

from solvor.sat import solve_sat

# Problem data (reduced for demonstration)
NURSES = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]  # Weekdays only
SHIFTS = ["Morning", "Afternoon", "Night"]

# Minimum nurses required per shift
MIN_COVERAGE = {
    "Morning": 2,
    "Afternoon": 2,
    "Night": 1,
}

# Maximum shifts per nurse per week
MAX_SHIFTS_PER_NURSE = 4

# Nurse preferences (days off requests)
PREFERENCES = {
    "Alice": ["Fri"],
    "Bob": ["Mon"],
    "Carol": ["Fri"],
    "Dave": [],
    "Eve": [],
    "Frank": ["Wed"],
}


def solve_nurse_scheduling():
    """Solve nurse scheduling with SAT encoding."""
    print("Nurse Scheduling Problem")
    print("=" * 60)
    print()

    print(f"Nurses: {len(NURSES)}")
    print(f"Days: {len(DAYS)}")
    print(f"Shifts per day: {len(SHIFTS)}")
    print()

    print("Coverage requirements:")
    for shift, min_nurses in MIN_COVERAGE.items():
        print(f"  {shift}: {min_nurses} nurses")
    print()

    # Variable encoding
    n_nurses = len(NURSES)
    n_days = len(DAYS)
    n_shifts = len(SHIFTS)

    next_var = [1]  # Mutable counter for variable IDs

    def new_var():
        v = next_var[0]
        next_var[0] += 1
        return v

    # Pre-allocate variable IDs for nurse-day-shift assignments
    var_map = {}
    for nurse_idx in range(n_nurses):
        for day_idx in range(n_days):
            for shift_idx in range(n_shifts):
                var_map[(nurse_idx, day_idx, shift_idx)] = new_var()

    def var_id(nurse_idx, day_idx, shift_idx):
        return var_map[(nurse_idx, day_idx, shift_idx)]

    clauses = []

    # Sequential counter encoding for at_most_k
    # More efficient than naive encoding for larger n
    def at_most_k_seq(lits, k):
        n = len(lits)
        if k >= n:
            return  # Always satisfied
        if k < 0:
            clauses.append([])
            return
        if k == 0:
            for lit in lits:
                clauses.append([-lit])
            return

        # Create counter variables: s[i][j] = true if at least j of x[0..i] are true
        s = [[new_var() for _ in range(k + 1)] for _ in range(n)]

        # s[0][0] is always true (at least 0 true among first 1)
        # s[0][1] = x[0]
        clauses.append([-lits[0], s[0][0]])  # x[0] -> s[0][0]
        clauses.append([s[0][0]])  # s[0][0] is always true (at least 0)
        if k >= 1:
            clauses.append([-lits[0], s[0][0]])
            # s[0][1] <-> x[0]
            clauses.append([-s[0][0], -lits[0], s[0][0]])  # trivially true

        # Simpler approach: use sequential counters
        # s[i][j] means "sum of x[0..i] >= j+1"
        # We want sum <= k, so s[n-1][k] must be false

        # Reset and use cleaner encoding
        # r[i] = number of true variables among x[0..i]
        # We use unary representation: r[i][j] = 1 iff count >= j

        # Actually, let's use a simpler approach for this example
        # For at_most_k with moderate n and k, pairwise encoding works
        if n <= 10 or k >= n - 2:
            # Use naive encoding for small problems
            for subset in combinations(range(n), k + 1):
                clauses.append([-lits[i] for i in subset])
        else:
            # For larger problems, use commander encoding
            # Split into groups and use auxiliary commander variables
            group_size = 3
            groups = [lits[i : i + group_size] for i in range(0, n, group_size)]

            # Each group gets a commander variable
            commanders = []
            for group in groups:
                if len(group) == 1:
                    commanders.append(group[0])
                else:
                    # Commander is true iff any in group is true
                    cmd = new_var()
                    commanders.append(cmd)
                    # cmd -> OR(group): -cmd OR g1 OR g2 OR ...
                    clauses.append([-cmd] + group)
                    # OR(group) -> cmd: for each g in group: -g OR cmd
                    for g in group:
                        clauses.append([-g, cmd])
                    # At most one in group can be true (for k=1 in group)
                    for g1, g2 in combinations(group, 2):
                        clauses.append([-g1, -g2])

            # Now apply at_most_k to commanders
            if len(commanders) > k + 1:
                for subset in combinations(range(len(commanders)), k + 1):
                    clauses.append([-commanders[i] for i in subset])

    # Helper: at_least_k constraint
    def at_least_k(lits, k):
        n = len(lits)
        if k <= 0:
            return
        if k > n:
            clauses.append([])
            return
        if k == 1:
            clauses.append(list(lits))
            return
        # For at least k: any n-k+1 must have at least one true
        for subset in combinations(range(n), n - k + 1):
            clauses.append([lits[i] for i in subset])

    # Helper: at_most_k constraint (uses sequential counter for efficiency)
    def at_most_k(lits, k):
        n = len(lits)
        if k >= n:
            return
        if k < 0:
            clauses.append([])
            return
        # Use naive encoding for small n, efficient encoding for large n
        if n <= 8:
            for subset in combinations(range(n), k + 1):
                clauses.append([-lits[i] for i in subset])
        else:
            at_most_k_seq(lits, k)

    # Constraint 1: Coverage - minimum nurses per shift
    for day_idx, day in enumerate(DAYS):
        for shift_idx, shift in enumerate(SHIFTS):
            nurse_vars = [var_id(n, day_idx, shift_idx) for n in range(n_nurses)]
            at_least_k(nurse_vars, MIN_COVERAGE[shift])

    # Constraint 2: Max one shift per nurse per day
    for nurse_idx in range(n_nurses):
        for day_idx in range(n_days):
            shift_vars = [var_id(nurse_idx, day_idx, s) for s in range(n_shifts)]
            at_most_k(shift_vars, 1)

    # Constraint 3: Max shifts per week per nurse
    for nurse_idx in range(n_nurses):
        all_shifts = [var_id(nurse_idx, d, s) for d in range(n_days) for s in range(n_shifts)]
        at_most_k(all_shifts, MAX_SHIFTS_PER_NURSE)

    # Constraint 4: No night shift followed by morning shift next day
    night_idx = SHIFTS.index("Night")
    morning_idx = SHIFTS.index("Morning")
    for nurse_idx in range(n_nurses):
        for day_idx in range(n_days - 1):
            night_var = var_id(nurse_idx, day_idx, night_idx)
            morning_var = var_id(nurse_idx, day_idx + 1, morning_idx)
            clauses.append([-night_var, -morning_var])

    # Constraint 5: Preferences are soft (tracked but not enforced)

    print(f"SAT instance: {next_var[0] - 1} variables, {len(clauses)} clauses")
    print()

    # Solve
    result = solve_sat(clauses)

    if result.ok:
        print("Schedule found!")
        print()

        sol = result.solution

        # Build schedule
        schedule = {day: {shift: [] for shift in SHIFTS} for day in DAYS}
        nurse_counts = {nurse: 0 for nurse in NURSES}

        for nurse_idx, nurse in enumerate(NURSES):
            for day_idx, day in enumerate(DAYS):
                for shift_idx, shift in enumerate(SHIFTS):
                    if sol.get(var_id(nurse_idx, day_idx, shift_idx), False):
                        schedule[day][shift].append(nurse)
                        nurse_counts[nurse] += 1

        # Display schedule
        print("Weekly Schedule:")
        print("-" * 52)
        header = f"{'Shift':<12}" + "".join(f"{day:<8}" for day in DAYS)
        print(header)
        print("-" * 52)

        for shift in SHIFTS:
            row = f"{shift:<12}"
            for day in DAYS:
                nurses = schedule[day][shift]
                if nurses:
                    row += f"{','.join(n[0] for n in nurses):<8}"
                else:
                    row += f"{'-':<8}"
            print(row)

        print("-" * 52)
        print()

        # Show nurse workloads
        print("Nurse workloads:")
        for nurse in NURSES:
            shifts = nurse_counts[nurse]
            prefs = PREFERENCES.get(nurse, [])
            violations = 0
            nurse_idx = NURSES.index(nurse)
            for day in prefs:
                if day in DAYS:
                    day_idx = DAYS.index(day)
                    for shift_idx in range(n_shifts):
                        if sol.get(var_id(nurse_idx, day_idx, shift_idx), False):
                            violations += 1
            pref_str = f" (wanted off: {', '.join(prefs)})" if prefs else ""
            viol_str = f" [violations: {violations}]" if violations else ""
            print(f"  {nurse}: {shifts} shifts{pref_str}{viol_str}")

        print()

        # Verify coverage
        print("Coverage verification:")
        all_covered = True
        for day in DAYS:
            for shift in SHIFTS:
                count = len(schedule[day][shift])
                required = MIN_COVERAGE[shift]
                if count < required:
                    all_covered = False
                    print(f"  {day} {shift}: {count}/{required} X")

        if all_covered:
            print("  All shifts adequately staffed OK")

    else:
        print(f"No feasible schedule found. Status: {result.status}")
        print("Consider adding more nurses or relaxing constraints.")

    return result


if __name__ == "__main__":
    solve_nurse_scheduling()

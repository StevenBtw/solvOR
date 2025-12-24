"""
School Timetabling with SAT Solver

Approach:
    1. Use SAT to find valid timeslot assignments (ignoring rooms)
    2. Use local search to repair room capacity violations

Why this solver:
    SAT efficiently handles the hard constraints (teacher/group conflicts)
    as boolean clauses. Room capacity is softer and handled by repair.
"""

from time import perf_counter

from timetabling_common import build_conflict_groups, load_data, print_schedule

from solvor import solve_sat


def solve(max_conflicts=10_000_000):
    """
    Solve using SAT for timeslots + local search repair for room capacity.
    """
    data = load_data()
    lessons = data["lessons"]
    timeslots = data["timeslots"]
    rooms = data["rooms"]
    n_timeslots = len(timeslots)
    n_rooms = len(rooms)
    n_lessons = len(lessons)

    teachers, groups = build_conflict_groups(lessons)

    # Variable: x[lesson][timeslot] = True if lesson assigned to timeslot
    def var(lesson, timeslot):
        return lesson * n_timeslots + timeslot + 1

    clauses = []

    # Each lesson gets exactly one timeslot
    for i in range(n_lessons):
        clauses.append([var(i, t) for t in range(n_timeslots)])
        for t1 in range(n_timeslots):
            for t2 in range(t1 + 1, n_timeslots):
                clauses.append([-var(i, t1), -var(i, t2)])

    # Teacher constraint
    for lesson_ids in teachers.values():
        if len(lesson_ids) > 1:
            for t in range(n_timeslots):
                for j, i1 in enumerate(lesson_ids):
                    for i2 in lesson_ids[j + 1 :]:
                        clauses.append([-var(i1, t), -var(i2, t)])

    # Group constraint
    for lesson_ids in groups.values():
        if len(lesson_ids) > 1:
            for t in range(n_timeslots):
                for j, i1 in enumerate(lesson_ids):
                    for i2 in lesson_ids[j + 1 :]:
                        clauses.append([-var(i1, t), -var(i2, t)])

    result = solve_sat(clauses, max_conflicts=max_conflicts)

    if result.status.name != "OPTIMAL":
        return result, None

    # Extract timeslot assignment
    ts_assignment = []
    for i in range(n_lessons):
        for t in range(n_timeslots):
            if result.solution.get(var(i, t), False):
                ts_assignment.append(t)
                break

    # Local search repair: move lessons from overloaded timeslots
    from collections import Counter

    assignment = list(ts_assignment)

    # Build conflict lookup for quick validation
    teacher_of = {i: lessons[i]["teacher"] for i in range(n_lessons)}
    group_of = {i: lessons[i]["group"] for i in range(n_lessons)}

    def can_move(lesson_idx, new_ts, current_assignment):
        """Check if lesson can move to new_ts without teacher/group conflict."""
        for j, ts in enumerate(current_assignment):
            if j != lesson_idx and ts == new_ts:
                if teacher_of[lesson_idx] == teacher_of[j]:
                    return False
                if group_of[lesson_idx] == group_of[j]:
                    return False
        return True

    # Repair loop
    for _ in range(1000):
        ts_counts = Counter(assignment)
        overloaded = [t for t, c in ts_counts.items() if c > n_rooms]

        if not overloaded:
            break

        # Find a lesson in overloaded timeslot that can move
        moved = False
        for viol_t in overloaded:
            lessons_at_t = [i for i, t in enumerate(assignment) if t == viol_t]
            for lesson_idx in lessons_at_t:
                # Try moving to underloaded timeslots
                for new_t in range(n_timeslots):
                    if new_t != viol_t and ts_counts[new_t] < n_rooms:
                        if can_move(lesson_idx, new_t, assignment):
                            assignment[lesson_idx] = new_t
                            moved = True
                            break
                if moved:
                    break
            if moved:
                break

        if not moved:
            break  # No valid moves found

    return result, assignment


if __name__ == "__main__":
    data = load_data()
    n_timeslots = len(data["timeslots"])
    n_rooms = len(data["rooms"])
    print(f"School Timetabling SAT: {len(data['lessons'])} lessons")
    print(f"  {n_timeslots} timeslots, {n_rooms} rooms")

    start = perf_counter()
    result, ts_assignment = solve()
    elapsed = perf_counter() - start

    print(f"Time: {elapsed:.3f}s")
    print(f"Status: {result.status}")
    print(f"Iterations: {result.iterations}")

    if ts_assignment:
        room_usage = {t: 0 for t in range(n_timeslots)}
        full_assignment = []
        for ts in ts_assignment:
            room = room_usage[ts] % n_rooms
            room_usage[ts] += 1
            full_assignment.append(ts * n_rooms + room)

        from timetabling_common import count_violations

        teachers, groups = build_conflict_groups(data["lessons"])
        violations = count_violations(full_assignment, teachers, groups, n_rooms)
        print(f"Violations: {violations}")

        if violations == 0:
            print("\nSchedule:")
            print_schedule(full_assignment, data["lessons"], data["timeslots"], data["rooms"])

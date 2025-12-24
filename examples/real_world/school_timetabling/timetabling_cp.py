"""
School Timetabling with Constraint Programming

Approach:
    1. Use CP with all_different constraints for teacher/group conflicts
    2. Use local search to repair room capacity violations

Why this solver:
    CP's all_different constraint naturally expresses "no two lessons
    for the same teacher can share a timeslot". More declarative than SAT.
"""

from collections import Counter
from time import perf_counter

from timetabling_common import build_conflict_groups, load_data

from solvor.cp import Model


def solve(max_conflicts=10_000_000):
    """Solve using CP for timeslots + local search repair for room capacity."""
    data = load_data()
    lessons = data["lessons"]
    timeslots = data["timeslots"]
    rooms = data["rooms"]
    n_timeslots = len(timeslots)
    n_rooms = len(rooms)
    n_lessons = len(lessons)

    teachers, groups = build_conflict_groups(lessons)

    model = Model()

    # Each lesson gets a timeslot
    ts_vars = {}
    for i in range(n_lessons):
        ts_vars[i] = model.int_var(0, n_timeslots - 1, f"ts_{i}")

    # Teacher constraint: different timeslots for same teacher
    for lesson_ids in teachers.values():
        if len(lesson_ids) > 1:
            model.add(model.all_different([ts_vars[i] for i in lesson_ids]))

    # Student group constraint: different timeslots for same group
    for lesson_ids in groups.values():
        if len(lesson_ids) > 1:
            model.add(model.all_different([ts_vars[i] for i in lesson_ids]))

    result = model.solve(max_conflicts=max_conflicts)

    if result.status.name != "OPTIMAL":
        return result

    # Extract timeslot assignment
    assignment = [result.solution[f"ts_{i}"] for i in range(n_lessons)]

    # Local search repair: move lessons from overloaded timeslots
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

        moved = False
        for viol_t in overloaded:
            lessons_at_t = [i for i, t in enumerate(assignment) if t == viol_t]
            for lesson_idx in lessons_at_t:
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
            break

    # Update result solution with repaired assignment
    repaired_solution = {f"ts_{i}": assignment[i] for i in range(n_lessons)}
    from solvor.types import Result

    return Result(repaired_solution, 0, result.iterations, result.evaluations)


if __name__ == "__main__":
    data = load_data()
    print(f"School Timetabling CP: {len(data['lessons'])} lessons")
    print(f"  {len(data['timeslots'])} timeslots, {len(data['rooms'])} rooms")

    start = perf_counter()
    result = solve()
    elapsed = perf_counter() - start

    print(f"Time: {elapsed:.3f}s")
    print(f"Status: {result.status}")

    if result.solution:
        n_lessons = len(data["lessons"])
        n_timeslots = len(data["timeslots"])
        n_rooms = len(data["rooms"])
        ts_assignment = [result.solution[f"ts_{i}"] for i in range(n_lessons)]

        # Convert to full slots
        room_usage = {t: 0 for t in range(n_timeslots)}
        full_assignment = []
        for ts in ts_assignment:
            full_assignment.append(ts * n_rooms + room_usage[ts] % n_rooms)
            room_usage[ts] += 1

        from timetabling_common import build_conflict_groups, count_violations, print_schedule

        teachers, groups = build_conflict_groups(data["lessons"])
        violations = count_violations(full_assignment, teachers, groups, n_rooms)
        print(f"Violations: {violations}")

        if violations == 0:
            print("\nSchedule:")
            print_schedule(full_assignment, data["lessons"], data["timeslots"], data["rooms"])

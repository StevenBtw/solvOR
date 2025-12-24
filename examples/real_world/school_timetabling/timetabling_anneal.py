"""
School Timetabling with Simulated Annealing

Approach:
    Random initial assignment, then anneal by moving random lessons
    to random slots, minimizing total constraint violations.

Why this solver:
    Simple to implement, handles all constraints uniformly. Works well
    for "soft" optimization where some violations might be acceptable.
"""

from random import randint
from random import seed as set_seed
from time import perf_counter

from timetabling_common import build_conflict_groups, count_violations, load_data, print_schedule

from solvor import anneal


def solve(max_iter=50000, seed=42):
    """Solve using simulated annealing."""
    data = load_data()
    lessons = data["lessons"]
    timeslots = data["timeslots"]
    rooms = data["rooms"]
    n_timeslots = len(timeslots)
    n_rooms = len(rooms)
    n_slots = n_timeslots * n_rooms  # 10 timeslots × 3 rooms = 30 slots
    n_lessons = len(lessons)

    teachers, groups = build_conflict_groups(lessons)

    set_seed(seed)
    initial = [randint(0, n_slots - 1) for _ in range(n_lessons)]

    def objective(state):
        return count_violations(state, teachers, groups, n_rooms)

    def neighbor(state):
        new_state = list(state)
        i = randint(0, n_lessons - 1)
        new_state[i] = randint(0, n_slots - 1)
        return new_state

    return anneal(initial, objective, neighbor, max_iter=max_iter, seed=seed)


if __name__ == "__main__":
    data = load_data()
    n_slots = len(data["timeslots"]) * len(data["rooms"])
    print(f"School Timetabling Anneal: {len(data['lessons'])} lessons, {n_slots} slots")

    start = perf_counter()
    result = solve()
    elapsed = perf_counter() - start

    teachers, groups = build_conflict_groups(data["lessons"])
    violations = count_violations(result.solution, teachers, groups, len(data["rooms"]))
    print(f"Time: {elapsed:.3f}s")
    print(f"Violations: {violations}")
    print(f"Iterations: {result.iterations}")

    if violations == 0:
        print("\nFEASIBLE! Schedule:")
        print_schedule(result.solution, data["lessons"], data["timeslots"], data["rooms"])

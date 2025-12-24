"""
School Timetabling with Genetic Algorithm

Approach:
    Population of random assignments evolves via tournament selection,
    uniform crossover, and point mutation. Fitness = violation count.
    Includes local repair phase for remaining violations.

Why this solver:
    GA maintains population diversity, good at escaping local optima.
    Crossover can combine good partial solutions from different parents.
"""

from random import randint
from random import seed as set_seed
from time import perf_counter

from timetabling_common import build_conflict_groups, count_violations, load_data, print_schedule

from solvor import evolve


def solve(max_gen=500, pop_size=100, seed=42):
    """Solve using genetic algorithm."""
    data = load_data()
    lessons = data["lessons"]
    timeslots = data["timeslots"]
    rooms = data["rooms"]
    n_timeslots = len(timeslots)
    n_rooms = len(rooms)
    n_slots = n_timeslots * n_rooms
    n_lessons = len(lessons)

    teachers, groups = build_conflict_groups(lessons)

    set_seed(seed)

    def fitness(individual):
        return count_violations(individual, teachers, groups, n_rooms)

    def crossover(parent1, parent2):
        child = []
        for i in range(n_lessons):
            child.append(parent1[i] if randint(0, 1) else parent2[i])
        return tuple(child)

    def mutate(individual):
        idx = randint(0, n_lessons - 1)
        new_slot = randint(0, n_slots - 1)
        return individual[:idx] + (new_slot,) + individual[idx + 1 :]

    # Create initial population
    population = [tuple(randint(0, n_slots - 1) for _ in range(n_lessons)) for _ in range(pop_size)]

    result = evolve(fitness, population, crossover, mutate, max_gen=max_gen, seed=seed)

    # Local repair: fix any remaining violations
    solution = list(result.solution)
    teacher_of = {i: lessons[i]["teacher"] for i in range(n_lessons)}
    group_of = {i: lessons[i]["group"] for i in range(n_lessons)}

    def get_timeslot(slot):
        return slot // n_rooms

    def can_move_to_slot(lesson_idx, new_slot, current):
        """Check if lesson can move to new_slot without conflicts."""
        new_ts = get_timeslot(new_slot)
        for j, slot in enumerate(current):
            if j != lesson_idx and get_timeslot(slot) == new_ts:
                if teacher_of[lesson_idx] == teacher_of[j]:
                    return False
                if group_of[lesson_idx] == group_of[j]:
                    return False
        # Check room conflict
        if new_slot in current:
            idx = current.index(new_slot)
            if idx != lesson_idx:
                return False
        return True

    # Repair loop - try to fix violations
    for _ in range(1000):
        violations = count_violations(solution, teachers, groups, n_rooms)
        if violations == 0:
            break

        # Find a lesson involved in a violation and try to move it
        moved = False
        for i in range(n_lessons):
            current_slot = solution[i]
            current_ts = get_timeslot(current_slot)

            # Check if this lesson has a conflict
            has_conflict = False
            for j in range(n_lessons):
                if i != j:
                    other_ts = get_timeslot(solution[j])
                    if current_ts == other_ts:
                        if teacher_of[i] == teacher_of[j] or group_of[i] == group_of[j]:
                            has_conflict = True
                            break
                    if solution[i] == solution[j]:
                        has_conflict = True
                        break

            if has_conflict:
                # Try moving to a different slot
                for new_slot in range(n_slots):
                    if new_slot != current_slot and can_move_to_slot(i, new_slot, solution):
                        solution[i] = new_slot
                        moved = True
                        break
                if moved:
                    break

        if not moved:
            break

    # Count final violations after repair
    final_violations = count_violations(solution, teachers, groups, n_rooms)
    from solvor.types import Result

    return Result(tuple(solution), final_violations, result.iterations, result.evaluations)


if __name__ == "__main__":
    data = load_data()
    n_slots = len(data["timeslots"]) * len(data["rooms"])
    print(f"School Timetabling Genetic: {len(data['lessons'])} lessons, {n_slots} slots")

    start = perf_counter()
    result = solve()
    elapsed = perf_counter() - start

    teachers, groups = build_conflict_groups(data["lessons"])
    violations = count_violations(result.solution, teachers, groups, len(data["rooms"]))
    print(f"Time: {elapsed:.3f}s")
    print(f"Violations: {violations}")
    print(f"Generations: {result.iterations}")

    if violations == 0:
        print("\nFEASIBLE! Schedule:")
        print_schedule(result.solution, data["lessons"], data["timeslots"], data["rooms"])

"""
School Timetabling with Tabu Search

Approach:
    Start from random assignment. Each iteration, sample random moves
    (lesson to new slot) and pick the best non-tabu move. Recently
    visited moves are forbidden for a cooldown period.

Why this solver:
    Tabu search is systematic and deterministic. Memory prevents cycling.
    Neighborhood sampling makes it practical for large instances.
"""

from random import Random, randint
from random import seed as set_seed
from time import perf_counter

from timetabling_common import build_conflict_groups, count_violations, load_data, print_schedule

from solvor import tabu_search


def solve(max_iter=2000, seed=42, sample_size=500):
    """Solve using tabu search with neighborhood sampling.

    For large problems, exploring all neighbors is expensive. Standard practice
    is to sample a random subset of the neighborhood each iteration.
    """
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
    rng = Random(seed)
    initial = tuple(randint(0, n_slots - 1) for _ in range(n_lessons))

    def objective(state):
        return count_violations(state, teachers, groups, n_rooms)

    def neighbors(state):
        """Generate sampled (move, new_state) pairs. Move is (lesson_idx, new_slot)."""
        result = []
        for _ in range(sample_size):
            i = rng.randint(0, n_lessons - 1)
            s = rng.randint(0, n_slots - 1)
            if state[i] != s:
                new_state = list(state)
                new_state[i] = s
                result.append(((i, s), tuple(new_state)))
        return result

    return tabu_search(initial, objective, neighbors, max_iter=max_iter, cooldown=20, seed=seed)


if __name__ == "__main__":
    data = load_data()
    n_slots = len(data["timeslots"]) * len(data["rooms"])
    print(f"School Timetabling Tabu: {len(data['lessons'])} lessons, {n_slots} slots")

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

"""
Common utilities for school timetabling solutions.

Dataset: ITC 2007 International Timetabling Competition, Track 3
https://www.itc2007.cs.qub.ac.uk/curricula/

The competition dataset (comp01) contains 160 lessons across 30 timeslots
and 6 rooms, representing a typical university course timetabling scenario.

Reference:
    Di Gaspero, L. and McCollum, B. and Schaerf, A. (2007)
    "The Second International Timetabling Competition (ITC-2007):
    Curriculum-based Course Timetabling (Track 3)"
"""

import json
from pathlib import Path

DATA_FILE = Path(__file__).parent / "itc2007_comp01.json"


def load_data():
    """Load timetabling data from JSON."""
    with open(DATA_FILE, encoding="utf-8") as f:
        return json.load(f)


def build_conflict_groups(lessons):
    """Build teacher and student group conflict sets."""
    teachers = {}
    groups = {}
    for i, lesson in enumerate(lessons):
        teachers.setdefault(lesson["teacher"], []).append(i)
        groups.setdefault(lesson["group"], []).append(i)
    return teachers, groups


def count_violations(assignment, teachers, groups, n_rooms):
    """Count constraint violations for a (timeslot, room) assignment.

    assignment[i] = timeslot * n_rooms + room, decoded as:
    - timeslot = assignment[i] // n_rooms
    - room = assignment[i] % n_rooms

    Hard constraints:
    - No teacher teaches two lessons at the same timeslot
    - No room hosts two lessons at the same (timeslot, room)
    - No student group has two lessons at the same timeslot
    """
    violations = 0

    # Teacher conflicts: same timeslot (any room)
    for lesson_ids in teachers.values():
        timeslots = [assignment[i] // n_rooms for i in lesson_ids]
        violations += len(timeslots) - len(set(timeslots))

    # Group conflicts: same timeslot (any room)
    for lesson_ids in groups.values():
        timeslots = [assignment[i] // n_rooms for i in lesson_ids]
        violations += len(timeslots) - len(set(timeslots))

    # Room conflicts: same (timeslot, room)
    all_slots = list(assignment)
    violations += len(all_slots) - len(set(all_slots))

    return violations


def print_schedule(assignment, lessons, timeslots, rooms):
    """Print schedule from (timeslot, room) assignment."""
    n_rooms = len(rooms)
    schedule = {}

    for i, lesson in enumerate(lessons):
        slot = assignment[i]
        ts_idx = slot // n_rooms
        room_idx = slot % n_rooms
        ts_name = timeslots[ts_idx]
        room_name = rooms[room_idx]
        schedule.setdefault(ts_name, []).append((room_name, lesson["subject"], lesson["teacher"], lesson["group"]))

    for ts_name in timeslots:
        if ts_name in schedule:
            print(f"\n{ts_name}:")
            for room, subj, teacher, group in sorted(schedule[ts_name]):
                print(f"  {room}: {subj} ({teacher}) - {group}")

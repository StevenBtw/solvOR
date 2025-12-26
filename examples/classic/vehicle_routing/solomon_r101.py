"""
Solomon R101 - VRPTW Benchmark Instance

The Solomon benchmarks are the most widely used VRPTW test instances.
R101 has 100 customers with randomly distributed locations and tight time windows.

Source: Solomon (1987) "Algorithms for the Vehicle Routing and Scheduling
        Problems with Time Window Constraints"
        https://www.sintef.no/projectweb/top/vrptw/solomon-benchmark/

Best known solution: 19 vehicles, total distance ~1637.7 (varies by distance rounding)

Instance characteristics:
- 100 customers + depot
- Random customer locations (R = Random)
- Tight time windows
- Vehicle capacity: 200
"""

# R101 data: (x, y, demand, ready_time, due_time, service_time)
# Customer 0 is the depot
R101_DATA = [
    (35, 35, 0, 0, 230, 0),  # 0: Depot
    (41, 49, 10, 161, 171, 10),  # 1
    (35, 17, 7, 50, 60, 10),  # 2
    (55, 45, 13, 116, 126, 10),  # 3
    (55, 20, 19, 149, 159, 10),  # 4
    (15, 30, 26, 34, 44, 10),  # 5
    (25, 30, 3, 99, 109, 10),  # 6
    (20, 50, 5, 81, 91, 10),  # 7
    (10, 43, 9, 95, 105, 10),  # 8
    (55, 60, 16, 97, 107, 10),  # 9
    (30, 60, 16, 124, 134, 10),  # 10
    (20, 65, 12, 67, 77, 10),  # 11
    (50, 35, 19, 63, 73, 10),  # 12
    (30, 25, 23, 159, 169, 10),  # 13
    (15, 10, 20, 32, 42, 10),  # 14
    (30, 5, 8, 61, 71, 10),  # 15
    (10, 20, 19, 75, 85, 10),  # 16
    (5, 30, 2, 157, 167, 10),  # 17
    (20, 40, 12, 87, 97, 10),  # 18
    (15, 60, 17, 76, 86, 10),  # 19
    (45, 65, 9, 126, 136, 10),  # 20
    (45, 20, 11, 62, 72, 10),  # 21
    (45, 10, 18, 97, 107, 10),  # 22
    (55, 5, 29, 68, 78, 10),  # 23
    (65, 35, 3, 153, 163, 10),  # 24
    (65, 20, 6, 172, 182, 10),  # 25
    (45, 30, 17, 132, 142, 10),  # 26
    (35, 40, 16, 37, 47, 10),  # 27
    (41, 37, 16, 39, 49, 10),  # 28
    (64, 42, 9, 63, 73, 10),  # 29
    (40, 60, 21, 71, 81, 10),  # 30
    (31, 52, 27, 50, 60, 10),  # 31
    (35, 69, 23, 141, 151, 10),  # 32
    (53, 52, 11, 37, 47, 10),  # 33
    (65, 55, 14, 117, 127, 10),  # 34
    (63, 65, 8, 143, 153, 10),  # 35
    (2, 60, 5, 41, 51, 10),  # 36
    (20, 20, 8, 134, 144, 10),  # 37
    (5, 5, 16, 83, 93, 10),  # 38
    (60, 12, 31, 44, 54, 10),  # 39
    (40, 25, 9, 85, 95, 10),  # 40
    (42, 7, 5, 97, 107, 10),  # 41
    (24, 12, 5, 31, 41, 10),  # 42
    (23, 3, 7, 132, 142, 10),  # 43
    (11, 14, 18, 69, 79, 10),  # 44
    (6, 38, 16, 32, 42, 10),  # 45
    (2, 48, 1, 117, 127, 10),  # 46
    (8, 56, 27, 51, 61, 10),  # 47
    (13, 52, 36, 165, 175, 10),  # 48
    (6, 68, 30, 108, 118, 10),  # 49
    (47, 47, 13, 124, 134, 10),  # 50
    # Truncated to 50 customers for faster demo
    # Full R101 has 100 customers
]

VEHICLE_CAPACITY = 200
BEST_KNOWN_VEHICLES = 19  # For full 100-customer instance
BEST_KNOWN_DISTANCE = 1637.7  # Approximate


def euclidean_distance(c1: tuple, c2: tuple) -> float:
    """Euclidean distance between two customers."""
    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]
    return (dx * dx + dy * dy) ** 0.5


if __name__ == "__main__":
    print("Solomon R101 (first 50 customers)")
    print(f"Customers: {len(R101_DATA) - 1}")
    print(f"Vehicle capacity: {VEHICLE_CAPACITY}")
    print(f"Best known (full): {BEST_KNOWN_VEHICLES} vehicles, {BEST_KNOWN_DISTANCE:.1f} distance")

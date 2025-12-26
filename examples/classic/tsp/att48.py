"""
ATT48 - 48 capitals of the contiguous US states

This is one of the most famous TSPLIB benchmark instances.
Coordinates represent pseudo-Euclidean distances between 48 US state capitals.

Source: TSPLIB (Reinelt, 1991)
http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/

Optimal tour length: 10628 (using att distance function)
"""

# ATT48 coordinates from TSPLIB
# Format: (x, y) for each city
CITIES = [
    (6734, 1453),  # 1
    (2233, 10),  # 2
    (5530, 1424),  # 3
    (401, 841),  # 4
    (3082, 1644),  # 5
    (7608, 4458),  # 6
    (7573, 3716),  # 7
    (7265, 1268),  # 8
    (6898, 1885),  # 9
    (1112, 2049),  # 10
    (5468, 2606),  # 11
    (5989, 2873),  # 12
    (4706, 2674),  # 13
    (4612, 2035),  # 14
    (6347, 2683),  # 15
    (6107, 669),  # 16
    (7611, 5184),  # 17
    (7462, 3590),  # 18
    (7732, 4723),  # 19
    (5900, 3561),  # 20
    (4483, 3369),  # 21
    (6101, 1110),  # 22
    (5199, 2182),  # 23
    (1633, 2809),  # 24
    (4307, 2322),  # 25
    (675, 1006),  # 26
    (7555, 4819),  # 27
    (7541, 3981),  # 28
    (3177, 756),  # 29
    (7352, 4506),  # 30
    (7545, 2801),  # 31
    (3245, 3305),  # 32
    (6426, 3173),  # 33
    (4608, 1198),  # 34
    (23, 2216),  # 35
    (7248, 3779),  # 36
    (7762, 4595),  # 37
    (7392, 2244),  # 38
    (3484, 2829),  # 39
    (6271, 2135),  # 40
    (4985, 140),  # 41
    (1916, 1569),  # 42
    (7280, 4899),  # 43
    (7509, 3239),  # 44
    (10, 2676),  # 45
    (6807, 2993),  # 46
    (5185, 3258),  # 47
    (3023, 1942),  # 48
]

OPTIMAL_TOUR_LENGTH = 10628

# Known optimal tour (0-indexed)
OPTIMAL_TOUR = [
    0,
    7,
    37,
    30,
    43,
    17,
    6,
    27,
    5,
    36,
    18,
    26,
    16,
    42,
    29,
    35,
    45,
    32,
    19,
    46,
    20,
    31,
    38,
    47,
    4,
    41,
    23,
    9,
    44,
    34,
    3,
    25,
    1,
    28,
    33,
    40,
    15,
    21,
    2,
    22,
    13,
    24,
    12,
    10,
    11,
    14,
    39,
    8,
]


def att_distance(c1: tuple[int, int], c2: tuple[int, int]) -> int:
    """
    ATT distance function from TSPLIB.

    This is a pseudo-Euclidean distance used in the original benchmark.
    """
    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]
    rij = (dx * dx + dy * dy) ** 0.5
    tij = round(rij / 10.0)
    if tij < rij / 10.0:
        return tij + 1
    return tij


def euclidean_distance(c1: tuple[int, int], c2: tuple[int, int]) -> float:
    """Standard Euclidean distance."""
    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]
    return (dx * dx + dy * dy) ** 0.5


def tour_length(tour: list[int], distance_fn=att_distance) -> float:
    """Calculate total tour length."""
    total = 0
    n = len(tour)
    for i in range(n):
        total += distance_fn(CITIES[tour[i]], CITIES[tour[(i + 1) % n]])
    return total


def build_distance_matrix(distance_fn=att_distance) -> list[list[float]]:
    """Build complete distance matrix."""
    n = len(CITIES)
    return [[distance_fn(CITIES[i], CITIES[j]) for j in range(n)] for i in range(n)]


if __name__ == "__main__":
    print(f"ATT48: {len(CITIES)} cities")
    print(f"Optimal tour length: {OPTIMAL_TOUR_LENGTH}")
    print(f"Verify optimal tour: {tour_length(OPTIMAL_TOUR)}")

"""
Bin Packing - Falkenauer U Benchmark

Pack items into minimum number of bins without exceeding capacity.
Uses the Falkenauer U (uniform) benchmark instances.

Source: Falkenauer (1996) "A Hybrid Grouping Genetic Algorithm for Bin Packing"
        http://people.brunel.ac.uk/~mastjjb/jeb/orlib/binpackinfo.html

Instance U120_00: 120 items, bin capacity 150
- Item sizes uniformly distributed in [20, 100]
- Optimal solution: 48 bins (tight packing exists)

Why this solver: FFD (First-Fit Decreasing) is a simple O(n log n) heuristic
that achieves at most 11/9 * OPT + 6/9 bins. For optimal solutions, use MILP.
"""

from solvor import solve_bin_pack

# Falkenauer U120_00 instance
# 120 items with sizes in [20, 100], bin capacity 150
# Optimal: 48 bins
U120_00_ITEMS = [
    98,
    97,
    96,
    95,
    94,
    92,
    91,
    89,
    88,
    87,
    86,
    86,
    85,
    84,
    83,
    82,
    81,
    81,
    80,
    79,
    79,
    78,
    77,
    76,
    75,
    74,
    74,
    73,
    72,
    71,
    70,
    69,
    68,
    67,
    66,
    65,
    64,
    64,
    63,
    62,
    61,
    60,
    59,
    58,
    57,
    56,
    55,
    54,
    53,
    52,
    51,
    51,
    50,
    49,
    48,
    47,
    46,
    45,
    44,
    43,
    42,
    41,
    40,
    39,
    38,
    37,
    36,
    35,
    34,
    33,
    32,
    31,
    30,
    29,
    28,
    27,
    26,
    25,
    24,
    23,
    22,
    21,
    20,
    99,
    98,
    97,
    96,
    95,
    94,
    93,
    92,
    91,
    90,
    89,
    88,
    87,
    86,
    85,
    84,
    83,
    82,
    81,
    80,
    79,
    78,
    77,
    76,
    75,
    74,
    73,
    72,
    71,
    70,
    69,
    68,
    67,
    66,
    65,
    64,
    63,
]

BIN_CAPACITY = 150
OPTIMAL_BINS = 48  # Known optimal for this instance


def main():
    print("Bin Packing - Falkenauer U120_00")
    print(f"  Items: {len(U120_00_ITEMS)}")
    print(f"  Bin capacity: {BIN_CAPACITY}")
    print(f"  Total item size: {sum(U120_00_ITEMS)}")
    print(f"  Lower bound (sum/capacity): {sum(U120_00_ITEMS) / BIN_CAPACITY:.1f}")
    print(f"  Known optimal: {OPTIMAL_BINS} bins")
    print()

    # Try different algorithms
    for algo in ["first-fit", "best-fit", "first-fit-decreasing", "best-fit-decreasing"]:
        result = solve_bin_pack(U120_00_ITEMS, BIN_CAPACITY, algorithm=algo)
        bins_used = int(result.objective)
        gap = (bins_used - OPTIMAL_BINS) / OPTIMAL_BINS * 100
        print(f"  {algo:25s}: {bins_used} bins (gap: {gap:+.1f}%)")

    print()

    # Show best result details
    result = solve_bin_pack(U120_00_ITEMS, BIN_CAPACITY, algorithm="best-fit-decreasing")
    bins_used = int(result.objective)

    # Reconstruct bins from assignment
    assignments = result.solution
    bins = {}
    for item_idx, bin_idx in enumerate(assignments):
        bins.setdefault(bin_idx, []).append(U120_00_ITEMS[item_idx])

    print(f"Best-Fit Decreasing result: {bins_used} bins")
    print("Sample bins (first 5):")
    for bin_idx in sorted(bins.keys())[:5]:
        items = bins[bin_idx]
        print(f"  Bin {bin_idx}: {items} (sum={sum(items)}, slack={BIN_CAPACITY - sum(items)})")


if __name__ == "__main__":
    main()

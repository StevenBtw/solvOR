"""
Bin Packing solver using heuristic algorithms.

Pack items into the minimum number of bins without exceeding capacity. NP-hard,
but heuristics get surprisingly close to optimal. Best-fit-decreasing (put big
items first, fill gaps with small ones) is within 11/9 OPT + 6/9 of optimal,
proven in 1973. Good enough for most real-world packing.

Applications: cutting stock, container loading, memory allocation, VM placement.

    from solvor import solve_bin_pack

    result = solve_bin_pack(item_sizes, bin_capacity)
    result = solve_bin_pack(sizes, capacity, algorithm='best-fit-decreasing')

Available algorithms:
- 'first-fit': Place item in first bin that fits
- 'best-fit': Place item in bin with least remaining space
- 'first-fit-decreasing': Sort items descending, then first-fit
- 'best-fit-decreasing': Sort items descending, then best-fit (default)

Decreasing variants typically produce better results. For optimal solutions
on small instances, use MILP formulation instead.
"""

from collections.abc import Sequence

from solvor.types import Result, Status
from solvor.utils import check_positive

__all__ = ["solve_bin_pack"]


def solve_bin_pack(
    item_sizes: Sequence[float],
    bin_capacity: float,
    *,
    algorithm: str = "best-fit-decreasing",
) -> Result:
    """Pack items into minimum bins using heuristic algorithms."""
    n = len(item_sizes)
    if n == 0:
        return Result((), 0.0, 0, 0, Status.OPTIMAL)

    check_positive(bin_capacity, name="bin_capacity")

    # Check for items that don't fit
    for i, size in enumerate(item_sizes):
        if size > bin_capacity:
            raise ValueError(f"Item {i} with size {size} exceeds bin capacity {bin_capacity}")
        if size < 0:
            raise ValueError(f"Item {i} has negative size {size}")

    # Parse algorithm
    algo = algorithm.lower().replace("_", "-")
    decreasing = algo.endswith("-decreasing")
    if decreasing:
        algo = algo.replace("-decreasing", "")

    if algo not in ("first-fit", "best-fit", "ff", "bf"):
        raise ValueError(f"Unknown algorithm: {algorithm}. Use 'first-fit', 'best-fit', or their -decreasing variants")

    use_best_fit = algo in ("best-fit", "bf")

    # Get item indices, optionally sorted by size descending
    if decreasing:
        indices = sorted(range(n), key=lambda i: item_sizes[i], reverse=True)
    else:
        indices = list(range(n))

    # Bins: list of (remaining_capacity, [item_indices])
    bins: list[tuple[float, list[int]]] = []
    assignments = [0] * n  # assignments[item] = bin_index

    for item_idx in indices:
        size = item_sizes[item_idx]

        if size == 0:
            # Zero-size items go in first bin (or create one)
            if not bins:
                bins.append((bin_capacity, []))
            bins[0][1].append(item_idx)
            assignments[item_idx] = 0
            continue

        best_bin = -1

        if use_best_fit:
            # Find bin with least remaining space that still fits
            best_remaining = float("inf")
            for b, (remaining, _) in enumerate(bins):
                if size <= remaining < best_remaining:
                    best_remaining = remaining
                    best_bin = b
        else:
            # First-fit: find first bin that fits
            for b, (remaining, _) in enumerate(bins):
                if size <= remaining:
                    best_bin = b
                    break

        if best_bin == -1:
            # Open new bin
            best_bin = len(bins)
            bins.append((bin_capacity, []))

        # Place item in bin
        remaining, items = bins[best_bin]
        items.append(item_idx)
        bins[best_bin] = (remaining - size, items)
        assignments[item_idx] = best_bin

    # Build result: tuple of bin assignments (item i -> bin index)
    num_bins = len(bins)

    # Status: we can't prove optimality with heuristics
    status = Status.FEASIBLE if num_bins > 1 else Status.OPTIMAL

    return Result(tuple(assignments), float(num_bins), 0, n, status)


def lower_bound(item_sizes: Sequence[float], bin_capacity: float) -> int:
    """Compute lower bound on number of bins needed."""
    total = sum(item_sizes)
    return max(1, int(-(-total // bin_capacity)))  # Ceiling division

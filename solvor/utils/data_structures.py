"""
Data structures for optimization algorithms.

Efficient data structures used internally by solvers and available for
custom implementations.
"""

__all__ = ["FenwickTree", "UnionFind"]


class FenwickTree:
    """
    Fenwick Tree (Binary Indexed Tree) for prefix sums with point updates.

    Supports O(log n) prefix sum queries and point updates. Useful for
    cumulative frequency tables, range sum queries, and inversion counting.

        ft = FenwickTree([1, 2, 3, 4, 5])
        ft.prefix(2)      # 6 (sum of indices 0, 1, 2)
        ft.update(1, 10)  # add 10 to index 1
        ft.prefix(2)      # 16
        ft.range_sum(1, 3)  # sum of indices 1, 2, 3
    """

    __slots__ = ("_tree", "_n")

    def __init__(self, values: list[float] | int) -> None:
        """Initialize from values list or size (zeros)."""
        if isinstance(values, int):
            self._n = values
            self._tree = [0.0] * values
        else:
            self._n = len(values)
            self._tree = list(values)
            for i in range(self._n):
                j = i | (i + 1)
                if j < self._n:
                    self._tree[j] += self._tree[i]

    def update(self, i: int, delta: float) -> None:
        """Add delta to element at index i."""
        while i < self._n:
            self._tree[i] += delta
            i |= i + 1

    def prefix(self, i: int) -> float:
        """Return sum of elements from index 0 to i (inclusive)."""
        total = 0.0
        while i >= 0:
            total += self._tree[i]
            i = (i & (i + 1)) - 1
        return total

    def range_sum(self, left: int, right: int) -> float:
        """Return sum of elements from left to right (inclusive)."""
        result = self.prefix(right)
        if left > 0:
            result -= self.prefix(left - 1)
        return result

    def __len__(self) -> int:
        """Return number of elements."""
        return self._n


class UnionFind:
    """
    Union-Find (Disjoint Set Union) data structure.

    Efficiently tracks connected components with near O(1) operations via
    path compression and union by rank.

        uf = UnionFind(10)
        uf.union(0, 1)
        uf.union(1, 2)
        uf.connected(0, 2)  # True
        uf.component_count  # 8 (started with 10, merged 3 into 1)
    """

    __slots__ = ("_parent", "_rank", "_count")

    def __init__(self, n: int) -> None:
        """Initialize n elements, each in its own component."""
        self._parent = list(range(n))
        self._rank = [0] * n
        self._count = n

    def find(self, x: int) -> int:
        """Find root of x with path compression."""
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, x: int, y: int) -> bool:
        """Merge components containing x and y. Returns True if merged."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False

        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1

        self._count -= 1
        return True

    def connected(self, x: int, y: int) -> bool:
        """Check if x and y are in the same component."""
        return self.find(x) == self.find(y)

    @property
    def component_count(self) -> int:
        """Number of disjoint components."""
        return self._count

    def component_sizes(self) -> list[int]:
        """Return list of component sizes."""
        size_map: dict[int, int] = {}
        for i in range(len(self._parent)):
            root = self.find(i)
            size_map[root] = size_map.get(root, 0) + 1
        return list(size_map.values())

    def get_components(self) -> list[set[int]]:
        """Return list of sets, each containing elements of one component."""
        comp_map: dict[int, set[int]] = {}
        for i in range(len(self._parent)):
            root = self.find(i)
            if root not in comp_map:
                comp_map[root] = set()
            comp_map[root].add(i)
        return list(comp_map.values())

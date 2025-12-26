"""Tests for data structure utilities."""

from solvor.utils import FenwickTree, UnionFind


class TestFenwickTree:
    def test_build_and_prefix(self):
        ft = FenwickTree([1, 2, 3, 4, 5])
        assert ft.prefix(0) == 1
        assert ft.prefix(1) == 3
        assert ft.prefix(2) == 6
        assert ft.prefix(4) == 15

    def test_update(self):
        ft = FenwickTree([1, 2, 3, 4, 5])
        ft.update(2, 10)
        assert ft.prefix(2) == 16
        assert ft.prefix(4) == 25

    def test_empty(self):
        ft = FenwickTree([])
        assert len(ft) == 0

    def test_single(self):
        ft = FenwickTree([5])
        assert ft.prefix(0) == 5

    def test_init_from_size(self):
        ft = FenwickTree(5)
        assert len(ft) == 5
        assert ft.prefix(4) == 0
        ft.update(2, 10)
        assert ft.prefix(4) == 10

    def test_range_sum(self):
        ft = FenwickTree([1, 2, 3, 4, 5])
        assert ft.range_sum(0, 4) == 15
        assert ft.range_sum(1, 3) == 9  # 2 + 3 + 4
        assert ft.range_sum(2, 2) == 3
        assert ft.range_sum(0, 0) == 1

    def test_len(self):
        ft = FenwickTree([1, 2, 3])
        assert len(ft) == 3


class TestUnionFind:
    def test_initial_state(self):
        uf = UnionFind(5)
        assert uf.component_count == 5
        for i in range(5):
            assert uf.find(i) == i

    def test_union_basic(self):
        uf = UnionFind(5)
        assert uf.union(0, 1) is True
        assert uf.component_count == 4
        assert uf.connected(0, 1) is True

    def test_union_already_connected(self):
        uf = UnionFind(5)
        uf.union(0, 1)
        assert uf.union(0, 1) is False
        assert uf.component_count == 4

    def test_connected(self):
        uf = UnionFind(5)
        assert uf.connected(0, 1) is False
        uf.union(0, 1)
        assert uf.connected(0, 1) is True
        assert uf.connected(0, 2) is False

    def test_transitive_connectivity(self):
        uf = UnionFind(5)
        uf.union(0, 1)
        uf.union(1, 2)
        assert uf.connected(0, 2) is True
        assert uf.component_count == 3

    def test_component_sizes(self):
        uf = UnionFind(6)
        uf.union(0, 1)
        uf.union(1, 2)
        uf.union(3, 4)
        # Components: {0,1,2}, {3,4}, {5}
        sizes = sorted(uf.component_sizes())
        assert sizes == [1, 2, 3]

    def test_get_components(self):
        uf = UnionFind(5)
        uf.union(0, 1)
        uf.union(2, 3)
        components = uf.get_components()
        assert len(components) == 3
        # Check each component
        component_sets = [frozenset(c) for c in components]
        assert frozenset({0, 1}) in component_sets
        assert frozenset({2, 3}) in component_sets
        assert frozenset({4}) in component_sets

    def test_single_element(self):
        uf = UnionFind(1)
        assert uf.component_count == 1
        assert uf.find(0) == 0
        assert uf.component_sizes() == [1]

    def test_all_merged(self):
        uf = UnionFind(4)
        uf.union(0, 1)
        uf.union(1, 2)
        uf.union(2, 3)
        assert uf.component_count == 1
        assert uf.component_sizes() == [4]
        for i in range(4):
            for j in range(4):
                assert uf.connected(i, j) is True

    def test_path_compression(self):
        # After find operations, paths should be compressed
        uf = UnionFind(5)
        uf.union(0, 1)
        uf.union(1, 2)
        uf.union(2, 3)
        uf.union(3, 4)
        # Find on deepest node should compress path
        root = uf.find(4)
        # All nodes should now point directly to root
        for i in range(5):
            assert uf.find(i) == root

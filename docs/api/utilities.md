# Utilities

Helper functions and data structures used internally by solvers, also available for custom implementations.

## Overview

| Category | Contents |
|----------|----------|
| Data Structures | `FenwickTree`, `UnionFind` - efficient structures for algorithms |
| Validation | Input checking functions with clear error messages |
| Helpers | Debugging, evaluation tracking, progress utilities |

## Quick Examples

```python
from solvor.utils import FenwickTree, UnionFind, debug

# Fenwick tree for prefix sums
ft = FenwickTree([1, 2, 3, 4, 5])
ft.prefix(2)       # 6 (sum of indices 0, 1, 2)
ft.update(1, 10)   # add 10 to index 1
ft.range_sum(1, 3) # sum of indices 1, 2, 3

# Union-Find for connected components
uf = UnionFind(10)
uf.union(0, 1)
uf.union(1, 2)
uf.connected(0, 2)  # True
uf.component_count  # 8

# Debug output (only prints when DEBUG=1)
debug("solving iteration", i)
```

## Validation Functions

Used internally by solvers to validate inputs before solving. Provides clear error messages for common mistakes.

```python
from solvor.utils import check_matrix_dims, check_positive, check_bounds

# Validate LP dimensions
check_matrix_dims(c, A, b)  # Raises if dimensions mismatch

# Validate parameter ranges
check_positive(n_nodes, name="n_nodes")
check_bounds(bounds)  # Validates (low, high) pairs
```

| Function | Purpose |
|----------|---------|
| `check_matrix_dims(c, A, b)` | LP/MILP dimension consistency |
| `check_positive(val, name)` | val > 0 |
| `check_non_negative(val, name)` | val >= 0 |
| `check_in_range(val, lo, hi, name)` | lo <= val <= hi |
| `check_bounds(bounds)` | Valid (low, high) pairs |
| `check_edge_nodes(edges, n_nodes)` | Edge endpoints valid |
| `check_sequence_lengths(seqs, names)` | Parallel sequences same length |
| `warn_large_coefficients(A)` | Warns if max > 1e10 |

## Data Structures

::: solvor.utils.data_structures

## Helpers

::: solvor.utils.helpers

## Validation Reference

::: solvor.utils.validate

//! PyO3 bindings for minimum spanning tree algorithms.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::algorithms::kruskal as k;
use crate::types::Status;

/// Kruskal's minimum spanning tree algorithm.
///
/// Args:
///     n_nodes: Number of nodes (0 to n_nodes-1)
///     edges: List of (from, to, weight) tuples
///
/// Returns:
///     Dict with 'mst_edges', 'total_weight', 'iterations', 'is_connected'
#[pyfunction]
#[pyo3(signature = (n_nodes, edges))]
pub fn kruskal(py: Python<'_>, n_nodes: usize, edges: Vec<(usize, usize, f64)>) -> PyResult<Py<PyDict>> {
    // Run algorithm (release GIL for computation)
    let result = py.detach(|| k::kruskal(n_nodes, &edges));

    // Convert result to Python dict
    let dict = PyDict::new(py);

    // Convert MST edges to list of tuples
    let py_edges = PyList::new(py, result.mst_edges.iter().map(|&(u, v, w)| (u, v, w)))?;
    dict.set_item("mst_edges", py_edges)?;

    dict.set_item("total_weight", result.total_weight)?;
    dict.set_item("iterations", result.iterations)?;
    dict.set_item("is_connected", result.is_connected)?;

    // Status based on connectivity
    let status = if result.is_connected {
        Status::Optimal
    } else {
        Status::Feasible // Partial MST (forest)
    };
    dict.set_item("status", status.as_i32())?;

    Ok(dict.into())
}

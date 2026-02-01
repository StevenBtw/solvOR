//! PyO3 bindings for centrality algorithms.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::algorithms::pagerank as pr;
use crate::types::Status;

/// PageRank algorithm (edge-list version).
///
/// Args:
///     n_nodes: Number of nodes (0 to n_nodes-1)
///     edges: List of (from, to) tuples
///     damping: Damping factor (default 0.85)
///     max_iter: Maximum iterations (default 100)
///     tol: Convergence tolerance (default 1e-6)
///
/// Returns:
///     Dict with 'scores', 'iterations', 'converged'
#[pyfunction]
#[pyo3(signature = (n_nodes, edges, damping=0.85, max_iter=100, tol=1e-6))]
pub fn pagerank(
    py: Python<'_>,
    n_nodes: usize,
    edges: Vec<(usize, usize)>,
    damping: f64,
    max_iter: usize,
    tol: f64,
) -> PyResult<Py<PyDict>> {
    let result = py.detach(|| pr::pagerank(n_nodes, &edges, damping, max_iter, tol));

    let dict = PyDict::new(py);

    let py_scores = PyList::new(py, result.scores.iter())?;
    dict.set_item("scores", py_scores)?;

    dict.set_item("iterations", result.iterations)?;
    dict.set_item("converged", result.converged)?;

    let status = if result.converged {
        Status::Optimal
    } else {
        Status::MaxIter
    };
    dict.set_item("status", status.as_i32())?;

    Ok(dict.into())
}

//! PyO3 bindings for centrality algorithms.

use pyo3::exceptions::PyValueError;
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
///     Dict with 'scores', 'iterations', 'converged', 'residual', 'status'
#[pyfunction]
#[pyo3(signature = (n_nodes, edges, damping=0.85, max_iter=100, tol=1e-6))]
pub fn pagerank(
    py: Python<'_>,
    n_nodes: usize,
    edges: Vec<(i64, i64)>,
    damping: f64,
    max_iter: usize,
    tol: f64,
) -> PyResult<Py<PyDict>> {
    if !(0.0..1.0).contains(&damping) {
        return Err(PyValueError::new_err(format!(
            "damping {damping} must be in [0, 1)"
        )));
    }
    if max_iter == 0 {
        return Err(PyValueError::new_err("max_iter must be positive"));
    }
    if !(tol > 0.0 && tol.is_finite()) {
        return Err(PyValueError::new_err("tol must be positive and finite"));
    }

    // Negative endpoints are ignored, like endpoints >= n_nodes (and like the Python backend)
    let edges: Vec<(usize, usize)> = edges
        .into_iter()
        .filter(|&(u, v)| u >= 0 && v >= 0)
        .map(|(u, v)| (u as usize, v as usize))
        .collect();
    let result = py.detach(|| pr::pagerank(n_nodes, &edges, damping, max_iter, tol));

    let dict = PyDict::new(py);

    let py_scores = PyList::new(py, result.scores.iter())?;
    dict.set_item("scores", py_scores)?;

    dict.set_item("iterations", result.iterations)?;
    dict.set_item("converged", result.converged)?;
    dict.set_item("residual", result.residual)?;

    let status = if result.converged {
        Status::Optimal
    } else {
        Status::MaxIter
    };
    dict.set_item("status", status.as_i32())?;

    Ok(dict.into())
}

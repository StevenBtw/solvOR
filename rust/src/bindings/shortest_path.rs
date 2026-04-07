//! PyO3 bindings for shortest path algorithms.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::algorithms::bellman_ford as bf;
use crate::algorithms::floyd_warshall as fw;
use crate::callback::ProgressCallback;
use crate::types::Status;

/// Floyd-Warshall all-pairs shortest paths.
///
/// Args:
///     n_nodes: Number of nodes (0 to n_nodes-1)
///     edges: List of (from, to, weight) tuples
///     on_progress: Optional progress callback
///     progress_interval: Call callback every N outer iterations
///
/// Returns:
///     Dict with 'distances', 'predecessors', 'has_negative_cycle',
///     'iterations', 'status'
#[pyfunction]
#[pyo3(signature = (n_nodes, edges, on_progress=None, progress_interval=0))]
pub fn floyd_warshall(
    py: Python<'_>,
    n_nodes: usize,
    edges: Vec<(usize, usize, f64)>,
    on_progress: Option<Py<PyAny>>,
    progress_interval: usize,
) -> PyResult<Py<PyDict>> {
    if n_nodes == 0 {
        return Err(PyValueError::new_err("n_nodes must be positive"));
    }

    // Convert edges
    let edges: Vec<fw::Edge> = edges;

    // Set up callback
    let mut callback = ProgressCallback::new(on_progress, progress_interval.max(1));

    // Run algorithm (release GIL for computation)
    let result = py.detach(|| fw::floyd_warshall(n_nodes, &edges, &mut callback))?;

    // Convert result to Python dict
    let dict = PyDict::new(py);

    // Convert distance matrix to nested list
    let dist_rows = result
        .distances
        .iter()
        .map(|row| PyList::new(py, row.iter().copied()))
        .collect::<PyResult<Vec<_>>>()?;
    let py_distances = PyList::new(py, dist_rows)?;
    dict.set_item("distances", py_distances)?;

    // Convert predecessor matrix
    let pred_rows = result
        .predecessors
        .iter()
        .map(|row| {
            PyList::new(
                py,
                row.iter().map(|&p| p.map(|x| x as i64).unwrap_or(-1)),
            )
        })
        .collect::<PyResult<Vec<_>>>()?;
    let py_predecessors = PyList::new(py, pred_rows)?;
    dict.set_item("predecessors", py_predecessors)?;

    dict.set_item("has_negative_cycle", result.has_negative_cycle)?;
    dict.set_item("iterations", result.iterations)?;

    // Status based on result
    let status = if result.has_negative_cycle {
        Status::Infeasible
    } else {
        Status::Optimal
    };
    dict.set_item("status", status.as_i32())?;

    Ok(dict.into())
}

/// Bellman-Ford single-source shortest paths with negative edge support.
///
/// Args:
///     n_nodes: Number of nodes (0 to n_nodes-1)
///     edges: List of (from, to, weight) tuples
///     source: Starting node
///
/// Returns:
///     Dict with 'distances', 'predecessors', 'has_negative_cycle', 'iterations'
#[pyfunction]
#[pyo3(signature = (n_nodes, edges, source))]
pub fn bellman_ford(
    py: Python<'_>,
    n_nodes: usize,
    edges: Vec<(usize, usize, f64)>,
    source: usize,
) -> PyResult<Py<PyDict>> {
    if source >= n_nodes {
        return Err(PyValueError::new_err(format!(
            "source {source} out of range for {n_nodes} nodes"
        )));
    }

    // Run algorithm (release GIL for computation)
    let result = py.detach(|| bf::bellman_ford(n_nodes, &edges, source));

    // Convert result to Python dict
    let dict = PyDict::new(py);

    // Convert distances to list
    let py_distances = PyList::new(
        py,
        result.distances.iter().map(|&d| {
            if d.is_infinite() {
                f64::INFINITY
            } else {
                d
            }
        }),
    )?;
    dict.set_item("distances", py_distances)?;

    // Convert predecessors to list
    let py_predecessors = PyList::new(py, result.predecessors.iter())?;
    dict.set_item("predecessors", py_predecessors)?;

    dict.set_item("has_negative_cycle", result.has_negative_cycle)?;
    dict.set_item("iterations", result.iterations)?;

    // Status based on result
    let status = if result.has_negative_cycle {
        Status::Infeasible
    } else {
        Status::Optimal
    };
    dict.set_item("status", status.as_i32())?;

    Ok(dict.into())
}

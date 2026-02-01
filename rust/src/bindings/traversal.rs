//! PyO3 bindings for graph traversal algorithms.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::algorithms::bfs as bfs_algo;
use crate::algorithms::dijkstra as dij;
use crate::types::Status;

/// Dijkstra's shortest path algorithm (edge-list version).
///
/// Args:
///     n_nodes: Number of nodes (0 to n_nodes-1)
///     edges: List of (from, to, weight) tuples
///     source: Starting node
///     target: Optional target node for early termination
///
/// Returns:
///     Dict with 'distances', 'predecessors', 'path', 'target_distance', 'iterations'
#[pyfunction]
#[pyo3(signature = (n_nodes, edges, source, target=None))]
pub fn dijkstra(
    py: Python<'_>,
    n_nodes: usize,
    edges: Vec<(usize, usize, f64)>,
    source: usize,
    target: Option<usize>,
) -> PyResult<Py<PyDict>> {
    let result = py.detach(|| dij::dijkstra(n_nodes, &edges, source, target));

    let dict = PyDict::new(py);

    // Convert distances
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

    // Convert predecessors
    let py_predecessors = PyList::new(py, result.predecessors.iter())?;
    dict.set_item("predecessors", py_predecessors)?;

    // Path
    let py_path = PyList::new(py, result.path.iter())?;
    dict.set_item("path", py_path)?;

    dict.set_item("target_reached", result.target_reached)?;
    dict.set_item("target_distance", result.target_distance)?;
    dict.set_item("iterations", result.iterations)?;

    let status = if result.target_reached || target.is_none() {
        Status::Optimal
    } else {
        Status::Infeasible
    };
    dict.set_item("status", status.as_i32())?;

    Ok(dict.into())
}

/// Breadth-first search (edge-list version).
///
/// Args:
///     n_nodes: Number of nodes (0 to n_nodes-1)
///     edges: List of (from, to) tuples
///     source: Starting node
///     target: Optional target node
///
/// Returns:
///     Dict with 'path', 'visited_order', 'target_reached', 'iterations'
#[pyfunction]
#[pyo3(signature = (n_nodes, edges, source, target=None))]
pub fn bfs(
    py: Python<'_>,
    n_nodes: usize,
    edges: Vec<(usize, usize)>,
    source: usize,
    target: Option<usize>,
) -> PyResult<Py<PyDict>> {
    let result = py.detach(|| bfs_algo::bfs(n_nodes, &edges, source, target));

    let dict = PyDict::new(py);

    let py_path = PyList::new(py, result.path.iter())?;
    dict.set_item("path", py_path)?;

    let py_visited = PyList::new(py, result.visited_order.iter())?;
    dict.set_item("visited_order", py_visited)?;

    dict.set_item("target_reached", result.target_reached)?;
    dict.set_item("iterations", result.iterations)?;

    let status = if result.target_reached || target.is_none() {
        Status::Optimal
    } else {
        Status::Infeasible
    };
    dict.set_item("status", status.as_i32())?;

    Ok(dict.into())
}

/// Depth-first search (edge-list version).
#[pyfunction]
#[pyo3(signature = (n_nodes, edges, source, target=None))]
pub fn dfs(
    py: Python<'_>,
    n_nodes: usize,
    edges: Vec<(usize, usize)>,
    source: usize,
    target: Option<usize>,
) -> PyResult<Py<PyDict>> {
    let result = py.detach(|| bfs_algo::dfs(n_nodes, &edges, source, target));

    let dict = PyDict::new(py);

    let py_path = PyList::new(py, result.path.iter())?;
    dict.set_item("path", py_path)?;

    let py_visited = PyList::new(py, result.visited_order.iter())?;
    dict.set_item("visited_order", py_visited)?;

    dict.set_item("target_reached", result.target_reached)?;
    dict.set_item("iterations", result.iterations)?;

    let status = if result.target_reached || target.is_none() {
        Status::Optimal
    } else {
        Status::Infeasible
    };
    dict.set_item("status", status.as_i32())?;

    Ok(dict.into())
}

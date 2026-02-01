//! PyO3 bindings for component algorithms.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::algorithms::scc;
use crate::types::Status;

/// Strongly connected components using Tarjan's algorithm.
///
/// Args:
///     n_nodes: Number of nodes (0 to n_nodes-1)
///     edges: List of (from, to) tuples
///
/// Returns:
///     Dict with 'components', 'n_components'
#[pyfunction]
#[pyo3(signature = (n_nodes, edges))]
pub fn strongly_connected_components(
    py: Python<'_>,
    n_nodes: usize,
    edges: Vec<(usize, usize)>,
) -> PyResult<Py<PyDict>> {
    let result = py.detach(|| scc::strongly_connected_components(n_nodes, &edges));

    let dict = PyDict::new(py);

    // Convert components to list of lists
    let py_components = PyList::new(
        py,
        result
            .components
            .iter()
            .map(|comp| PyList::new(py, comp.iter()).unwrap()),
    )?;
    dict.set_item("components", py_components)?;
    dict.set_item("n_components", result.n_components)?;
    dict.set_item("status", Status::Optimal.as_i32())?;

    Ok(dict.into())
}

/// Topological sort using Kahn's algorithm.
///
/// Args:
///     n_nodes: Number of nodes (0 to n_nodes-1)
///     edges: List of (from, to) tuples
///
/// Returns:
///     Dict with 'order' (None if cycle), 'is_acyclic', 'iterations'
#[pyfunction]
#[pyo3(signature = (n_nodes, edges))]
pub fn topological_sort(
    py: Python<'_>,
    n_nodes: usize,
    edges: Vec<(usize, usize)>,
) -> PyResult<Py<PyDict>> {
    let result = py.detach(|| scc::topological_sort(n_nodes, &edges));

    let dict = PyDict::new(py);

    if let Some(order) = result.order {
        let py_order = PyList::new(py, order.iter())?;
        dict.set_item("order", py_order)?;
    } else {
        dict.set_item("order", py.None())?;
    }

    dict.set_item("is_acyclic", result.is_acyclic)?;
    dict.set_item("iterations", result.iterations)?;

    let status = if result.is_acyclic {
        Status::Optimal
    } else {
        Status::Infeasible
    };
    dict.set_item("status", status.as_i32())?;

    Ok(dict.into())
}

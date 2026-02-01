//! solvOR Rust acceleration module.
//!
//! Provides high-performance implementations of graph algorithms
//! that can be used as drop-in replacements for the Python versions.

use pyo3::prelude::*;

mod algorithms;
mod bindings;
mod callback;
mod types;

/// solvOR Rust acceleration module.
///
/// This module provides optimized Rust implementations of solvOR's
/// graph algorithms. Functions have the same signatures as their
/// Python counterparts for seamless integration.
#[pymodule]
fn _solvor_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Shortest paths
    m.add_function(wrap_pyfunction!(bindings::floyd_warshall, m)?)?;
    m.add_function(wrap_pyfunction!(bindings::bellman_ford, m)?)?;
    m.add_function(wrap_pyfunction!(bindings::dijkstra, m)?)?;

    // Graph traversal
    m.add_function(wrap_pyfunction!(bindings::bfs, m)?)?;
    m.add_function(wrap_pyfunction!(bindings::dfs, m)?)?;

    // Minimum spanning tree
    m.add_function(wrap_pyfunction!(bindings::kruskal, m)?)?;

    // Centrality
    m.add_function(wrap_pyfunction!(bindings::pagerank, m)?)?;

    // Components
    m.add_function(wrap_pyfunction!(bindings::strongly_connected_components, m)?)?;
    m.add_function(wrap_pyfunction!(bindings::topological_sort, m)?)?;

    // Module metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}

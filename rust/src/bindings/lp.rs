//! PyO3 binding for the LP kernel.
//!
//! `BoundedSimplex` has the same constructor, methods and attributes as
//! `solvor.lp_engine.BoundedSimplex` and returns bit-identical results, so
//! `solvor.lp_engine.WarmLP` can use either class.

use pyo3::exceptions::{PyIndexError, PyOverflowError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyType};

use crate::algorithms::simplex as sx;
use crate::types::Status;

static STATUS: PyOnceLock<Py<PyType>> = PyOnceLock::new();

/// The Python `solvor.types.Status` member for `status`.
fn py_status<'py>(py: Python<'py>, status: Status) -> PyResult<Bound<'py, PyAny>> {
    STATUS
        .import(py, "solvor.types", "Status")?
        .call1((status.as_i32(),))
}

/// Python's `max_iter` as i64; ints beyond its range saturate (no solve gets that far).
fn iteration_limit(max_iter: &Bound<'_, PyAny>) -> PyResult<i64> {
    match max_iter.extract::<i64>() {
        Err(err) if err.is_instance_of::<PyOverflowError>(max_iter.py()) => {
            Ok(if max_iter.gt(0)? { i64::MAX } else { i64::MIN })
        }
        result => result,
    }
}

/// Pickle state: scalars, bounds, flags and indices, then the tableau.
type State = (
    (usize, usize, f64, usize, usize, u64, bool, bool),
    (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>),
    (Vec<bool>, Vec<bool>, Vec<usize>, Vec<usize>),
    Vec<Vec<f64>>,
);

/// A `{column: coefficient}` dict as pairs, in the dict's own order (sums depend on it).
fn row_items(coefs: &Bound<'_, PyDict>, width: usize) -> PyResult<Vec<(usize, f64)>> {
    let mut items = Vec::with_capacity(coefs.len());
    for (key, value) in coefs.iter() {
        let j: usize = key.extract()?;
        if j >= width {
            return Err(PyIndexError::new_err(format!(
                "column {j} out of range for {width} columns"
            )));
        }
        items.push((j, value.extract()?));
    }
    Ok(items)
}

/// Dense bounded-variable simplex tableau (Rust port of solvor.lp_engine.BoundedSimplex).
#[pyclass(name = "BoundedSimplex", module = "solvor._solvor_rust")]
pub struct PyBoundedSimplex {
    inner: sx::BoundedSimplex,
}

#[pymethods]
impl PyBoundedSimplex {
    #[new]
    fn new(
        rows: Vec<Bound<'_, PyDict>>,
        rhs: Vec<f64>,
        is_eq: Vec<bool>,
        upper: Vec<f64>,
        eps: f64,
    ) -> PyResult<Self> {
        if rhs.len() != rows.len() || is_eq.len() != rows.len() {
            return Err(PyValueError::new_err(format!(
                "rows, rhs and is_eq differ in length: {}, {}, {}",
                rows.len(),
                rhs.len(),
                is_eq.len()
            )));
        }
        let rows = rows
            .iter()
            .map(|r| row_items(r, upper.len()))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            inner: sx::BoundedSimplex::new(&rows, &rhs, &is_eq, &upper, eps),
        })
    }

    /// Pivots since construction (the caller may reset it).
    #[getter]
    fn pivots(&self) -> u64 {
        self.inner.pivots
    }

    #[setter]
    fn set_pivots(&mut self, value: u64) {
        self.inner.pivots = value;
    }

    /// False until phase 1 has found a feasible basis.
    #[getter]
    fn phase1_done(&self) -> bool {
        self.inner.phase1_done
    }

    /// Whether `primal` also enters columns whose tiny reduced cost hides a large gain.
    #[getter]
    fn check_gains(&self) -> bool {
        self.inner.check_gains
    }

    #[setter]
    fn set_check_gains(&mut self, value: bool) {
        self.inner.check_gains = value;
    }

    /// Number of rows.
    #[getter]
    fn m(&self) -> usize {
        self.inner.m
    }

    /// Number of columns (structural, slack, artificial and added-row slack).
    #[getter]
    fn width(&self) -> usize {
        self.inner.width
    }

    /// Phase 1 (only if artificials exist), then phase 2 on `cost`. Returns (status, iterations).
    fn solve<'py>(
        &mut self,
        py: Python<'py>,
        cost: Vec<f64>,
        max_iter: &Bound<'_, PyAny>,
    ) -> PyResult<(Bound<'py, PyAny>, i64)> {
        let limit = iteration_limit(max_iter)?;
        let inner = &mut self.inner;
        let (status, iters) = py.detach(|| inner.solve(&cost, limit));
        Ok((py_status(py, status)?, iters))
    }

    /// Value of every column in y (not in the flipped representation).
    fn column_values(&self) -> Vec<f64> {
        self.inner.column_values()
    }

    /// Objective row = reduced costs of `cost` for the current basis (missing entries are 0).
    fn set_objective(&mut self, cost: Vec<f64>) {
        self.inner.set_objective(&cost);
    }

    /// Primal simplex from a primal feasible basis. Returns (status, iterations).
    fn primal<'py>(
        &mut self,
        py: Python<'py>,
        max_iter: &Bound<'_, PyAny>,
    ) -> PyResult<(Bound<'py, PyAny>, i64)> {
        let limit = iteration_limit(max_iter)?;
        let inner = &mut self.inner;
        let (status, iters) = py.detach(|| inner.primal(limit));
        Ok((py_status(py, status)?, iters))
    }

    /// Move boxed nonbasic columns to the bound their reduced cost prefers (False: cannot).
    fn make_dual_feasible(&mut self) -> bool {
        self.inner.make_dual_feasible()
    }

    /// Dual simplex from a dual feasible basis. Returns (status, iterations).
    fn dual<'py>(
        &mut self,
        py: Python<'py>,
        max_iter: &Bound<'_, PyAny>,
    ) -> PyResult<(Bound<'py, PyAny>, i64)> {
        let limit = iteration_limit(max_iter)?;
        let inner = &mut self.inner;
        let (status, iters) = py.detach(|| inner.dual(limit));
        Ok((py_status(py, status)?, iters))
    }

    /// Change column j's bounds in place (lo must be finite).
    fn set_bounds(&mut self, j: usize, lo: f64, hi: f64) -> PyResult<()> {
        if j >= self.inner.width {
            return Err(PyIndexError::new_err(format!(
                "column {j} out of range for {} columns",
                self.inner.width
            )));
        }
        self.inner.set_bounds(j, lo, hi);
        Ok(())
    }

    /// Arguments for `__new__` when unpickling: an empty tableau that `__setstate__` fills.
    fn __getnewargs__(&self) -> (Vec<f64>, Vec<f64>, Vec<bool>, Vec<f64>, f64) {
        (vec![], vec![], vec![], vec![], 0.0)
    }

    fn __getstate__(&self) -> State {
        let lp = &self.inner;
        (
            (
                lp.m,
                lp.width,
                lp.eps,
                lp.art_start,
                lp.art_end,
                lp.pivots,
                lp.phase1_done,
                lp.check_gains,
            ),
            (
                lp.art_tol.clone(),
                lp.lo.clone(),
                lp.hi.clone(),
                lp.span.clone(),
            ),
            (
                lp.flipped.clone(),
                lp.is_basic.clone(),
                lp.basis.clone(),
                lp.row_of.clone(),
            ),
            lp.t.clone(),
        )
    }

    fn __setstate__(&mut self, state: State) -> PyResult<()> {
        let (
            (m, width, eps, art_start, art_end, pivots, phase1_done, check_gains),
            (art_tol, lo, hi, span),
            (flipped, is_basic, basis, row_of),
            t,
        ) = state;
        let columns = [
            lo.len(),
            hi.len(),
            span.len(),
            flipped.len(),
            is_basic.len(),
            row_of.len(),
        ];
        if columns.iter().any(|&len| len != width)
            || basis.len() != m
            || t.len() != m + 1
            || t.iter().any(|row| row.len() != width + 1)
            || !(art_start <= art_end && art_end <= width && art_tol.len() == art_end - art_start)
            || basis.iter().any(|&j| j >= width)
            || row_of.iter().any(|&r| r != sx::NONE && r >= m)
        {
            return Err(PyValueError::new_err("inconsistent BoundedSimplex state"));
        }
        self.inner = sx::BoundedSimplex {
            m,
            width,
            eps,
            art_start,
            art_end,
            pivots,
            art_tol,
            phase1_done,
            check_gains,
            lo,
            hi,
            span,
            flipped,
            is_basic,
            basis,
            row_of,
            t,
        };
        Ok(())
    }

    /// Append `coefs . y <= rhs` (or `=` if is_eq) with a new basic slack.
    fn add_row(&mut self, coefs: Bound<'_, PyDict>, rhs: f64, is_eq: bool) -> PyResult<()> {
        let items = row_items(&coefs, self.inner.width)?;
        self.inner.add_row(&items, rhs, is_eq);
        Ok(())
    }
}

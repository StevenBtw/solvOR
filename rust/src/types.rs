//! Shared types that map to Python's solvor.types module.

use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Status enum matching Python's Status(IntEnum).
/// Values must match exactly for correct marshalling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Status {
    Optimal = 1,
    Feasible = 2,
    Infeasible = 3,
    Unbounded = 4,
    MaxIter = 5,
}

impl Status {
    pub fn as_i32(self) -> i32 {
        self as i32
    }
}

/// Progress data for callbacks, matching Python's Progress dataclass.
#[derive(Debug, Clone)]
pub struct Progress {
    pub iteration: usize,
    pub objective: f64,
    pub best: Option<f64>,
    pub evaluations: usize,
}

impl Progress {
    pub fn new(iteration: usize, objective: f64) -> Self {
        Self {
            iteration,
            objective,
            best: None,
            evaluations: 0,
        }
    }

    pub fn with_best(mut self, best: f64) -> Self {
        self.best = Some(best);
        self
    }

    pub fn with_evaluations(mut self, evaluations: usize) -> Self {
        self.evaluations = evaluations;
        self
    }

    /// Convert to Python dict for callback.
    pub fn to_pydict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("iteration", self.iteration)?;
        dict.set_item("objective", self.objective)?;
        dict.set_item("best", self.best)?;
        dict.set_item("evaluations", self.evaluations)?;
        Ok(dict)
    }
}


//! Progress callback handling for Rust -> Python calls.

use pyo3::prelude::*;

use crate::types::Progress;

/// Wrapper for optional Python progress callbacks.
///
/// Handles GIL acquisition and result conversion.
pub struct ProgressCallback {
    callback: Option<Py<PyAny>>,
    interval: usize,
    last_call: usize,
}

impl ProgressCallback {
    /// Create a new callback wrapper.
    ///
    /// - `callback`: Optional Python callable that takes a Progress dict
    /// - `interval`: Only call every N iterations (0 = every time)
    pub fn new(callback: Option<Py<PyAny>>, interval: usize) -> Self {
        Self {
            callback,
            interval,
            last_call: 0,
        }
    }

    /// Create a no-op callback (never calls Python).
    pub fn none() -> Self {
        Self {
            callback: None,
            interval: 0,
            last_call: 0,
        }
    }

    /// Check if callback should be called at this iteration.
    pub fn should_call(&self, iteration: usize) -> bool {
        if self.callback.is_none() {
            return false;
        }
        if self.interval == 0 {
            return true;
        }
        iteration >= self.last_call + self.interval
    }

    /// Call the Python callback with progress data.
    ///
    /// Returns `true` if the algorithm should stop early (callback returned True).
    /// Returns `false` to continue.
    ///
    /// If no callback is set, always returns `false`.
    pub fn call(&mut self, progress: &Progress) -> PyResult<bool> {
        let cb = match &self.callback {
            Some(cb) => cb,
            None => return Ok(false),
        };

        self.last_call = progress.iteration;

        Python::attach(|py| {
            let py_progress = progress.to_pydict(py)?;
            let result = cb.call1(py, (py_progress,))?;

            // Python callback returns True to stop, False/None to continue
            if result.is_none(py) {
                Ok(false)
            } else {
                result.extract::<bool>(py)
            }
        })
    }

    /// Convenience method: check and call if needed.
    ///
    /// Returns `true` if should stop early.
    pub fn report(&mut self, progress: &Progress) -> PyResult<bool> {
        if self.should_call(progress.iteration) {
            self.call(progress)
        } else {
            Ok(false)
        }
    }
}

impl Default for ProgressCallback {
    fn default() -> Self {
        Self::none()
    }
}

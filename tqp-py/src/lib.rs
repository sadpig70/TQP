#![allow(non_local_definitions)]

use ndarray::Array2;
use num_complex::Complex64;
use pyo3::prelude::*;
use tqp_core::ops;
use tqp_core::TQPState;

/// Python wrapper for TQPState
#[pyclass]
struct PyTQPState {
    inner: TQPState,
}

#[pymethods]
impl PyTQPState {
    #[new]
    fn new(num_qubits: usize, num_time_bins: usize, num_layers: usize) -> Self {
        PyTQPState {
            inner: TQPState::new(num_qubits, num_time_bins, num_layers),
        }
    }

    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    fn probability(&self, index: usize) -> f64 {
        self.inner.probability(index)
    }

    fn normalize(&mut self) {
        self.inner.normalize();
    }

    /// Returns the expectation value of an observable (simplified: just Z on a qubit for now, or generic matrix?)
    /// For MVP Phase 3, let's implement a generic expectation value for a given operator matrix.
    /// <psi|O|psi>
    /// Note: This is computationally expensive for full state.
    /// Let's implement a Z-expectation for a specific qubit which is common for QAOA.
    fn expval_z(&self, qubit_idx: usize) -> f64 {
        // Z operator on qubit_idx: |0><0| - |1><1|
        // Sum over all basis states: if qubit_idx is 0, add prob; if 1, subtract prob.

        let mut expval = 0.0;
        let dim = self.inner.dimension();
        let _spatial_dim = 1 << self.inner.dims.num_qubits;
        let target_bit = 1 << qubit_idx;

        // We need to iterate over all indices and check the bit at qubit_idx.
        // Index mapping: layer * ... + bin * ... + spatial
        // We only care about spatial part.

        for i in 0..dim {
            let (_, _, s) = self.inner.get_indices(i);
            let prob = self.inner.probability(i);

            if (s & target_bit) == 0 {
                expval += prob;
            } else {
                expval -= prob;
            }
        }
        expval
    }
}

/// Python wrapper for TQP Operations
#[pyclass]
struct PyOps;

#[pymethods]
impl PyOps {
    #[staticmethod]
    fn fast_mux_shift(state: &mut PyTQPState, delta: usize) {
        ops::fast_mux_shift(&mut state.inner, delta);
    }

    #[staticmethod]
    fn temporal_entangle(state: &mut PyTQPState, qubit_idx: usize, bin1: usize, bin2: usize) {
        ops::temporal_entangle(&mut state.inner, qubit_idx, bin1, bin2);
    }

    #[staticmethod]
    fn measure(state: &mut PyTQPState) -> usize {
        ops::measure(&mut state.inner)
    }

    #[staticmethod]
    fn apply_spatial_gate(
        state: &mut PyTQPState,
        qubit_idx: usize,
        gate_matrix: Vec<Vec<Complex64>>,
    ) -> PyResult<()> {
        // Convert Vec<Vec<Complex64>> to Array2<Complex64>
        let rows = gate_matrix.len();
        let cols = gate_matrix[0].len();

        if rows != 2 || cols != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Gate must be 2x2 matrix",
            ));
        }

        let mut arr = Array2::<Complex64>::zeros((2, 2));
        for i in 0..2 {
            for j in 0..2 {
                arr[[i, j]] = gate_matrix[i][j];
            }
        }

        ops::apply_spatial_gate(&mut state.inner, qubit_idx, &arr);
        Ok(())
    }

    #[staticmethod]
    fn apply_spatial_gate_2q(
        state: &mut PyTQPState,
        qubit1: usize,
        qubit2: usize,
        gate_matrix: Vec<Vec<Complex64>>,
    ) -> PyResult<()> {
        let rows = gate_matrix.len();
        let cols = gate_matrix[0].len();

        if rows != 4 || cols != 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Gate must be 4x4 matrix",
            ));
        }

        let mut arr = Array2::<Complex64>::zeros((4, 4));
        for i in 0..4 {
            for j in 0..4 {
                arr[[i, j]] = gate_matrix[i][j];
            }
        }

        ops::apply_spatial_gate_2q(&mut state.inner, qubit1, qubit2, &arr);
        Ok(())
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn tqp(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTQPState>()?;
    m.add_class::<PyOps>()?;
    Ok(())
}

//! Sparse Quantum Operations for TQP
//!
//! Provides quantum gate operations optimized for sparse state vectors.
//! These operations maintain O(nnz) complexity where possible.
//!
//! # Design Principles
//! 1. **Lazy Evaluation**: Collect affected indices first, then compute
//! 2. **Minimal Allocation**: Reuse buffers, avoid temporary vectors
//! 3. **Automatic Pruning**: Remove near-zero amplitudes after operations
//! 4. **Dense Compatibility**: Results match dense operations within numerical precision

use ndarray::Array2;
use num_complex::Complex64;
use rustc_hash::FxHashMap;

use crate::sparse::SparseStateVector;

/// Applies a single-qubit gate to a sparse state vector
///
/// # Algorithm
/// For each non-zero amplitude at index `idx`:
/// 1. Compute partner index `idx ^ (1 << qubit_idx)`
/// 2. Apply 2x2 unitary matrix to (idx, partner) pair
/// 3. Prune results below threshold
///
/// # Complexity
/// - Time: O(nnz) where nnz is non-zero count
/// - Space: O(nnz) for result storage
///
/// # Arguments
/// * `state` - Mutable reference to sparse state vector
/// * `qubit_idx` - Target qubit index (0-indexed, LSB)
/// * `gate` - 2x2 unitary matrix
///
/// # Example
/// ```ignore
/// let h = hadamard_gate();
/// apply_gate_sparse(&mut state, 0, &h);
/// ```
pub fn apply_gate_sparse(
    state: &mut SparseStateVector,
    qubit_idx: usize,
    gate: &Array2<Complex64>,
) {
    debug_assert!(gate.shape() == [2, 2], "Gate must be 2x2");
    debug_assert!(
        qubit_idx < state.dims.num_qubits,
        "Qubit index out of range"
    );

    let bit = 1 << qubit_idx;

    // Extract gate elements
    let u00 = gate[[0, 0]];
    let u01 = gate[[0, 1]];
    let u10 = gate[[1, 0]];
    let u11 = gate[[1, 1]];

    // Collect all unique pairs that need processing
    // A pair (idx0, idx1) where idx0 has bit=0 and idx1 has bit=1
    let mut pairs_to_process: FxHashMap<usize, ()> = FxHashMap::default();

    for &idx in state.iter().map(|(idx, _)| idx) {
        // Get the base index (with target bit = 0)
        let base_idx = idx & !bit;
        pairs_to_process.insert(base_idx, ());
    }

    // Store original amplitudes for reading (avoid borrow conflicts)
    let old_amplitudes: FxHashMap<usize, Complex64> = state.iter().map(|(&k, &v)| (k, v)).collect();

    // Helper to get amplitude from old state
    let get_old = |idx: usize| -> Complex64 {
        old_amplitudes
            .get(&idx)
            .copied()
            .unwrap_or(Complex64::new(0.0, 0.0))
    };

    // Process each pair
    for &base_idx in pairs_to_process.keys() {
        let idx0 = base_idx; // bit = 0
        let idx1 = base_idx | bit; // bit = 1

        let alpha = get_old(idx0); // amplitude for |0⟩ state
        let beta = get_old(idx1); // amplitude for |1⟩ state

        // Apply gate: [u00 u01] [alpha]   [new_alpha]
        //             [u10 u11] [beta ] = [new_beta ]
        let new_alpha = u00 * alpha + u01 * beta;
        let new_beta = u10 * alpha + u11 * beta;

        // Update state with pruning
        state.set_amplitude(idx0, new_alpha);
        state.set_amplitude(idx1, new_beta);
    }
}

/// Applies a two-qubit gate to a sparse state vector
///
/// # Algorithm
/// For each affected 4-tuple of indices:
/// 1. Identify (idx00, idx01, idx10, idx11) based on qubit positions
/// 2. Apply 4x4 unitary matrix
/// 3. Prune results below threshold
///
/// # Index Convention
/// - `qubit1` is the first qubit (controls row index in gate matrix)
/// - `qubit2` is the second qubit (controls column index in gate matrix)
/// - idx = |...q1...q2...⟩ where q1, q2 are the target qubit values
///
/// # Complexity
/// - Time: O(nnz)
/// - Space: O(nnz)
///
/// # Arguments
/// * `state` - Mutable reference to sparse state vector
/// * `qubit1` - First qubit index
/// * `qubit2` - Second qubit index
/// * `gate` - 4x4 unitary matrix
pub fn apply_gate_2q_sparse(
    state: &mut SparseStateVector,
    qubit1: usize,
    qubit2: usize,
    gate: &Array2<Complex64>,
) {
    debug_assert!(gate.shape() == [4, 4], "Gate must be 4x4");
    debug_assert!(qubit1 < state.dims.num_qubits, "Qubit1 index out of range");
    debug_assert!(qubit2 < state.dims.num_qubits, "Qubit2 index out of range");
    debug_assert!(qubit1 != qubit2, "Qubit indices must be different");

    let bit1 = 1 << qubit1;
    let bit2 = 1 << qubit2;
    let mask = bit1 | bit2;

    // Collect all unique 4-tuples (base indices with both bits = 0)
    let mut quads_to_process: FxHashMap<usize, ()> = FxHashMap::default();

    for &idx in state.iter().map(|(idx, _)| idx) {
        let base_idx = idx & !mask;
        quads_to_process.insert(base_idx, ());
    }

    // Store original amplitudes
    let old_amplitudes: FxHashMap<usize, Complex64> = state.iter().map(|(&k, &v)| (k, v)).collect();

    let get_old = |idx: usize| -> Complex64 {
        old_amplitudes
            .get(&idx)
            .copied()
            .unwrap_or(Complex64::new(0.0, 0.0))
    };

    // Process each quad
    for &base_idx in quads_to_process.keys() {
        // Construct the 4 indices:
        // |q1=0, q2=0⟩, |q1=0, q2=1⟩, |q1=1, q2=0⟩, |q1=1, q2=1⟩
        let idx00 = base_idx;
        let idx01 = base_idx | bit2;
        let idx10 = base_idx | bit1;
        let idx11 = base_idx | bit1 | bit2;

        // Get old amplitudes
        let v = [
            get_old(idx00),
            get_old(idx01),
            get_old(idx10),
            get_old(idx11),
        ];

        // Apply 4x4 gate matrix
        let mut result = [Complex64::new(0.0, 0.0); 4];
        for r in 0..4 {
            for c in 0..4 {
                result[r] += gate[[r, c]] * v[c];
            }
        }

        // Update state with pruning
        state.set_amplitude(idx00, result[0]);
        state.set_amplitude(idx01, result[1]);
        state.set_amplitude(idx10, result[2]);
        state.set_amplitude(idx11, result[3]);
    }
}

/// Applies a controlled-NOT (CNOT) gate optimized for sparse states
///
/// CNOT flips the target qubit if the control qubit is |1⟩.
/// This is more efficient than the general 2Q gate for CNOT.
///
/// # Arguments
/// * `state` - Mutable reference to sparse state vector
/// * `control` - Control qubit index
/// * `target` - Target qubit index
pub fn apply_cnot_sparse(state: &mut SparseStateVector, control: usize, target: usize) {
    debug_assert!(control != target, "Control and target must differ");

    let control_bit = 1 << control;
    let target_bit = 1 << target;

    // Collect indices where control qubit is |1⟩
    let indices_to_flip: Vec<usize> = state
        .iter()
        .filter_map(|(&idx, _)| {
            let (_, _, spatial) = state.get_indices(idx);
            if (spatial & control_bit) != 0 {
                Some(idx)
            } else {
                None
            }
        })
        .collect();

    // Store amplitudes that will be swapped
    let mut swaps: Vec<(usize, usize, Complex64, Complex64)> = Vec::new();

    for &idx in &indices_to_flip {
        let partner = idx ^ target_bit;

        // Only process each pair once (when idx < partner)
        if idx < partner {
            let amp_idx = state.get_amplitude(idx);
            let amp_partner = state.get_amplitude(partner);
            swaps.push((idx, partner, amp_idx, amp_partner));
        }
    }

    // Apply swaps
    for (idx, partner, amp_idx, amp_partner) in swaps {
        state.set_amplitude(idx, amp_partner);
        state.set_amplitude(partner, amp_idx);
    }
}

/// Applies a controlled-Z (CZ) gate optimized for sparse states
///
/// CZ applies a phase of -1 to |11⟩ state.
///
/// # Arguments
/// * `state` - Mutable reference to sparse state vector
/// * `qubit1` - First qubit index
/// * `qubit2` - Second qubit index
pub fn apply_cz_sparse(state: &mut SparseStateVector, qubit1: usize, qubit2: usize) {
    let bit1 = 1 << qubit1;
    let bit2 = 1 << qubit2;

    // Collect indices where both qubits are |1⟩
    let indices_to_flip: Vec<usize> = state
        .iter()
        .filter_map(|(&idx, _)| {
            let (_, _, spatial) = state.get_indices(idx);
            if (spatial & bit1) != 0 && (spatial & bit2) != 0 {
                Some(idx)
            } else {
                None
            }
        })
        .collect();

    // Apply -1 phase
    for idx in indices_to_flip {
        let amp = state.get_amplitude(idx);
        state.set_amplitude(idx, -amp);
    }
}

/// Applies a SWAP gate between two qubits
///
/// # Arguments
/// * `state` - Mutable reference to sparse state vector
/// * `qubit1` - First qubit index
/// * `qubit2` - Second qubit index
pub fn apply_swap_sparse(state: &mut SparseStateVector, qubit1: usize, qubit2: usize) {
    let bit1 = 1 << qubit1;
    let bit2 = 1 << qubit2;

    // Collect pairs where exactly one of the bits differs
    // (01) ↔ (10) states
    let mut swaps: Vec<(usize, usize, Complex64, Complex64)> = Vec::new();
    let mut processed: FxHashMap<usize, ()> = FxHashMap::default();

    for (&idx, _) in state.iter() {
        let (_, _, spatial) = state.get_indices(idx);
        let q1_val = (spatial & bit1) != 0;
        let q2_val = (spatial & bit2) != 0;

        // Only process |01⟩ and |10⟩ cases (XOR = 1)
        if q1_val != q2_val {
            let base = idx & !(bit1 | bit2);
            if let std::collections::hash_map::Entry::Vacant(e) = processed.entry(base) {
                e.insert(());

                let idx_01 = base | bit2; // q1=0, q2=1
                let idx_10 = base | bit1; // q1=1, q2=0

                let amp_01 = state.get_amplitude(idx_01);
                let amp_10 = state.get_amplitude(idx_10);

                swaps.push((idx_01, idx_10, amp_01, amp_10));
            }
        }
    }

    // Apply swaps
    for (idx_01, idx_10, amp_01, amp_10) in swaps {
        state.set_amplitude(idx_01, amp_10);
        state.set_amplitude(idx_10, amp_01);
    }
}

/// Measures a qubit in the computational basis (sparse version)
///
/// Collapses the state and returns the measurement outcome.
///
/// # Arguments
/// * `state` - Mutable reference to sparse state vector
/// * `qubit_idx` - Qubit to measure
///
/// # Returns
/// Measurement outcome (0 or 1)
pub fn measure_qubit_sparse(state: &mut SparseStateVector, qubit_idx: usize) -> u8 {
    let bit = 1 << qubit_idx;

    // Calculate probability of measuring |1⟩
    let prob_one = state.get_marginal_probability(qubit_idx);

    // Sample outcome
    let outcome = if rand::random::<f64>() < prob_one {
        1
    } else {
        0
    };

    // Collapse state
    let normalization = if outcome == 1 {
        prob_one
    } else {
        1.0 - prob_one
    };
    let norm_factor = if normalization > 1e-30 {
        1.0 / normalization.sqrt()
    } else {
        1.0
    };

    // Collect indices to remove and rescale
    let indices: Vec<(usize, Complex64)> = state.iter().map(|(&idx, &amp)| (idx, amp)).collect();

    for (idx, amp) in indices {
        let (_, _, spatial) = state.get_indices(idx);
        let qubit_val = if (spatial & bit) != 0 { 1 } else { 0 };

        if qubit_val == outcome {
            // Keep and rescale
            state.set_amplitude(idx, amp * norm_factor);
        } else {
            // Remove (collapse)
            state.set_amplitude(idx, Complex64::new(0.0, 0.0));
        }
    }

    outcome
}

/// Measures the entire state in computational basis (sparse version)
///
/// # Arguments
/// * `state` - Reference to sparse state vector (not modified)
///
/// # Returns
/// Sampled basis state index
pub fn measure_sparse(state: &SparseStateVector) -> usize {
    let r: f64 = rand::random();
    let mut cumulative = 0.0;

    for (&idx, &amp) in state.iter() {
        cumulative += amp.norm_sqr();
        if r <= cumulative {
            return idx;
        }
    }

    // Fallback to last non-zero index (shouldn't happen for normalized states)
    state.iter().map(|(&idx, _)| idx).max().unwrap_or(0)
}

/// Computes expectation value of Z operator on a qubit
///
/// ⟨Z_i⟩ = P(0) - P(1)
///
/// # Arguments
/// * `state` - Reference to sparse state vector
/// * `qubit_idx` - Target qubit index
pub fn expval_z_sparse(state: &SparseStateVector, qubit_idx: usize) -> f64 {
    let prob_one = state.get_marginal_probability(qubit_idx);
    let prob_zero = 1.0 - prob_one;
    prob_zero - prob_one
}

/// Computes inner product ⟨ψ|φ⟩ between two sparse states
///
/// # Arguments
/// * `state1` - First sparse state (bra)
/// * `state2` - Second sparse state (ket)
///
/// # Returns
/// Complex inner product
pub fn inner_product_sparse(state1: &SparseStateVector, state2: &SparseStateVector) -> Complex64 {
    debug_assert_eq!(state1.dimension(), state2.dimension(), "Dimension mismatch");

    let mut result = Complex64::new(0.0, 0.0);

    // Only need to iterate over indices present in both states
    for (&idx, &amp1) in state1.iter() {
        let amp2 = state2.get_amplitude(idx);
        result += amp1.conj() * amp2;
    }

    result
}

/// Computes fidelity |⟨ψ|φ⟩|² between two sparse states
///
/// # Arguments
/// * `state1` - First sparse state
/// * `state2` - Second sparse state
pub fn fidelity_sparse(state1: &SparseStateVector, state2: &SparseStateVector) -> f64 {
    inner_product_sparse(state1, state2).norm_sqr()
}

/// Standard gate matrices for convenience
pub mod gates {
    use ndarray::{array, Array2};
    use num_complex::Complex64;
    use std::f64::consts::FRAC_1_SQRT_2;

    /// Hadamard gate
    pub fn hadamard() -> Array2<Complex64> {
        let h = FRAC_1_SQRT_2;
        array![
            [Complex64::new(h, 0.0), Complex64::new(h, 0.0)],
            [Complex64::new(h, 0.0), Complex64::new(-h, 0.0)]
        ]
    }

    /// Pauli-X gate (NOT)
    pub fn pauli_x() -> Array2<Complex64> {
        array![
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]
        ]
    }

    /// Pauli-Y gate
    pub fn pauli_y() -> Array2<Complex64> {
        array![
            [Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0)],
            [Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0)]
        ]
    }

    /// Pauli-Z gate
    pub fn pauli_z() -> Array2<Complex64> {
        array![
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)]
        ]
    }

    /// S gate (sqrt(Z))
    pub fn s_gate() -> Array2<Complex64> {
        array![
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(0.0, 1.0)]
        ]
    }

    /// T gate (sqrt(S))
    pub fn t_gate() -> Array2<Complex64> {
        let t = std::f64::consts::FRAC_1_SQRT_2;
        array![
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(t, t)]
        ]
    }

    /// Rotation around X-axis
    pub fn rx(theta: f64) -> Array2<Complex64> {
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        array![
            [Complex64::new(c, 0.0), Complex64::new(0.0, -s)],
            [Complex64::new(0.0, -s), Complex64::new(c, 0.0)]
        ]
    }

    /// Rotation around Y-axis
    pub fn ry(theta: f64) -> Array2<Complex64> {
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        array![
            [Complex64::new(c, 0.0), Complex64::new(-s, 0.0)],
            [Complex64::new(s, 0.0), Complex64::new(c, 0.0)]
        ]
    }

    /// Rotation around Z-axis
    pub fn rz(theta: f64) -> Array2<Complex64> {
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        array![
            [Complex64::new(c, -s), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(c, s)]
        ]
    }

    /// CNOT gate matrix (4x4)
    pub fn cnot() -> Array2<Complex64> {
        array![
            [
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0)
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0)
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0)
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0)
            ]
        ]
    }

    /// CZ gate matrix (4x4)
    pub fn cz() -> Array2<Complex64> {
        array![
            [
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0)
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0)
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0)
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-1.0, 0.0)
            ]
        ]
    }

    /// SWAP gate matrix (4x4)
    pub fn swap() -> Array2<Complex64> {
        array![
            [
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0)
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0)
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0)
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0)
            ]
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops;
    use crate::state::TQPState;

    const EPSILON: f64 = 1e-10;

    fn assert_states_equal(sparse: &SparseStateVector, dense: &TQPState) {
        let sparse_dense = sparse.to_dense();
        for i in 0..dense.dimension() {
            let diff = (sparse_dense.state_vector[i] - dense.state_vector[i]).norm();
            assert!(
                diff < EPSILON,
                "Mismatch at index {}: sparse={:?}, dense={:?}, diff={}",
                i,
                sparse_dense.state_vector[i],
                dense.state_vector[i],
                diff
            );
        }
    }

    #[test]
    fn test_hadamard_sparse() {
        let mut sparse = SparseStateVector::new(2, 1, 1);
        let mut dense = TQPState::new(2, 1, 1);

        let h = gates::hadamard();

        apply_gate_sparse(&mut sparse, 0, &h);
        ops::apply_spatial_gate(&mut dense, 0, &h);

        assert_states_equal(&sparse, &dense);
    }

    #[test]
    fn test_pauli_x_sparse() {
        let mut sparse = SparseStateVector::new(2, 1, 1);
        let mut dense = TQPState::new(2, 1, 1);

        let x = gates::pauli_x();

        apply_gate_sparse(&mut sparse, 1, &x);
        ops::apply_spatial_gate(&mut dense, 1, &x);

        assert_states_equal(&sparse, &dense);
    }

    #[test]
    fn test_sequence_hzh() {
        // H-Z-H should equal X
        let mut sparse = SparseStateVector::new(1, 1, 1);
        let mut dense = TQPState::new(1, 1, 1);

        let h = gates::hadamard();
        let z = gates::pauli_z();

        apply_gate_sparse(&mut sparse, 0, &h);
        apply_gate_sparse(&mut sparse, 0, &z);
        apply_gate_sparse(&mut sparse, 0, &h);

        ops::apply_spatial_gate(&mut dense, 0, &h);
        ops::apply_spatial_gate(&mut dense, 0, &z);
        ops::apply_spatial_gate(&mut dense, 0, &h);

        assert_states_equal(&sparse, &dense);
    }

    #[test]
    fn test_cnot_sparse() {
        let mut sparse = SparseStateVector::new(2, 1, 1);
        let mut dense = TQPState::new(2, 1, 1);

        // Apply H to control, then CNOT → Bell state
        let h = gates::hadamard();
        let cnot = gates::cnot();

        apply_gate_sparse(&mut sparse, 0, &h);
        apply_gate_2q_sparse(&mut sparse, 0, 1, &cnot);

        ops::apply_spatial_gate(&mut dense, 0, &h);
        ops::apply_spatial_gate_2q(&mut dense, 0, 1, &cnot);

        assert_states_equal(&sparse, &dense);

        // Verify Bell state properties
        assert!((sparse.probability(0) - 0.5).abs() < EPSILON);
        assert!((sparse.probability(3) - 0.5).abs() < EPSILON);
        assert!(sparse.probability(1) < EPSILON);
        assert!(sparse.probability(2) < EPSILON);
    }

    #[test]
    fn test_cnot_optimized() {
        let mut sparse1 = SparseStateVector::new(2, 1, 1);
        let mut sparse2 = SparseStateVector::new(2, 1, 1);

        // Prepare |10⟩ state
        let x = gates::pauli_x();
        apply_gate_sparse(&mut sparse1, 0, &x);
        apply_gate_sparse(&mut sparse2, 0, &x);

        // Apply CNOT using matrix vs optimized
        let cnot = gates::cnot();
        apply_gate_2q_sparse(&mut sparse1, 0, 1, &cnot);
        apply_cnot_sparse(&mut sparse2, 0, 1);

        // Results should match
        for i in 0..sparse1.dimension() {
            let diff = (sparse1.get_amplitude(i) - sparse2.get_amplitude(i)).norm();
            assert!(diff < EPSILON, "Mismatch at {}", i);
        }
    }

    #[test]
    fn test_cz_sparse() {
        let mut sparse = SparseStateVector::new(2, 1, 1);

        // Prepare |11⟩ state
        let x = gates::pauli_x();
        apply_gate_sparse(&mut sparse, 0, &x);
        apply_gate_sparse(&mut sparse, 1, &x);

        // Apply CZ
        apply_cz_sparse(&mut sparse, 0, 1);

        // |11⟩ should get -1 phase
        let amp = sparse.get_amplitude(3);
        assert!((amp.re - (-1.0)).abs() < EPSILON);
        assert!(amp.im.abs() < EPSILON);
    }

    #[test]
    fn test_swap_sparse() {
        let mut sparse = SparseStateVector::new(2, 1, 1);

        // Prepare |01⟩ state (qubit 1 is |1⟩)
        let x = gates::pauli_x();
        apply_gate_sparse(&mut sparse, 1, &x);

        assert!((sparse.probability(2) - 1.0).abs() < EPSILON); // |01⟩ = index 2

        // SWAP should give |10⟩
        apply_swap_sparse(&mut sparse, 0, 1);

        assert!((sparse.probability(1) - 1.0).abs() < EPSILON); // |10⟩ = index 1
    }

    #[test]
    fn test_rotation_gates() {
        let mut sparse = SparseStateVector::new(1, 1, 1);
        let mut dense = TQPState::new(1, 1, 1);

        let theta = std::f64::consts::PI / 4.0;
        let rx = gates::rx(theta);
        let ry = gates::ry(theta);
        let rz = gates::rz(theta);

        // Apply Rx, Ry, Rz sequence
        apply_gate_sparse(&mut sparse, 0, &rx);
        apply_gate_sparse(&mut sparse, 0, &ry);
        apply_gate_sparse(&mut sparse, 0, &rz);

        ops::apply_spatial_gate(&mut dense, 0, &rx);
        ops::apply_spatial_gate(&mut dense, 0, &ry);
        ops::apply_spatial_gate(&mut dense, 0, &rz);

        assert_states_equal(&sparse, &dense);
    }

    #[test]
    fn test_normalization_preserved() {
        let mut sparse = SparseStateVector::new(3, 1, 1);

        let h = gates::hadamard();
        let cnot = gates::cnot();

        // Apply random circuit
        apply_gate_sparse(&mut sparse, 0, &h);
        apply_gate_sparse(&mut sparse, 1, &h);
        apply_gate_2q_sparse(&mut sparse, 0, 1, &cnot);
        apply_gate_sparse(&mut sparse, 2, &h);
        apply_gate_2q_sparse(&mut sparse, 1, 2, &cnot);

        let total_prob = sparse.total_probability();
        assert!(
            (total_prob - 1.0).abs() < EPSILON,
            "Normalization violated: {}",
            total_prob
        );
    }

    #[test]
    fn test_multi_time_bin() {
        // Test with multiple time bins
        let mut sparse = SparseStateVector::new(2, 2, 1);
        let mut dense = TQPState::new(2, 2, 1);

        let h = gates::hadamard();

        apply_gate_sparse(&mut sparse, 0, &h);
        ops::apply_spatial_gate(&mut dense, 0, &h);

        assert_states_equal(&sparse, &dense);
    }

    #[test]
    fn test_sparsity_after_operations() {
        let mut sparse = SparseStateVector::new(10, 1, 1); // 2^10 = 1024 states

        // Initially only |0⟩
        assert_eq!(sparse.nnz(), 1);

        // H on qubit 0 → 2 non-zero
        apply_gate_sparse(&mut sparse, 0, &gates::hadamard());
        assert_eq!(sparse.nnz(), 2);

        // H on qubit 1 → 4 non-zero
        apply_gate_sparse(&mut sparse, 1, &gates::hadamard());
        assert_eq!(sparse.nnz(), 4);

        // H on qubit 2 → 8 non-zero
        apply_gate_sparse(&mut sparse, 2, &gates::hadamard());
        assert_eq!(sparse.nnz(), 8);

        // Still very sparse (8/1024 = 0.78%)
        assert!(sparse.sparsity_ratio() < 0.01);
    }

    #[test]
    fn test_expval_z() {
        let mut sparse = SparseStateVector::new(1, 1, 1);

        // |0⟩ → ⟨Z⟩ = 1
        assert!((expval_z_sparse(&sparse, 0) - 1.0).abs() < EPSILON);

        // |+⟩ → ⟨Z⟩ = 0
        apply_gate_sparse(&mut sparse, 0, &gates::hadamard());
        assert!(expval_z_sparse(&sparse, 0).abs() < EPSILON);

        // |1⟩ → ⟨Z⟩ = -1
        let mut sparse2 = SparseStateVector::new(1, 1, 1);
        apply_gate_sparse(&mut sparse2, 0, &gates::pauli_x());
        assert!((expval_z_sparse(&sparse2, 0) - (-1.0)).abs() < EPSILON);
    }

    #[test]
    fn test_inner_product() {
        let sparse1 = SparseStateVector::new(2, 1, 1); // |00⟩
        let sparse2 = SparseStateVector::new(2, 1, 1); // |00⟩

        // ⟨00|00⟩ = 1
        let ip = inner_product_sparse(&sparse1, &sparse2);
        assert!((ip.re - 1.0).abs() < EPSILON);
        assert!(ip.im.abs() < EPSILON);

        // Orthogonal states
        let mut sparse3 = SparseStateVector::new(2, 1, 1);
        apply_gate_sparse(&mut sparse3, 0, &gates::pauli_x()); // |10⟩

        let ip_orth = inner_product_sparse(&sparse1, &sparse3);
        assert!(ip_orth.norm() < EPSILON);
    }

    #[test]
    fn test_fidelity() {
        let sparse1 = SparseStateVector::new(2, 1, 1);

        // Fidelity with itself = 1
        assert!((fidelity_sparse(&sparse1, &sparse1) - 1.0).abs() < EPSILON);

        // Fidelity with orthogonal state = 0
        let mut sparse2 = SparseStateVector::new(2, 1, 1);
        apply_gate_sparse(&mut sparse2, 0, &gates::pauli_x());
        assert!(fidelity_sparse(&sparse1, &sparse2) < EPSILON);
    }

    #[test]
    fn test_measure_sparse() {
        let mut sparse = SparseStateVector::new(2, 1, 1);

        // |00⟩ should always measure 0
        for _ in 0..10 {
            assert_eq!(measure_sparse(&sparse), 0);
        }

        // Bell state should measure 0 or 3
        apply_gate_sparse(&mut sparse, 0, &gates::hadamard());
        apply_gate_2q_sparse(&mut sparse, 0, 1, &gates::cnot());

        for _ in 0..100 {
            let result = measure_sparse(&sparse);
            assert!(result == 0 || result == 3, "Unexpected result: {}", result);
        }
    }
}

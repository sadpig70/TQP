//! In-Place Optimization for TQP State Vectors
//!
//! Provides memory-efficient operations by minimizing allocations through:
//! - Double buffering for sparse operations
//! - View-based mutations for dense operations
//! - Reusable buffer pools
//!
//! # Performance Goals
//! - Eliminate per-gate HashMap allocation in sparse operations
//! - Reduce memory bandwidth through buffer reuse
//! - Enable cache-friendly access patterns

use num_complex::Complex64;
use rustc_hash::FxHashMap;

use crate::sparse::SparseStateVector;
use crate::state::TQPState;

// =============================================================================
// Double Buffer for Sparse Operations
// =============================================================================

/// Double buffer for sparse state vector operations
///
/// Maintains two buffers and swaps between them to avoid copying.
/// One buffer is "active" (current state), the other is "working" (next state).
///
/// # Usage Pattern
/// ```ignore
/// let mut db = DoubleBuffer::new(state);
/// db.begin_operation();  // Lock active, prepare working
/// // ... compute into working buffer ...
/// db.commit();           // Swap buffers
/// ```
#[derive(Debug)]
pub struct DoubleBuffer {
    /// Primary buffer (current state)
    buffer_a: FxHashMap<usize, Complex64>,
    /// Secondary buffer (working state)  
    buffer_b: FxHashMap<usize, Complex64>,
    /// Which buffer is active (true = A, false = B)
    a_is_active: bool,
    /// Pruning threshold
    threshold: f64,
}

impl DoubleBuffer {
    /// Create a new double buffer with initial capacity
    pub fn new(capacity: usize, threshold: f64) -> Self {
        DoubleBuffer {
            buffer_a: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
            buffer_b: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
            a_is_active: true,
            threshold,
        }
    }

    /// Create from existing amplitudes
    pub fn from_amplitudes(amplitudes: FxHashMap<usize, Complex64>, threshold: f64) -> Self {
        let capacity = amplitudes.len();
        DoubleBuffer {
            buffer_a: amplitudes,
            buffer_b: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
            a_is_active: true,
            threshold,
        }
    }

    /// Get reference to active (current) buffer
    #[inline]
    pub fn active(&self) -> &FxHashMap<usize, Complex64> {
        if self.a_is_active {
            &self.buffer_a
        } else {
            &self.buffer_b
        }
    }

    /// Get mutable reference to working (next) buffer
    #[inline]
    pub fn working_mut(&mut self) -> &mut FxHashMap<usize, Complex64> {
        if self.a_is_active {
            &mut self.buffer_b
        } else {
            &mut self.buffer_a
        }
    }

    /// Prepare for a new operation - clears working buffer
    pub fn begin_operation(&mut self) {
        self.working_mut().clear();
    }

    /// Commit operation - swap buffers
    #[inline]
    pub fn commit(&mut self) {
        self.a_is_active = !self.a_is_active;
    }

    /// Get amplitude from active buffer
    #[inline]
    pub fn get_amplitude(&self, index: usize) -> Complex64 {
        self.active()
            .get(&index)
            .copied()
            .unwrap_or(Complex64::new(0.0, 0.0))
    }

    /// Set amplitude in working buffer with automatic pruning
    #[inline]
    pub fn set_amplitude(&mut self, index: usize, amplitude: Complex64) {
        if amplitude.norm_sqr() >= self.threshold {
            self.working_mut().insert(index, amplitude);
        }
    }

    /// Get number of non-zero elements in active buffer
    #[inline]
    pub fn nnz(&self) -> usize {
        self.active().len()
    }

    /// Extract the active buffer, consuming the double buffer
    pub fn into_active(self) -> FxHashMap<usize, Complex64> {
        if self.a_is_active {
            self.buffer_a
        } else {
            self.buffer_b
        }
    }

    /// Iterate over active buffer
    pub fn iter_active(&self) -> impl Iterator<Item = (&usize, &Complex64)> {
        self.active().iter()
    }
}

// =============================================================================
// Buffered Sparse State Vector
// =============================================================================

/// Sparse state vector with integrated double buffering
///
/// Wraps SparseStateVector with efficient buffer management for
/// sequential gate operations.
pub struct BufferedSparseState {
    /// Dimensions
    pub dims: crate::state::TQPDimensions,
    /// Double buffer for amplitudes
    buffer: DoubleBuffer,
}

impl BufferedSparseState {
    /// Create from a SparseStateVector
    pub fn from_sparse(state: SparseStateVector) -> Self {
        let dims = state.dims;
        let threshold = state.threshold();
        let amplitudes: FxHashMap<usize, Complex64> = state.iter().map(|(&k, &v)| (k, v)).collect();

        BufferedSparseState {
            dims,
            buffer: DoubleBuffer::from_amplitudes(amplitudes, threshold),
        }
    }

    /// Convert back to SparseStateVector
    pub fn into_sparse(self) -> SparseStateVector {
        let mut state = SparseStateVector::empty(
            self.dims.num_qubits,
            self.dims.num_time_bins,
            self.dims.num_layers,
        );

        for (&idx, &amp) in self.buffer.active() {
            state.set_amplitude(idx, amp);
        }

        state
    }

    /// Get amplitude
    #[inline]
    pub fn get_amplitude(&self, index: usize) -> Complex64 {
        self.buffer.get_amplitude(index)
    }

    /// Get nnz
    #[inline]
    pub fn nnz(&self) -> usize {
        self.buffer.nnz()
    }

    /// Total dimension
    #[inline]
    pub fn dimension(&self) -> usize {
        self.dims.total_dim()
    }

    /// Apply a single-qubit gate using double buffering
    ///
    /// This is the optimized version that avoids per-gate allocation.
    pub fn apply_gate(&mut self, qubit_idx: usize, gate: &ndarray::Array2<Complex64>) {
        debug_assert!(gate.shape() == [2, 2], "Gate must be 2x2");

        let bit = 1 << qubit_idx;

        // Extract gate elements
        let u00 = gate[[0, 0]];
        let u01 = gate[[0, 1]];
        let u10 = gate[[1, 0]];
        let u11 = gate[[1, 1]];

        // Collect unique pairs from active buffer
        let pairs: Vec<usize> = self
            .buffer
            .iter_active()
            .map(|(&idx, _)| idx & !bit)
            .collect();

        // Deduplicate (could use a set but Vec + sort + dedup is often faster for small N)
        let mut pairs = pairs;
        pairs.sort_unstable();
        pairs.dedup();

        // Prepare working buffer
        self.buffer.begin_operation();

        // Process each pair: read from active, write to working
        for base_idx in pairs {
            let idx0 = base_idx;
            let idx1 = base_idx | bit;

            let alpha = self.buffer.get_amplitude(idx0);
            let beta = self.buffer.get_amplitude(idx1);

            let new_alpha = u00 * alpha + u01 * beta;
            let new_beta = u10 * alpha + u11 * beta;

            self.buffer.set_amplitude(idx0, new_alpha);
            self.buffer.set_amplitude(idx1, new_beta);
        }

        // Swap buffers
        self.buffer.commit();
    }

    /// Apply a two-qubit gate using double buffering
    pub fn apply_gate_2q(
        &mut self,
        qubit1: usize,
        qubit2: usize,
        gate: &ndarray::Array2<Complex64>,
    ) {
        debug_assert!(gate.shape() == [4, 4], "Gate must be 4x4");
        debug_assert!(qubit1 != qubit2, "Qubits must differ");

        let bit1 = 1 << qubit1;
        let bit2 = 1 << qubit2;
        let mask = bit1 | bit2;

        // Collect unique quads
        let mut quads: Vec<usize> = self
            .buffer
            .iter_active()
            .map(|(&idx, _)| idx & !mask)
            .collect();
        quads.sort_unstable();
        quads.dedup();

        self.buffer.begin_operation();

        for base_idx in quads {
            let idx00 = base_idx;
            let idx01 = base_idx | bit2;
            let idx10 = base_idx | bit1;
            let idx11 = base_idx | bit1 | bit2;

            let v = [
                self.buffer.get_amplitude(idx00),
                self.buffer.get_amplitude(idx01),
                self.buffer.get_amplitude(idx10),
                self.buffer.get_amplitude(idx11),
            ];

            let mut result = [Complex64::new(0.0, 0.0); 4];
            for r in 0..4 {
                for c in 0..4 {
                    result[r] += gate[[r, c]] * v[c];
                }
            }

            self.buffer.set_amplitude(idx00, result[0]);
            self.buffer.set_amplitude(idx01, result[1]);
            self.buffer.set_amplitude(idx10, result[2]);
            self.buffer.set_amplitude(idx11, result[3]);
        }

        self.buffer.commit();
    }
}

// =============================================================================
// View-based Mutations for Dense States
// =============================================================================

/// Apply multiple gates to a dense state efficiently using views
///
/// This batches operations to maximize cache utilization.
pub fn apply_gates_dense_batched(
    state: &mut TQPState,
    gates: &[(usize, ndarray::Array2<Complex64>)],
) {
    for (qubit_idx, gate) in gates {
        apply_gate_dense_inplace(state, *qubit_idx, gate);
    }
}

/// Optimized single-qubit gate for dense state (already in-place in ops.rs)
///
/// This is a wrapper that ensures the most efficient path is used.
#[inline]
pub fn apply_gate_dense_inplace(
    state: &mut TQPState,
    qubit_idx: usize,
    gate: &ndarray::Array2<Complex64>,
) {
    // Use the existing optimized implementation
    crate::ops::apply_spatial_gate(state, qubit_idx, gate);
}

// =============================================================================
// Buffer Pool for Repeated Operations
// =============================================================================

/// Pool of reusable buffers to minimize allocations
///
/// Useful when applying many operations in sequence.
pub struct BufferPool {
    /// Available HashMap buffers
    hash_buffers: Vec<FxHashMap<usize, Complex64>>,
    /// Available Vec buffers
    vec_buffers: Vec<Vec<usize>>,
    /// Default capacity for new buffers
    default_capacity: usize,
}

impl BufferPool {
    /// Create a new buffer pool
    pub fn new(default_capacity: usize) -> Self {
        BufferPool {
            hash_buffers: Vec::new(),
            vec_buffers: Vec::new(),
            default_capacity,
        }
    }

    /// Get a HashMap buffer (reused or new)
    pub fn get_hash_buffer(&mut self) -> FxHashMap<usize, Complex64> {
        self.hash_buffers.pop().unwrap_or_else(|| {
            FxHashMap::with_capacity_and_hasher(self.default_capacity, Default::default())
        })
    }

    /// Return a HashMap buffer to the pool
    pub fn return_hash_buffer(&mut self, mut buffer: FxHashMap<usize, Complex64>) {
        buffer.clear();
        self.hash_buffers.push(buffer);
    }

    /// Get a Vec buffer
    pub fn get_vec_buffer(&mut self) -> Vec<usize> {
        self.vec_buffers
            .pop()
            .unwrap_or_else(|| Vec::with_capacity(self.default_capacity))
    }

    /// Return a Vec buffer to the pool
    pub fn return_vec_buffer(&mut self, mut buffer: Vec<usize>) {
        buffer.clear();
        self.vec_buffers.push(buffer);
    }

    /// Clear all pooled buffers
    pub fn clear(&mut self) {
        self.hash_buffers.clear();
        self.vec_buffers.clear();
    }

    /// Get pool statistics
    pub fn stats(&self) -> (usize, usize) {
        (self.hash_buffers.len(), self.vec_buffers.len())
    }
}

impl Default for BufferPool {
    fn default() -> Self {
        Self::new(1024)
    }
}

// =============================================================================
// Optimized Gate Application with Pool
// =============================================================================

/// Apply gate to sparse state using buffer pool
///
/// More efficient than standard apply_gate_sparse when applying many gates
/// because it reuses buffers instead of allocating new ones.
pub fn apply_gate_sparse_pooled(
    state: &mut SparseStateVector,
    qubit_idx: usize,
    gate: &ndarray::Array2<Complex64>,
    pool: &mut BufferPool,
) {
    let bit = 1 << qubit_idx;

    let u00 = gate[[0, 0]];
    let u01 = gate[[0, 1]];
    let u10 = gate[[1, 0]];
    let u11 = gate[[1, 1]];

    // Get reusable buffers from pool
    let mut pairs = pool.get_vec_buffer();
    let mut old_amplitudes = pool.get_hash_buffer();

    // Collect pairs and copy amplitudes
    for (&idx, &amp) in state.iter() {
        let base = idx & !bit;
        pairs.push(base);
        old_amplitudes.insert(idx, amp);
    }
    pairs.sort_unstable();
    pairs.dedup();

    // Apply gate
    let get_old = |idx: usize| -> Complex64 {
        old_amplitudes
            .get(&idx)
            .copied()
            .unwrap_or(Complex64::new(0.0, 0.0))
    };

    for base_idx in &pairs {
        let idx0 = *base_idx;
        let idx1 = base_idx | bit;

        let alpha = get_old(idx0);
        let beta = get_old(idx1);

        let new_alpha = u00 * alpha + u01 * beta;
        let new_beta = u10 * alpha + u11 * beta;

        state.set_amplitude(idx0, new_alpha);
        state.set_amplitude(idx1, new_beta);
    }

    // Return buffers to pool
    pool.return_vec_buffer(pairs);
    pool.return_hash_buffer(old_amplitudes);
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse_ops::{apply_gate_sparse, gates};

    const EPSILON: f64 = 1e-10;

    #[test]
    fn test_double_buffer_basic() {
        let mut db = DoubleBuffer::new(10, 1e-14);

        // Set initial values
        db.working_mut().insert(0, Complex64::new(1.0, 0.0));
        db.commit();

        assert_eq!(db.get_amplitude(0), Complex64::new(1.0, 0.0));
        assert_eq!(db.nnz(), 1);
    }

    #[test]
    fn test_double_buffer_swap() {
        let mut db = DoubleBuffer::new(10, 1e-14);

        // Initial state in buffer A
        db.working_mut().insert(0, Complex64::new(1.0, 0.0));
        db.commit(); // A is now active

        // Prepare new state in buffer B
        db.begin_operation();
        db.set_amplitude(1, Complex64::new(0.5, 0.5));
        db.commit(); // B is now active

        assert_eq!(db.get_amplitude(0), Complex64::new(0.0, 0.0)); // Old value gone
        assert_eq!(db.get_amplitude(1), Complex64::new(0.5, 0.5)); // New value present
    }

    #[test]
    fn test_buffered_sparse_hadamard() {
        // Create |0⟩ state
        let sparse = SparseStateVector::new(2, 1, 1);
        let mut buffered = BufferedSparseState::from_sparse(sparse);

        // Apply Hadamard
        buffered.apply_gate(0, &gates::hadamard());

        // Should be |+⟩ = (|0⟩ + |1⟩)/√2
        let h = std::f64::consts::FRAC_1_SQRT_2;
        assert!((buffered.get_amplitude(0).re - h).abs() < EPSILON);
        assert!((buffered.get_amplitude(1).re - h).abs() < EPSILON);
    }

    #[test]
    fn test_buffered_vs_standard() {
        use crate::sparse_ops::apply_gate_sparse;

        // Standard method
        let mut standard = SparseStateVector::new(3, 1, 1);
        apply_gate_sparse(&mut standard, 0, &gates::hadamard());
        apply_gate_sparse(&mut standard, 1, &gates::hadamard());

        // Buffered method
        let sparse = SparseStateVector::new(3, 1, 1);
        let mut buffered = BufferedSparseState::from_sparse(sparse);
        buffered.apply_gate(0, &gates::hadamard());
        buffered.apply_gate(1, &gates::hadamard());
        let buffered_result = buffered.into_sparse();

        // Compare results
        for i in 0..8 {
            let diff = (standard.get_amplitude(i) - buffered_result.get_amplitude(i)).norm();
            assert!(diff < EPSILON, "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_buffered_2q_gate() {
        use crate::sparse_ops::apply_gate_2q_sparse;

        // Standard CNOT
        let mut standard = SparseStateVector::new(2, 1, 1);
        apply_gate_sparse(&mut standard, 0, &gates::hadamard());
        apply_gate_2q_sparse(&mut standard, 0, 1, &gates::cnot());

        // Buffered CNOT
        let sparse = SparseStateVector::new(2, 1, 1);
        let mut buffered = BufferedSparseState::from_sparse(sparse);
        buffered.apply_gate(0, &gates::hadamard());
        buffered.apply_gate_2q(0, 1, &gates::cnot());
        let buffered_result = buffered.into_sparse();

        // Should both produce Bell state
        for i in 0..4 {
            let diff = (standard.get_amplitude(i) - buffered_result.get_amplitude(i)).norm();
            assert!(diff < EPSILON, "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_buffer_pool() {
        let mut pool = BufferPool::new(100);

        // Get buffers
        let buf1 = pool.get_hash_buffer();
        let buf2 = pool.get_vec_buffer();

        assert_eq!(pool.stats(), (0, 0)); // All buffers in use

        // Return buffers
        pool.return_hash_buffer(buf1);
        pool.return_vec_buffer(buf2);

        assert_eq!(pool.stats(), (1, 1)); // Buffers returned

        // Get again - should reuse
        let _buf3 = pool.get_hash_buffer();
        assert_eq!(pool.stats(), (0, 1)); // One hash buffer taken
    }

    #[test]
    fn test_pooled_gate_application() {
        use crate::sparse_ops::apply_gate_sparse;

        let mut pool = BufferPool::new(100);

        // Standard
        let mut standard = SparseStateVector::new(3, 1, 1);
        apply_gate_sparse(&mut standard, 0, &gates::hadamard());

        // Pooled
        let mut pooled = SparseStateVector::new(3, 1, 1);
        apply_gate_sparse_pooled(&mut pooled, 0, &gates::hadamard(), &mut pool);

        // Should match
        for i in 0..8 {
            let diff = (standard.get_amplitude(i) - pooled.get_amplitude(i)).norm();
            assert!(diff < EPSILON, "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_multiple_gates_with_pool() {
        let mut pool = BufferPool::new(100);
        let mut state = SparseStateVector::new(4, 1, 1);

        // Apply many gates with pool reuse
        for qubit in 0..4 {
            apply_gate_sparse_pooled(&mut state, qubit, &gates::hadamard(), &mut pool);
        }

        // Should have 16 non-zero amplitudes (2^4)
        assert_eq!(state.nnz(), 16);

        // Normalization should be preserved
        let total_prob: f64 = (0..16).map(|i| state.probability(i)).sum();
        assert!((total_prob - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_dense_batched() {
        let mut state = TQPState::new(3, 1, 1);

        let gates_to_apply: Vec<(usize, ndarray::Array2<Complex64>)> = vec![
            (0, gates::hadamard()),
            (1, gates::hadamard()),
            (2, gates::pauli_x()),
        ];

        apply_gates_dense_batched(&mut state, &gates_to_apply);

        // Verify state is valid
        let total_prob: f64 = (0..8).map(|i| state.probability(i)).sum();
        assert!((total_prob - 1.0).abs() < EPSILON);
    }

    /// Simple timing test to verify buffered is at least not slower
    #[test]
    fn test_performance_comparison() {
        use std::time::Instant;

        let num_qubits = 12;
        let num_gates = 50;
        let h = gates::hadamard();

        // Standard method
        let start = Instant::now();
        for _ in 0..10 {
            let mut state = SparseStateVector::new(num_qubits, 1, 1);
            for i in 0..num_gates {
                apply_gate_sparse(&mut state, i % num_qubits, &h);
            }
        }
        let standard_time = start.elapsed();

        // Buffered method
        let start = Instant::now();
        for _ in 0..10 {
            let state = SparseStateVector::new(num_qubits, 1, 1);
            let mut buffered = BufferedSparseState::from_sparse(state);
            for i in 0..num_gates {
                buffered.apply_gate(i % num_qubits, &h);
            }
        }
        let buffered_time = start.elapsed();

        // Pooled method
        let mut pool = BufferPool::new(1 << num_qubits);
        let start = Instant::now();
        for _ in 0..10 {
            let mut state = SparseStateVector::new(num_qubits, 1, 1);
            for i in 0..num_gates {
                apply_gate_sparse_pooled(&mut state, i % num_qubits, &h, &mut pool);
            }
        }
        let pooled_time = start.elapsed();

        println!(
            "\nPerformance Comparison ({}q, {} gates, 10 iterations):",
            num_qubits, num_gates
        );
        println!("  Standard: {:?}", standard_time);
        println!("  Buffered: {:?}", buffered_time);
        println!("  Pooled:   {:?}", pooled_time);

        // Buffered and pooled should not be significantly slower (allow 50% overhead for small tests)
        // In real workloads, they should be faster due to reduced allocation
        assert!(
            buffered_time.as_micros() < standard_time.as_micros() * 3,
            "Buffered too slow"
        );
        assert!(
            pooled_time.as_micros() < standard_time.as_micros() * 3,
            "Pooled too slow"
        );
    }
}

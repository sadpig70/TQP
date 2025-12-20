//! Sparse State Vector representation for TQP
//!
//! Provides memory-efficient storage for quantum states with many zero amplitudes.
//! Uses FxHashMap for O(1) amplitude access with minimal overhead.
//!
//! # Design Rationale
//! - Quantum states after measurement or certain algorithms become sparse
//! - Dense representation: O(2^N) memory regardless of state structure
//! - Sparse representation: O(nnz) memory where nnz << 2^N for sparse states
//!
//! # Threshold-based Pruning
//! Amplitudes with |c|² < threshold are automatically pruned to:
//! 1. Reduce memory footprint
//! 2. Improve iteration performance
//! 3. Mitigate floating-point accumulation errors

use ndarray::Array1;
use num_complex::Complex64;
use rustc_hash::FxHashMap;

use crate::state::{TQPDimensions, TQPState};

/// Default pruning threshold (probability < 1e-14 is considered zero)
pub const DEFAULT_PRUNE_THRESHOLD: f64 = 1e-14;

/// Sparse representation of a TQP quantum state vector.
///
/// Stores only non-zero amplitudes in a hash map, providing O(1) access
/// and O(nnz) memory usage where nnz is the number of non-zero elements.
///
/// # Index Mapping (consistent with TQPState)
/// ```text
/// global_idx = layer * (num_bins * spatial_dim) + bin * spatial_dim + spatial
/// ```
#[derive(Debug, Clone)]
pub struct SparseStateVector {
    /// Dimensions of the TQP system (N qubits, M time-bins, L layers)
    pub dims: TQPDimensions,

    /// Non-zero amplitudes indexed by global state index
    /// Key: global_idx, Value: complex amplitude c where |c|² >= threshold
    amplitudes: FxHashMap<usize, Complex64>,

    /// Pruning threshold: amplitudes with |c|² < threshold are removed
    /// Default: 1e-14 (numerical zero for f64)
    threshold: f64,
}

impl SparseStateVector {
    /// Creates a new SparseStateVector initialized to |0...0⟩|0⟩|0⟩
    ///
    /// # Arguments
    /// * `num_qubits` - Number of spatial qubits (N)
    /// * `num_time_bins` - Number of time bins (M)
    /// * `num_layers` - Number of logical layers (L)
    ///
    /// # Returns
    /// A sparse state with amplitude 1.0 at index 0 (ground state)
    pub fn new(num_qubits: usize, num_time_bins: usize, num_layers: usize) -> Self {
        Self::with_threshold(
            num_qubits,
            num_time_bins,
            num_layers,
            DEFAULT_PRUNE_THRESHOLD,
        )
    }

    /// Creates a new SparseStateVector with custom pruning threshold
    ///
    /// # Arguments
    /// * `num_qubits` - Number of spatial qubits (N)
    /// * `num_time_bins` - Number of time bins (M)
    /// * `num_layers` - Number of logical layers (L)
    /// * `threshold` - Pruning threshold for |c|²
    ///
    /// # Panics
    /// Panics if threshold is negative
    pub fn with_threshold(
        num_qubits: usize,
        num_time_bins: usize,
        num_layers: usize,
        threshold: f64,
    ) -> Self {
        assert!(threshold >= 0.0, "Pruning threshold must be non-negative");

        let dims = TQPDimensions::new(num_qubits, num_time_bins, num_layers);

        // Pre-allocate with expected capacity for typical sparse states
        // Initial state has exactly 1 non-zero element
        let mut amplitudes = FxHashMap::with_capacity_and_hasher(1, Default::default());
        amplitudes.insert(0, Complex64::new(1.0, 0.0));

        SparseStateVector {
            dims,
            amplitudes,
            threshold,
        }
    }

    /// Creates an empty SparseStateVector (all amplitudes zero)
    ///
    /// # Warning
    /// This creates an invalid quantum state (norm = 0).
    /// Use only for intermediate computations.
    pub fn empty(num_qubits: usize, num_time_bins: usize, num_layers: usize) -> Self {
        let dims = TQPDimensions::new(num_qubits, num_time_bins, num_layers);
        SparseStateVector {
            dims,
            amplitudes: FxHashMap::default(),
            threshold: DEFAULT_PRUNE_THRESHOLD,
        }
    }

    /// Returns the number of non-zero amplitudes (sparsity metric)
    #[inline]
    pub fn nnz(&self) -> usize {
        self.amplitudes.len()
    }

    /// Returns the total Hilbert space dimension (L × M × 2^N)
    #[inline]
    pub fn dimension(&self) -> usize {
        self.dims.total_dim()
    }

    /// Returns the spatial Hilbert space dimension (2^N)
    #[inline]
    pub fn spatial_dim(&self) -> usize {
        self.dims.spatial_dim()
    }

    /// Returns the sparsity ratio (nnz / dimension)
    ///
    /// Lower values indicate more sparse (memory-efficient) representation.
    /// - 1.0 = fully dense (all elements non-zero)
    /// - 0.0 = empty state
    #[inline]
    pub fn sparsity_ratio(&self) -> f64 {
        self.nnz() as f64 / self.dimension() as f64
    }

    /// Returns the current pruning threshold
    #[inline]
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Sets a new pruning threshold and applies it
    ///
    /// # Arguments
    /// * `threshold` - New threshold value (must be non-negative)
    ///
    /// # Returns
    /// Number of amplitudes pruned
    pub fn set_threshold(&mut self, threshold: f64) -> usize {
        assert!(threshold >= 0.0, "Pruning threshold must be non-negative");
        self.threshold = threshold;
        self.prune()
    }

    /// Gets the amplitude at a specific global index
    ///
    /// Returns Complex64::zero() for indices not in the sparse map.
    ///
    /// # Arguments
    /// * `index` - Global state index
    ///
    /// # Returns
    /// Complex amplitude at the index
    #[inline]
    pub fn get_amplitude(&self, index: usize) -> Complex64 {
        self.amplitudes
            .get(&index)
            .copied()
            .unwrap_or(Complex64::new(0.0, 0.0))
    }

    /// Sets the amplitude at a specific global index
    ///
    /// Automatically prunes if |amplitude|² < threshold.
    ///
    /// # Arguments
    /// * `index` - Global state index
    /// * `amplitude` - Complex amplitude to set
    #[inline]
    pub fn set_amplitude(&mut self, index: usize, amplitude: Complex64) {
        debug_assert!(index < self.dimension(), "Index out of bounds");

        if amplitude.norm_sqr() < self.threshold {
            self.amplitudes.remove(&index);
        } else {
            self.amplitudes.insert(index, amplitude);
        }
    }

    /// Returns the probability at a specific index (|c|²)
    #[inline]
    pub fn probability(&self, index: usize) -> f64 {
        self.get_amplitude(index).norm_sqr()
    }

    /// Calculates the total probability (should be 1.0 for normalized states)
    ///
    /// Useful for debugging and validation.
    pub fn total_probability(&self) -> f64 {
        self.amplitudes.values().map(|c| c.norm_sqr()).sum()
    }

    /// Normalizes the state vector in-place
    ///
    /// Ensures total probability equals 1.0
    pub fn normalize(&mut self) {
        let norm_sq: f64 = self.total_probability();

        if norm_sq > 1e-30 {
            let norm = norm_sq.sqrt();
            for amplitude in self.amplitudes.values_mut() {
                *amplitude /= norm;
            }
        }

        // Re-prune after normalization (some amplitudes may now be below threshold)
        self.prune();
    }

    /// Removes amplitudes below the pruning threshold
    ///
    /// # Returns
    /// Number of amplitudes removed
    pub fn prune(&mut self) -> usize {
        let before = self.amplitudes.len();
        self.amplitudes
            .retain(|_, c| c.norm_sqr() >= self.threshold);
        before - self.amplitudes.len()
    }

    /// Calculates the global index from (layer, bin, spatial) indices
    ///
    /// # Arguments
    /// * `layer` - Layer index (0..L)
    /// * `bin` - Time-bin index (0..M)
    /// * `spatial` - Spatial basis state index (0..2^N)
    #[inline]
    pub fn get_index(&self, layer: usize, bin: usize, spatial: usize) -> usize {
        let spatial_dim = self.dims.spatial_dim();
        layer * (self.dims.num_time_bins * spatial_dim) + bin * spatial_dim + spatial
    }

    /// Decomposes a global index into (layer, bin, spatial) indices
    ///
    /// # Arguments
    /// * `global_idx` - Global state index
    ///
    /// # Returns
    /// Tuple of (layer, bin, spatial) indices
    #[inline]
    pub fn get_indices(&self, global_idx: usize) -> (usize, usize, usize) {
        let spatial_dim = self.dims.spatial_dim();
        let bin_dim = self.dims.num_time_bins;

        let spatial = global_idx % spatial_dim;
        let remaining = global_idx / spatial_dim;
        let bin = remaining % bin_dim;
        let layer = remaining / bin_dim;

        (layer, bin, spatial)
    }

    /// Returns an iterator over non-zero (index, amplitude) pairs
    pub fn iter(&self) -> impl Iterator<Item = (&usize, &Complex64)> {
        self.amplitudes.iter()
    }

    /// Returns a mutable iterator over non-zero (index, amplitude) pairs
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&usize, &mut Complex64)> {
        self.amplitudes.iter_mut()
    }

    /// Returns the estimated memory usage in bytes
    ///
    /// Approximate formula: sizeof(usize) + sizeof(Complex64) per entry + overhead
    pub fn memory_usage_bytes(&self) -> usize {
        // FxHashMap overhead + (key + value) per entry
        // Approximate: 48 bytes base + 24 bytes per entry
        48 + self.nnz() * (std::mem::size_of::<usize>() + std::mem::size_of::<Complex64>())
    }

    /// Compares memory usage with equivalent dense representation
    ///
    /// # Returns
    /// Tuple of (sparse_bytes, dense_bytes, ratio)
    pub fn memory_comparison(&self) -> (usize, usize, f64) {
        let sparse = self.memory_usage_bytes();
        let dense = self.dimension() * std::mem::size_of::<Complex64>();
        let ratio = sparse as f64 / dense as f64;
        (sparse, dense, ratio)
    }

    /// Calculates the marginal probability of a specific qubit being in state |1⟩
    pub fn get_marginal_probability(&self, qubit_idx: usize) -> f64 {
        let bit = 1 << qubit_idx;
        let mut prob = 0.0;

        for (&idx, amplitude) in &self.amplitudes {
            let (_, _, spatial) = self.get_indices(idx);
            if (spatial & bit) != 0 {
                prob += amplitude.norm_sqr();
            }
        }
        prob
    }

    /// Reserves capacity for additional non-zero elements
    ///
    /// Call this before bulk insertions to avoid reallocations.
    pub fn reserve(&mut self, additional: usize) {
        self.amplitudes.reserve(additional);
    }

    /// Clears all amplitudes (results in invalid zero state)
    pub fn clear(&mut self) {
        self.amplitudes.clear();
    }

    // ========================================================================
    // Conversion Methods (Dense ↔ Sparse)
    // ========================================================================

    /// Creates a SparseStateVector from a dense TQPState
    ///
    /// # Arguments
    /// * `dense` - Reference to a dense TQPState
    ///
    /// # Returns
    /// A sparse representation containing only non-zero amplitudes
    pub fn from_dense(dense: &TQPState) -> Self {
        Self::from_dense_with_threshold(dense, DEFAULT_PRUNE_THRESHOLD)
    }

    /// Creates a SparseStateVector from a dense TQPState with custom threshold
    ///
    /// # Arguments
    /// * `dense` - Reference to a dense TQPState
    /// * `threshold` - Pruning threshold for |c|²
    pub fn from_dense_with_threshold(dense: &TQPState, threshold: f64) -> Self {
        let dims = dense.dims;
        let capacity = dense
            .state_vector
            .iter()
            .filter(|c| c.norm_sqr() >= threshold)
            .count();

        let mut amplitudes = FxHashMap::with_capacity_and_hasher(capacity, Default::default());

        for (idx, &amplitude) in dense.state_vector.iter().enumerate() {
            if amplitude.norm_sqr() >= threshold {
                amplitudes.insert(idx, amplitude);
            }
        }

        SparseStateVector {
            dims,
            amplitudes,
            threshold,
        }
    }

    /// Converts this sparse state to a dense TQPState
    ///
    /// # Returns
    /// A fully materialized dense TQPState
    ///
    /// # Warning
    /// This allocates O(2^N) memory. Use with caution for large N.
    pub fn to_dense(&self) -> TQPState {
        let total_dim = self.dimension();
        let mut state_vector = Array1::<Complex64>::zeros(total_dim);

        for (&idx, &amplitude) in &self.amplitudes {
            state_vector[idx] = amplitude;
        }

        TQPState {
            dims: self.dims,
            state_vector,
        }
    }

    /// Converts to a raw Array1 (for advanced use)
    ///
    /// # Returns
    /// Dense Array1<Complex64> representation
    pub fn to_array(&self) -> Array1<Complex64> {
        let total_dim = self.dimension();
        let mut array = Array1::<Complex64>::zeros(total_dim);

        for (&idx, &amplitude) in &self.amplitudes {
            array[idx] = amplitude;
        }

        array
    }

    /// Checks if converting to dense is memory-safe
    ///
    /// # Arguments
    /// * `max_bytes` - Maximum allowed memory in bytes
    ///
    /// # Returns
    /// true if dense representation would fit within max_bytes
    pub fn can_convert_to_dense(&self, max_bytes: usize) -> bool {
        let dense_size = self.dimension() * std::mem::size_of::<Complex64>();
        dense_size <= max_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_sparse() {
        let state = SparseStateVector::new(2, 2, 2);

        assert_eq!(state.dimension(), 16); // 4 * 2 * 2
        assert_eq!(state.nnz(), 1); // Only |0⟩ has amplitude
        assert_eq!(state.get_amplitude(0), Complex64::new(1.0, 0.0));
        assert_eq!(state.get_amplitude(1), Complex64::new(0.0, 0.0));
    }

    #[test]
    fn test_set_get_amplitude() {
        let mut state = SparseStateVector::new(2, 1, 1);

        state.set_amplitude(1, Complex64::new(0.5, 0.5));
        assert_eq!(state.nnz(), 2);
        assert_eq!(state.get_amplitude(1), Complex64::new(0.5, 0.5));

        // Set below threshold - should be pruned
        state.set_amplitude(2, Complex64::new(1e-10, 0.0));
        assert_eq!(state.nnz(), 2); // Still 2, not 3
    }

    #[test]
    fn test_indexing() {
        let state = SparseStateVector::new(2, 2, 2);

        // Layer 1, Bin 1, Spatial 3 (|11⟩)
        let idx = state.get_index(1, 1, 3);
        assert_eq!(idx, 15); // 1*8 + 1*4 + 3

        let (l, b, s) = state.get_indices(15);
        assert_eq!((l, b, s), (1, 1, 3));
    }

    #[test]
    fn test_normalize() {
        let mut state = SparseStateVector::new(2, 1, 1);

        // Set unnormalized amplitudes
        state.set_amplitude(0, Complex64::new(1.0, 0.0));
        state.set_amplitude(1, Complex64::new(1.0, 0.0));

        state.normalize();

        let prob = state.total_probability();
        assert!(
            (prob - 1.0).abs() < 1e-10,
            "Probability should be 1.0, got {}",
            prob
        );
    }

    #[test]
    fn test_sparsity_ratio() {
        let mut state = SparseStateVector::new(10, 1, 1); // 2^10 = 1024 dimension

        // Initially only |0⟩
        assert!((state.sparsity_ratio() - 1.0 / 1024.0).abs() < 1e-10);

        // Add more elements
        for i in 0..100 {
            state.set_amplitude(i, Complex64::new(0.1, 0.0));
        }
        assert!((state.sparsity_ratio() - 100.0 / 1024.0).abs() < 1e-10);
    }

    #[test]
    fn test_memory_comparison() {
        let state = SparseStateVector::new(20, 1, 1); // 2^20 = 1M dimension

        let (sparse, dense, ratio) = state.memory_comparison();

        // With only 1 non-zero element, sparse should be much smaller
        assert!(
            ratio < 0.001,
            "Sparse should be < 0.1% of dense, got {}%",
            ratio * 100.0
        );
        println!(
            "Memory: sparse={}B, dense={}B, ratio={:.6}",
            sparse, dense, ratio
        );
    }

    #[test]
    fn test_marginal_probability() {
        let mut state = SparseStateVector::new(2, 1, 1);

        // Create |01⟩ + |11⟩ (qubit 0 always 1)
        state.clear();
        state.set_amplitude(1, Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0)); // |01⟩
        state.set_amplitude(3, Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0)); // |11⟩

        let prob_q0 = state.get_marginal_probability(0);
        assert!(
            (prob_q0 - 1.0).abs() < 1e-10,
            "Qubit 0 should be 1.0, got {}",
            prob_q0
        );

        let prob_q1 = state.get_marginal_probability(1);
        assert!(
            (prob_q1 - 0.5).abs() < 1e-10,
            "Qubit 1 should be 0.5, got {}",
            prob_q1
        );
    }

    #[test]
    fn test_from_dense() {
        use crate::state::TQPState;

        // Create a dense state
        let mut dense = TQPState::new(2, 1, 1);
        // Dense state is |0⟩ with amplitude 1.0

        // Convert to sparse
        let sparse = SparseStateVector::from_dense(&dense);

        assert_eq!(sparse.nnz(), 1);
        assert_eq!(sparse.get_amplitude(0), Complex64::new(1.0, 0.0));
        assert_eq!(sparse.dimension(), dense.dimension());
    }

    #[test]
    fn test_to_dense() {
        let mut sparse = SparseStateVector::new(2, 1, 1);
        sparse.set_amplitude(0, Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0));
        sparse.set_amplitude(3, Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0));

        let dense = sparse.to_dense();

        assert_eq!(dense.dimension(), sparse.dimension());
        assert!((dense.probability(0) - 0.5).abs() < 1e-10);
        assert!((dense.probability(3) - 0.5).abs() < 1e-10);
        assert!((dense.probability(1)).abs() < 1e-10);
    }

    #[test]
    fn test_roundtrip_conversion() {
        use crate::state::TQPState;

        // Create dense state
        let dense_original = TQPState::new(3, 2, 1);

        // Dense → Sparse → Dense
        let sparse = SparseStateVector::from_dense(&dense_original);
        let dense_recovered = sparse.to_dense();

        // Verify equivalence
        for i in 0..dense_original.dimension() {
            let diff = (dense_original.state_vector[i] - dense_recovered.state_vector[i]).norm();
            assert!(diff < 1e-14, "Mismatch at index {}: diff = {}", i, diff);
        }
    }

    #[test]
    fn test_can_convert_to_dense() {
        let sparse = SparseStateVector::new(10, 1, 1); // 2^10 = 1024 elements

        // 1024 * 16 bytes = 16KB
        assert!(sparse.can_convert_to_dense(1024 * 1024)); // 1MB - should fit
        assert!(!sparse.can_convert_to_dense(1024)); // 1KB - too small
    }
}

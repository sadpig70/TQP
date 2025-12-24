//! Unified Dispatcher for TQP Quantum Operations
//!
//! Automatically selects the optimal backend based on:
//! - State representation (Dense vs Sparse)
//! - CPU features (SIMD vs Scalar)
//! - State size (Parallel vs Sequential)
//!
//! # Backend Selection Strategy
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                    Dispatcher                            │
//! ├─────────────────────────────────────────────────────────┤
//! │  1. State Type Selection (Dense/Sparse)                 │
//! │     - sparsity_ratio = nnz / total_dim                  │
//! │     - if ratio < SPARSE_THRESHOLD → Sparse              │
//! │     - else → Dense                                      │
//! ├─────────────────────────────────────────────────────────┤
//! │  2. Execution Selection (SIMD/Parallel/Scalar)          │
//! │     - if nnz > PARALLEL_THRESHOLD → Parallel            │
//! │     - elif AVX2 available && Dense → SIMD               │
//! │     - else → Scalar                                     │
//! └─────────────────────────────────────────────────────────┘
//! ```

use ndarray::Array2;
use num_complex::Complex64;

use crate::ops::{apply_spatial_gate, apply_spatial_gate_2q};
use crate::parallel_sparse::{parallel_apply_gate_2q_sparse, parallel_apply_gate_sparse};
use crate::simd_avx2::{apply_gate_1q_simd, apply_gate_2q_simd, has_avx2};
use crate::sparse::SparseStateVector;
use crate::sparse_ops::{apply_gate_2q_sparse, apply_gate_sparse, gates};
use crate::state::TQPState;

// =============================================================================
// Configuration Constants
// =============================================================================

/// Sparsity ratio threshold: if nnz/total < this, use sparse representation
/// Based on empirical testing: sparse overhead pays off below ~1% density
pub const SPARSE_THRESHOLD: f64 = 0.01;

/// State size threshold for parallel execution
/// Based on Day 1 results: parallelization effective for nnz >= 8192
pub const PARALLEL_THRESHOLD: usize = 8192;

/// Minimum state dimension to consider SIMD (very small states have more overhead)
pub const SIMD_MIN_DIM: usize = 64;

// =============================================================================
// Backend Selection
// =============================================================================

/// Backend type for state representation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateBackend {
    /// Dense representation (Array1<Complex64>)
    Dense,
    /// Sparse representation (FxHashMap<usize, Complex64>)
    Sparse,
}

/// Execution backend for operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionBackend {
    /// Scalar (no SIMD, no parallelization)
    Scalar,
    /// SIMD optimized (AVX2/AVX-512)
    Simd,
    /// Parallel execution (Rayon)
    Parallel,
    /// Parallel + SIMD (future)
    ParallelSimd,
}

/// Combined backend selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Backend {
    pub state: StateBackend,
    pub execution: ExecutionBackend,
}

impl Backend {
    pub fn new(state: StateBackend, execution: ExecutionBackend) -> Self {
        Backend { state, execution }
    }

    /// Select optimal backend for a dense state
    pub fn for_dense(state: &TQPState) -> Self {
        let dim = state.dimension();

        let execution = if dim >= PARALLEL_THRESHOLD {
            // Large state: consider parallelization
            // But for dense, SIMD is often better than parallelization
            if has_avx2() && dim >= SIMD_MIN_DIM {
                ExecutionBackend::Simd
            } else {
                ExecutionBackend::Scalar
            }
        } else if has_avx2() && dim >= SIMD_MIN_DIM {
            ExecutionBackend::Simd
        } else {
            ExecutionBackend::Scalar
        };

        Backend::new(StateBackend::Dense, execution)
    }

    /// Select optimal backend for a sparse state
    pub fn for_sparse(state: &SparseStateVector) -> Self {
        let nnz = state.nnz();
        let total = state.dimension();
        let density = nnz as f64 / total as f64;

        // Check if we should convert to dense
        if density > SPARSE_THRESHOLD {
            // High density: recommend dense
            let execution = if has_avx2() && total >= SIMD_MIN_DIM {
                ExecutionBackend::Simd
            } else {
                ExecutionBackend::Scalar
            };
            return Backend::new(StateBackend::Dense, execution);
        }

        // Sparse is appropriate
        let execution = if nnz >= PARALLEL_THRESHOLD {
            ExecutionBackend::Parallel
        } else {
            ExecutionBackend::Scalar
        };

        Backend::new(StateBackend::Sparse, execution)
    }

    /// Get descriptive string
    pub fn description(&self) -> String {
        format!("{:?} + {:?}", self.state, self.execution)
    }
}

/// Select backend based on sparsity ratio
pub fn select_backend(nnz: usize, total_dim: usize) -> Backend {
    let density = nnz as f64 / total_dim as f64;

    let state_backend = if density < SPARSE_THRESHOLD {
        StateBackend::Sparse
    } else {
        StateBackend::Dense
    };

    let execution_backend = match state_backend {
        StateBackend::Sparse => {
            if nnz >= PARALLEL_THRESHOLD {
                ExecutionBackend::Parallel
            } else {
                ExecutionBackend::Scalar
            }
        }
        StateBackend::Dense => {
            if has_avx2() && total_dim >= SIMD_MIN_DIM {
                ExecutionBackend::Simd
            } else {
                ExecutionBackend::Scalar
            }
        }
    };

    Backend::new(state_backend, execution_backend)
}

// =============================================================================
// Unified State Wrapper
// =============================================================================

/// Unified quantum state that can be either dense or sparse
#[derive(Debug, Clone)]
pub enum QuantumState {
    Dense(TQPState),
    Sparse(SparseStateVector),
}

impl QuantumState {
    /// Create a new dense state
    pub fn new_dense(num_qubits: usize, num_time_bins: usize, num_layers: usize) -> Self {
        QuantumState::Dense(TQPState::new(num_qubits, num_time_bins, num_layers))
    }

    /// Create a new sparse state
    pub fn new_sparse(num_qubits: usize, num_time_bins: usize, num_layers: usize) -> Self {
        QuantumState::Sparse(SparseStateVector::new(
            num_qubits,
            num_time_bins,
            num_layers,
        ))
    }

    /// Get the number of non-zero elements
    pub fn nnz(&self) -> usize {
        match self {
            QuantumState::Dense(s) => s
                .state_vector
                .iter()
                .filter(|c| c.norm_sqr() > 1e-14)
                .count(),
            QuantumState::Sparse(s) => s.nnz(),
        }
    }

    /// Get total dimension
    pub fn total_dim(&self) -> usize {
        match self {
            QuantumState::Dense(s) => s.dimension(),
            QuantumState::Sparse(s) => s.dimension(),
        }
    }

    /// Get density ratio
    pub fn density(&self) -> f64 {
        self.nnz() as f64 / self.total_dim() as f64
    }

    /// Check if sparse representation
    pub fn is_sparse(&self) -> bool {
        matches!(self, QuantumState::Sparse(_))
    }

    /// Convert to dense
    pub fn to_dense(&self) -> TQPState {
        match self {
            QuantumState::Dense(s) => s.clone(),
            QuantumState::Sparse(s) => s.to_dense(),
        }
    }

    /// Convert to sparse
    pub fn to_sparse(&self) -> SparseStateVector {
        match self {
            QuantumState::Dense(s) => SparseStateVector::from_dense(s),
            QuantumState::Sparse(s) => s.clone(),
        }
    }

    /// Get recommended backend
    pub fn recommended_backend(&self) -> Backend {
        select_backend(self.nnz(), self.total_dim())
    }

    /// Get amplitude at index
    pub fn get(&self, index: usize) -> Complex64 {
        match self {
            QuantumState::Dense(s) => {
                if index < s.dimension() {
                    s.state_vector[index]
                } else {
                    Complex64::new(0.0, 0.0)
                }
            }
            QuantumState::Sparse(s) => s.get_amplitude(index),
        }
    }
}

// =============================================================================
// Dispatcher Functions
// =============================================================================

/// Apply 1-qubit gate with automatic backend selection
pub fn dispatch_gate_1q(state: &mut QuantumState, qubit: usize, gate: &Array2<Complex64>) {
    match state {
        QuantumState::Dense(s) => {
            // Dense state: use SIMD if available
            if has_avx2() && s.dimension() >= SIMD_MIN_DIM {
                apply_gate_1q_simd(s, qubit, gate);
            } else {
                apply_spatial_gate(s, qubit, gate);
            }
        }
        QuantumState::Sparse(s) => {
            // Sparse state: use parallel if large enough
            if s.nnz() >= PARALLEL_THRESHOLD {
                parallel_apply_gate_sparse(s, qubit, gate);
            } else {
                apply_gate_sparse(s, qubit, gate);
            }
        }
    }
}

/// Apply 2-qubit gate with automatic backend selection
pub fn dispatch_gate_2q(
    state: &mut QuantumState,
    qubit1: usize,
    qubit2: usize,
    gate: &Array2<Complex64>,
) {
    match state {
        QuantumState::Dense(s) => {
            // Dense state: use SIMD if available
            if has_avx2() && s.dimension() >= SIMD_MIN_DIM {
                apply_gate_2q_simd(s, qubit1, qubit2, gate);
            } else {
                apply_spatial_gate_2q(s, qubit1, qubit2, gate);
            }
        }
        QuantumState::Sparse(s) => {
            // Sparse state: use parallel if large enough
            if s.nnz() >= PARALLEL_THRESHOLD {
                parallel_apply_gate_2q_sparse(s, qubit1, qubit2, gate);
            } else {
                apply_gate_2q_sparse(s, qubit1, qubit2, gate);
            }
        }
    }
}

/// Apply a circuit (sequence of gates) with automatic backend selection
pub fn dispatch_circuit(state: &mut QuantumState, circuit: &[CircuitGate]) {
    for op in circuit {
        match op {
            CircuitGate::Gate1Q { qubit, gate } => {
                dispatch_gate_1q(state, *qubit, gate);
            }
            CircuitGate::Gate2Q {
                qubit1,
                qubit2,
                gate,
            } => {
                dispatch_gate_2q(state, *qubit1, *qubit2, gate);
            }
        }
    }
}

/// Gate operation for circuit
#[derive(Debug, Clone)]
pub enum CircuitGate {
    Gate1Q {
        qubit: usize,
        gate: Array2<Complex64>,
    },
    Gate2Q {
        qubit1: usize,
        qubit2: usize,
        gate: Array2<Complex64>,
    },
}

impl CircuitGate {
    pub fn h(qubit: usize) -> Self {
        CircuitGate::Gate1Q {
            qubit,
            gate: gates::hadamard(),
        }
    }

    pub fn x(qubit: usize) -> Self {
        CircuitGate::Gate1Q {
            qubit,
            gate: gates::pauli_x(),
        }
    }

    pub fn y(qubit: usize) -> Self {
        CircuitGate::Gate1Q {
            qubit,
            gate: gates::pauli_y(),
        }
    }

    pub fn z(qubit: usize) -> Self {
        CircuitGate::Gate1Q {
            qubit,
            gate: gates::pauli_z(),
        }
    }

    pub fn cnot(control: usize, target: usize) -> Self {
        CircuitGate::Gate2Q {
            qubit1: control,
            qubit2: target,
            gate: gates::cnot(),
        }
    }

    pub fn cz(qubit1: usize, qubit2: usize) -> Self {
        CircuitGate::Gate2Q {
            qubit1,
            qubit2,
            gate: gates::cz(),
        }
    }

    pub fn swap(qubit1: usize, qubit2: usize) -> Self {
        CircuitGate::Gate2Q {
            qubit1,
            qubit2,
            gate: gates::swap(),
        }
    }
}

// =============================================================================
// Dispatcher Configuration
// =============================================================================

/// Configuration for the dispatcher
#[derive(Debug, Clone)]
pub struct DispatcherConfig {
    /// Threshold for switching to sparse representation
    pub sparse_threshold: f64,
    /// Threshold for parallel execution
    pub parallel_threshold: usize,
    /// Force specific state backend (None = auto)
    pub force_state_backend: Option<StateBackend>,
    /// Force specific execution backend (None = auto)
    pub force_execution_backend: Option<ExecutionBackend>,
}

impl Default for DispatcherConfig {
    fn default() -> Self {
        DispatcherConfig {
            sparse_threshold: SPARSE_THRESHOLD,
            parallel_threshold: PARALLEL_THRESHOLD,
            force_state_backend: None,
            force_execution_backend: None,
        }
    }
}

impl DispatcherConfig {
    /// Create config that forces dense representation
    pub fn force_dense() -> Self {
        DispatcherConfig {
            force_state_backend: Some(StateBackend::Dense),
            ..Default::default()
        }
    }

    /// Create config that forces sparse representation
    pub fn force_sparse() -> Self {
        DispatcherConfig {
            force_state_backend: Some(StateBackend::Sparse),
            ..Default::default()
        }
    }

    /// Create config that forces SIMD execution
    pub fn force_simd() -> Self {
        DispatcherConfig {
            force_execution_backend: Some(ExecutionBackend::Simd),
            ..Default::default()
        }
    }

    /// Create config that forces parallel execution
    pub fn force_parallel() -> Self {
        DispatcherConfig {
            force_execution_backend: Some(ExecutionBackend::Parallel),
            ..Default::default()
        }
    }

    /// Select backend with this configuration
    pub fn select_backend(&self, nnz: usize, total_dim: usize) -> Backend {
        let density = nnz as f64 / total_dim as f64;

        let state_backend = self.force_state_backend.unwrap_or({
            if density < self.sparse_threshold {
                StateBackend::Sparse
            } else {
                StateBackend::Dense
            }
        });

        let execution_backend =
            self.force_execution_backend
                .unwrap_or_else(|| match state_backend {
                    StateBackend::Sparse => {
                        if nnz >= self.parallel_threshold {
                            ExecutionBackend::Parallel
                        } else {
                            ExecutionBackend::Scalar
                        }
                    }
                    StateBackend::Dense => {
                        if has_avx2() && total_dim >= SIMD_MIN_DIM {
                            ExecutionBackend::Simd
                        } else {
                            ExecutionBackend::Scalar
                        }
                    }
                });

        Backend::new(state_backend, execution_backend)
    }
}

// =============================================================================
// Statistics and Diagnostics
// =============================================================================

/// Statistics about dispatch decisions
#[derive(Debug, Default, Clone)]
pub struct DispatchStats {
    pub gate_1q_count: usize,
    pub gate_2q_count: usize,
    pub dense_ops: usize,
    pub sparse_ops: usize,
    pub simd_ops: usize,
    pub parallel_ops: usize,
    pub scalar_ops: usize,
}

impl DispatchStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, backend: Backend, is_2q: bool) {
        if is_2q {
            self.gate_2q_count += 1;
        } else {
            self.gate_1q_count += 1;
        }

        match backend.state {
            StateBackend::Dense => self.dense_ops += 1,
            StateBackend::Sparse => self.sparse_ops += 1,
        }

        match backend.execution {
            ExecutionBackend::Scalar => self.scalar_ops += 1,
            ExecutionBackend::Simd => self.simd_ops += 1,
            ExecutionBackend::Parallel => self.parallel_ops += 1,
            ExecutionBackend::ParallelSimd => {
                self.parallel_ops += 1;
                self.simd_ops += 1;
            }
        }
    }

    pub fn summary(&self) -> String {
        format!(
            "Gates: {}×1Q + {}×2Q | Dense: {} Sparse: {} | SIMD: {} Parallel: {} Scalar: {}",
            self.gate_1q_count,
            self.gate_2q_count,
            self.dense_ops,
            self.sparse_ops,
            self.simd_ops,
            self.parallel_ops,
            self.scalar_ops
        )
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    #[test]
    fn test_backend_selection_sparse() {
        // Very sparse: 10 nnz out of 1M
        let backend = select_backend(10, 1_000_000);
        assert_eq!(backend.state, StateBackend::Sparse);
        assert_eq!(backend.execution, ExecutionBackend::Scalar); // Below parallel threshold
    }

    #[test]
    fn test_backend_selection_dense() {
        // Dense: 50% filled
        let backend = select_backend(500_000, 1_000_000);
        assert_eq!(backend.state, StateBackend::Dense);
        // Execution depends on SIMD availability
    }

    #[test]
    fn test_backend_selection_large_sparse() {
        // Large sparse: 9K nnz out of 1M (0.9% density, below 1% threshold)
        let backend = select_backend(9_000, 1_000_000);
        assert_eq!(backend.state, StateBackend::Sparse);
        assert_eq!(backend.execution, ExecutionBackend::Parallel); // Above parallel threshold (8192)
    }

    #[test]
    fn test_quantum_state_dense() {
        let mut state = QuantumState::new_dense(4, 1, 1);
        let h = gates::hadamard();

        dispatch_gate_1q(&mut state, 0, &h);

        // Verify superposition
        let amp0 = state.get(0);
        let amp1 = state.get(1);
        let expected = 1.0 / 2.0_f64.sqrt();

        assert!((amp0.re - expected).abs() < EPSILON);
        assert!((amp1.re - expected).abs() < EPSILON);
    }

    #[test]
    fn test_quantum_state_sparse() {
        let mut state = QuantumState::new_sparse(4, 1, 1);
        let x = gates::pauli_x();

        dispatch_gate_1q(&mut state, 0, &x);

        // |0⟩ -> |1⟩
        assert!(state.get(0).norm() < EPSILON);
        assert!((state.get(1).re - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_dispatch_2q_gate() {
        let mut state = QuantumState::new_dense(4, 1, 1);
        let h = gates::hadamard();
        let cnot = gates::cnot();

        // Create Bell state
        dispatch_gate_1q(&mut state, 0, &h);
        dispatch_gate_2q(&mut state, 0, 1, &cnot);

        // Verify (|00⟩ + |11⟩)/√2
        let expected = 1.0 / 2.0_f64.sqrt();
        assert!((state.get(0).re - expected).abs() < EPSILON); // |0000⟩
        assert!(state.get(1).norm() < EPSILON); // |0001⟩
        assert!(state.get(2).norm() < EPSILON); // |0010⟩
        assert!((state.get(3).re - expected).abs() < EPSILON); // |0011⟩
    }

    #[test]
    fn test_dispatch_circuit() {
        let mut state = QuantumState::new_dense(4, 1, 1);

        let circuit = vec![
            CircuitGate::h(0),
            CircuitGate::cnot(0, 1),
            CircuitGate::h(2),
        ];

        dispatch_circuit(&mut state, &circuit);

        // Verify state is normalized
        let total_prob: f64 = (0..state.total_dim())
            .map(|i| state.get(i).norm_sqr())
            .sum();
        assert!((total_prob - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_gate_op_constructors() {
        let h = CircuitGate::h(0);
        let _x = CircuitGate::x(1);
        let cnot = CircuitGate::cnot(0, 1);
        let _cz = CircuitGate::cz(1, 2);

        match h {
            CircuitGate::Gate1Q { qubit, .. } => assert_eq!(qubit, 0),
            _ => panic!("Expected 1Q gate"),
        }

        match cnot {
            CircuitGate::Gate2Q { qubit1, qubit2, .. } => {
                assert_eq!(qubit1, 0);
                assert_eq!(qubit2, 1);
            }
            _ => panic!("Expected 2Q gate"),
        }
    }

    #[test]
    fn test_dispatcher_config() {
        // Force dense
        let config = DispatcherConfig::force_dense();
        let backend = config.select_backend(10, 1_000_000);
        assert_eq!(backend.state, StateBackend::Dense);

        // Force sparse
        let config = DispatcherConfig::force_sparse();
        let backend = config.select_backend(500_000, 1_000_000);
        assert_eq!(backend.state, StateBackend::Sparse);

        // Force parallel
        let config = DispatcherConfig::force_parallel();
        let backend = config.select_backend(100, 1000);
        assert_eq!(backend.execution, ExecutionBackend::Parallel);
    }

    #[test]
    fn test_dispatch_stats() {
        let mut stats = DispatchStats::new();

        stats.record(
            Backend::new(StateBackend::Dense, ExecutionBackend::Simd),
            false,
        );
        stats.record(
            Backend::new(StateBackend::Sparse, ExecutionBackend::Parallel),
            true,
        );

        assert_eq!(stats.gate_1q_count, 1);
        assert_eq!(stats.gate_2q_count, 1);
        assert_eq!(stats.dense_ops, 1);
        assert_eq!(stats.sparse_ops, 1);
        assert_eq!(stats.simd_ops, 1);
        assert_eq!(stats.parallel_ops, 1);
    }

    #[test]
    fn test_quantum_state_conversion() {
        let dense = QuantumState::new_dense(4, 1, 1);
        let sparse = dense.to_sparse();
        let back_to_dense = QuantumState::Sparse(sparse).to_dense();

        // Should be identical
        for i in 0..16 {
            let diff = (dense.get(i) - back_to_dense.state_vector[i]).norm();
            assert!(diff < EPSILON);
        }
    }

    #[test]
    fn test_backend_description() {
        let backend = Backend::new(StateBackend::Dense, ExecutionBackend::Simd);
        assert!(backend.description().contains("Dense"));
        assert!(backend.description().contains("Simd"));
    }

    #[test]
    fn test_performance_comparison() {
        use std::time::Instant;

        let num_qubits = 10;
        let num_gates = 50;
        let h = gates::hadamard();

        // Dense dispatch
        let mut dense_state = QuantumState::new_dense(num_qubits, 1, 1);
        let start = Instant::now();
        for _ in 0..num_gates {
            for q in 0..num_qubits {
                dispatch_gate_1q(&mut dense_state, q, &h);
            }
        }
        let dense_time = start.elapsed();

        // Sparse dispatch (stays sparse since H makes superposition)
        let mut sparse_state = QuantumState::new_sparse(num_qubits, 1, 1);
        let start = Instant::now();
        for _ in 0..num_gates {
            for q in 0..num_qubits {
                dispatch_gate_1q(&mut sparse_state, q, &h);
            }
        }
        let sparse_time = start.elapsed();

        println!("\n=== Dispatcher Performance ===");
        println!(
            "Qubits: {}, Gates per iter: {}, Iterations: {}",
            num_qubits, num_qubits, num_gates
        );
        println!("Dense dispatch: {:?}", dense_time);
        println!("Sparse dispatch: {:?}", sparse_time);
        println!("Dense backend: {:?}", dense_state.recommended_backend());
        println!("Sparse backend: {:?}", sparse_state.recommended_backend());
    }
}

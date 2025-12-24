//! Week 2 Day 4: Dispatcher Integration Tests
//!
//! Validates:
//! - Backend selection logic
//! - Unified QuantumState API
//! - Dispatch correctness across backends

use tqp_core::{
    dispatch_circuit, dispatch_gate_1q, dispatch_gate_2q, gates, select_backend, Backend,
    CircuitGate, DispatchStats, DispatcherConfig, ExecutionBackend, QuantumState, StateBackend,
};

const EPSILON: f64 = 1e-10;

// =============================================================================
// Backend Selection Tests
// =============================================================================

#[test]
fn test_backend_selection_thresholds() {
    // Very sparse: should select Sparse + Scalar
    let backend1 = select_backend(100, 1_000_000);
    assert_eq!(backend1.state, StateBackend::Sparse);
    assert_eq!(backend1.execution, ExecutionBackend::Scalar);

    // Large sparse: should select Sparse + Parallel
    let backend2 = select_backend(9_000, 1_000_000);
    assert_eq!(backend2.state, StateBackend::Sparse);
    assert_eq!(backend2.execution, ExecutionBackend::Parallel);

    // Dense: should select Dense + SIMD (if available)
    let backend3 = select_backend(500_000, 1_000_000);
    assert_eq!(backend3.state, StateBackend::Dense);
}

#[test]
fn test_backend_description() {
    let backend = Backend::new(StateBackend::Dense, ExecutionBackend::Simd);
    let desc = backend.description();
    assert!(desc.contains("Dense"));
    assert!(desc.contains("Simd"));
}

// =============================================================================
// QuantumState API Tests
// =============================================================================

#[test]
fn test_quantum_state_creation() {
    let dense = QuantumState::new_dense(4, 2, 2);
    let sparse = QuantumState::new_sparse(4, 2, 2);

    assert!(!dense.is_sparse());
    assert!(sparse.is_sparse());

    assert_eq!(dense.total_dim(), 16 * 2 * 2);
    assert_eq!(sparse.total_dim(), 16 * 2 * 2);
}

#[test]
fn test_quantum_state_density() {
    // Dense with ground state: 1 nnz
    let dense = QuantumState::new_dense(10, 1, 1);
    assert_eq!(dense.nnz(), 1);
    assert!(dense.density() < 0.001);

    // Sparse with ground state: 1 nnz
    let sparse = QuantumState::new_sparse(10, 1, 1);
    assert_eq!(sparse.nnz(), 1);
}

#[test]
fn test_quantum_state_conversion() {
    let mut dense = QuantumState::new_dense(4, 1, 1);
    dispatch_gate_1q(&mut dense, 0, &gates::hadamard());

    let sparse = dense.to_sparse();
    let back_to_dense = sparse.to_dense();

    // Verify roundtrip preserves amplitudes
    for i in 0..16 {
        let diff = (dense.get(i) - back_to_dense.state_vector[i]).norm();
        assert!(diff < EPSILON, "Roundtrip mismatch at index {}", i);
    }
}

// =============================================================================
// Dispatch Correctness Tests
// =============================================================================

#[test]
fn test_dispatch_1q_dense_correctness() {
    let mut state = QuantumState::new_dense(4, 1, 1);
    let h = gates::hadamard();

    dispatch_gate_1q(&mut state, 0, &h);

    // |0⟩ -> (|0⟩ + |1⟩)/√2
    let expected = 1.0 / 2.0_f64.sqrt();
    assert!((state.get(0).re - expected).abs() < EPSILON);
    assert!((state.get(1).re - expected).abs() < EPSILON);

    // Verify normalization
    let total_prob: f64 = (0..state.total_dim())
        .map(|i| state.get(i).norm_sqr())
        .sum();
    assert!((total_prob - 1.0).abs() < EPSILON);
}

#[test]
fn test_dispatch_1q_sparse_correctness() {
    let mut state = QuantumState::new_sparse(4, 1, 1);
    let x = gates::pauli_x();

    dispatch_gate_1q(&mut state, 0, &x);

    // |0⟩ -> |1⟩
    assert!(state.get(0).norm() < EPSILON);
    assert!((state.get(1).re - 1.0).abs() < EPSILON);
}

#[test]
fn test_dispatch_2q_bell_state() {
    let mut state = QuantumState::new_dense(4, 1, 1);

    dispatch_gate_1q(&mut state, 0, &gates::hadamard());
    dispatch_gate_2q(&mut state, 0, 1, &gates::cnot());

    // Bell state: (|00⟩ + |11⟩)/√2
    let expected = 1.0 / 2.0_f64.sqrt();
    assert!((state.get(0).re - expected).abs() < EPSILON);
    assert!(state.get(1).norm() < EPSILON);
    assert!(state.get(2).norm() < EPSILON);
    assert!((state.get(3).re - expected).abs() < EPSILON);
}

#[test]
fn test_dispatch_dense_vs_sparse_equivalence() {
    // Same circuit on dense and sparse should give same result
    let mut dense = QuantumState::new_dense(5, 1, 1);
    let mut sparse = QuantumState::new_sparse(5, 1, 1);

    let h = gates::hadamard();
    let cnot = gates::cnot();

    // Apply same gates
    dispatch_gate_1q(&mut dense, 0, &h);
    dispatch_gate_1q(&mut sparse, 0, &h);

    dispatch_gate_2q(&mut dense, 0, 1, &cnot);
    dispatch_gate_2q(&mut sparse, 0, 1, &cnot);

    dispatch_gate_1q(&mut dense, 2, &h);
    dispatch_gate_1q(&mut sparse, 2, &h);

    // Compare
    for i in 0..dense.total_dim() {
        let diff = (dense.get(i) - sparse.get(i)).norm();
        assert!(diff < EPSILON, "Dense/Sparse mismatch at index {}", i);
    }
}

// =============================================================================
// Circuit Dispatch Tests
// =============================================================================

#[test]
fn test_dispatch_circuit_correctness() {
    let mut state = QuantumState::new_dense(4, 1, 1);

    let circuit = vec![
        CircuitGate::h(0),
        CircuitGate::h(1),
        CircuitGate::cnot(0, 1),
        CircuitGate::h(2),
    ];

    dispatch_circuit(&mut state, &circuit);

    // Verify normalization
    let total_prob: f64 = (0..state.total_dim())
        .map(|i| state.get(i).norm_sqr())
        .sum();
    assert!((total_prob - 1.0).abs() < EPSILON);

    // Should have multiple non-zero amplitudes (entangled state)
    let nnz = (0..state.total_dim())
        .filter(|&i| state.get(i).norm_sqr() > 1e-14)
        .count();
    assert!(nnz > 1);
}

#[test]
fn test_circuit_gate_constructors() {
    let circuit = vec![
        CircuitGate::h(0),
        CircuitGate::x(1),
        CircuitGate::y(2),
        CircuitGate::z(3),
        CircuitGate::cnot(0, 1),
        CircuitGate::cz(1, 2),
        CircuitGate::swap(2, 3),
    ];

    assert_eq!(circuit.len(), 7);

    // Verify gate types
    match &circuit[0] {
        CircuitGate::Gate1Q { qubit, .. } => assert_eq!(*qubit, 0),
        _ => panic!("Expected 1Q gate"),
    }

    match &circuit[4] {
        CircuitGate::Gate2Q { qubit1, qubit2, .. } => {
            assert_eq!(*qubit1, 0);
            assert_eq!(*qubit2, 1);
        }
        _ => panic!("Expected 2Q gate"),
    }
}

// =============================================================================
// Configuration Tests
// =============================================================================

#[test]
fn test_dispatcher_config_force_backends() {
    // Force dense
    let config_dense = DispatcherConfig::force_dense();
    let backend = config_dense.select_backend(100, 1_000_000);
    assert_eq!(backend.state, StateBackend::Dense);

    // Force sparse
    let config_sparse = DispatcherConfig::force_sparse();
    let backend = config_sparse.select_backend(999_999, 1_000_000);
    assert_eq!(backend.state, StateBackend::Sparse);

    // Force parallel
    let config_parallel = DispatcherConfig::force_parallel();
    let backend = config_parallel.select_backend(10, 100);
    assert_eq!(backend.execution, ExecutionBackend::Parallel);
}

#[test]
fn test_dispatch_stats_tracking() {
    let mut stats = DispatchStats::new();

    // Record various operations
    stats.record(
        Backend::new(StateBackend::Dense, ExecutionBackend::Simd),
        false,
    );
    stats.record(
        Backend::new(StateBackend::Dense, ExecutionBackend::Simd),
        false,
    );
    stats.record(
        Backend::new(StateBackend::Sparse, ExecutionBackend::Parallel),
        true,
    );
    stats.record(
        Backend::new(StateBackend::Sparse, ExecutionBackend::Scalar),
        false,
    );

    assert_eq!(stats.gate_1q_count, 3);
    assert_eq!(stats.gate_2q_count, 1);
    assert_eq!(stats.dense_ops, 2);
    assert_eq!(stats.sparse_ops, 2);
    assert_eq!(stats.simd_ops, 2);
    assert_eq!(stats.parallel_ops, 1);
    assert_eq!(stats.scalar_ops, 1);

    let summary = stats.summary();
    assert!(summary.contains("3×1Q"));
    assert!(summary.contains("1×2Q"));
}

// =============================================================================
// Performance Tests
// =============================================================================

#[test]
fn test_dispatch_performance() {
    use std::time::Instant;

    let num_qubits = 10;
    let num_gates = 20;

    // Dense dispatch
    let mut dense = QuantumState::new_dense(num_qubits, 1, 1);
    let start = Instant::now();
    for _ in 0..num_gates {
        for q in 0..num_qubits {
            dispatch_gate_1q(&mut dense, q, &gates::hadamard());
        }
    }
    let dense_time = start.elapsed();

    // Sparse dispatch (starts sparse but becomes dense-like)
    let mut sparse = QuantumState::new_sparse(num_qubits, 1, 1);
    let start = Instant::now();
    for _ in 0..num_gates {
        for q in 0..num_qubits {
            dispatch_gate_1q(&mut sparse, q, &gates::hadamard());
        }
    }
    let sparse_time = start.elapsed();

    println!("\n=== Dispatcher Performance ===");
    println!("Qubits: {}, Iterations: {}", num_qubits, num_gates);
    println!("Total gates: {}", num_qubits * num_gates);
    println!("Dense dispatch: {:?}", dense_time);
    println!("Sparse dispatch: {:?}", sparse_time);
    println!("Dense nnz after: {}", dense.nnz());
    println!("Sparse nnz after: {}", sparse.nnz());

    // Both should be normalized
    let dense_prob: f64 = (0..dense.total_dim())
        .map(|i| dense.get(i).norm_sqr())
        .sum();
    let sparse_prob: f64 = (0..sparse.total_dim())
        .map(|i| sparse.get(i).norm_sqr())
        .sum();

    assert!((dense_prob - 1.0).abs() < EPSILON);
    assert!((sparse_prob - 1.0).abs() < EPSILON);
}

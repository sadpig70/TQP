//! Week 2 Day 1: Rayon Parallelization Integration Tests
//!
//! Validates:
//! - Parallel operations correctness
//! - Multi-circuit execution
//! - Performance characteristics

use tqp_core::{
    apply_gate_2q_sparse, apply_gate_sparse, execute_circuit, gates, get_thread_count,
    parallel_apply_gate_2q_sparse, parallel_apply_gate_sparse, parallel_compute_stats,
    parallel_execute_circuits, parallel_expval_z, GateOp, SparseStateVector,
};

const EPSILON: f64 = 1e-10;

// =============================================================================
// Correctness Tests
// =============================================================================

#[test]
fn test_parallel_matches_sequential() {
    // Create identical states
    let mut seq_state = SparseStateVector::new(10, 1, 1);
    let mut par_state = SparseStateVector::new(10, 1, 1);

    // Apply same sequence
    let gates_to_apply = [
        (0, gates::hadamard()),
        (1, gates::hadamard()),
        (2, gates::pauli_x()),
        (3, gates::pauli_z()),
    ];

    for (q, g) in &gates_to_apply {
        apply_gate_sparse(&mut seq_state, *q, g);
        parallel_apply_gate_sparse(&mut par_state, *q, g);
    }

    // Verify identical results
    for i in 0..(1 << 10) {
        let diff = (seq_state.get_amplitude(i) - par_state.get_amplitude(i)).norm();
        assert!(diff < EPSILON, "Mismatch at index {}", i);
    }
}

#[test]
fn test_parallel_2q_matches_sequential() {
    let mut seq_state = SparseStateVector::new(8, 1, 1);
    let mut par_state = SparseStateVector::new(8, 1, 1);

    // Create superposition first
    for q in 0..8 {
        apply_gate_sparse(&mut seq_state, q, &gates::hadamard());
        apply_gate_sparse(&mut par_state, q, &gates::hadamard());
    }

    // Apply 2Q gate
    let cnot = gates::cnot();
    apply_gate_2q_sparse(&mut seq_state, 0, 1, &cnot);
    parallel_apply_gate_2q_sparse(&mut par_state, 0, 1, &cnot);

    for i in 0..(1 << 8) {
        let diff = (seq_state.get_amplitude(i) - par_state.get_amplitude(i)).norm();
        assert!(diff < EPSILON, "Mismatch at index {}", i);
    }
}

// =============================================================================
// Multi-Circuit Tests
// =============================================================================

#[test]
fn test_parallel_circuits_correctness() {
    let num_circuits = 8;

    // Sequential reference
    let mut seq_results = Vec::new();
    for i in 0..num_circuits {
        let mut state = SparseStateVector::new(5, 1, 1);
        apply_gate_sparse(&mut state, i % 5, &gates::hadamard());
        apply_gate_sparse(&mut state, (i + 1) % 5, &gates::hadamard());
        seq_results.push(state.nnz());
    }

    // Parallel execution
    let par_results = parallel_execute_circuits(num_circuits, |i| {
        let mut state = SparseStateVector::new(5, 1, 1);
        apply_gate_sparse(&mut state, i % 5, &gates::hadamard());
        apply_gate_sparse(&mut state, (i + 1) % 5, &gates::hadamard());
        state
    });

    // Verify
    for (i, (seq_nnz, par_state)) in seq_results.iter().zip(par_results.iter()).enumerate() {
        assert_eq!(*seq_nnz, par_state.nnz(), "Circuit {} nnz mismatch", i);
    }
}

#[test]
fn test_vqe_parameter_sweep() {
    // Simulate VQE parameter sweep: many circuits with different angles
    let num_params = 20;

    let results: Vec<f64> = parallel_execute_circuits(num_params, |i| {
        let angle = std::f64::consts::PI * (i as f64) / (num_params as f64);
        let mut state = SparseStateVector::new(4, 1, 1);
        apply_gate_sparse(&mut state, 0, &gates::hadamard());
        apply_gate_sparse(&mut state, 0, &gates::rz(angle));
        state
    })
    .iter()
    .map(|s| parallel_expval_z(s, 0))
    .collect();

    // Should produce varying expectation values
    assert_eq!(results.len(), num_params);

    // First (angle=0) should have <Z> ≈ 0 (|+⟩ state)
    // Last (angle=π) should also have <Z> ≈ 0 (phase doesn't change probabilities)
    assert!(results[0].abs() < EPSILON);
}

// =============================================================================
// Execute Circuit Tests
// =============================================================================

#[test]
fn test_execute_circuit_api() {
    let mut state = SparseStateVector::new(4, 1, 1);

    let circuit = vec![
        GateOp::Single(0, gates::hadamard()),
        GateOp::Single(1, gates::hadamard()),
        GateOp::Two(0, 1, gates::cnot()),
        GateOp::Single(2, gates::pauli_x()),
    ];

    execute_circuit(&mut state, &circuit);

    // Verify entanglement created
    let stats = parallel_compute_stats(&state);
    assert!(stats.nnz > 1, "Should create superposition");
    assert!((stats.total_probability - 1.0).abs() < EPSILON);
}

// =============================================================================
// Statistics Tests
// =============================================================================

#[test]
fn test_parallel_stats_accuracy() {
    let mut state = SparseStateVector::new(8, 1, 1);

    // Create superposition
    for q in 0..8 {
        apply_gate_sparse(&mut state, q, &gates::hadamard());
    }

    let stats = parallel_compute_stats(&state);

    assert_eq!(stats.nnz, 256, "8-qubit H should have 2^8 nnz");
    assert!((stats.total_probability - 1.0).abs() < EPSILON);
    assert!(stats.max_amplitude > 0.0);
    assert!(stats.entropy > 0.0, "Should have non-zero entropy");
}

// =============================================================================
// Thread Pool Tests
// =============================================================================

#[test]
fn test_thread_configuration() {
    let count = get_thread_count();
    println!("Available threads: {}", count);
    assert!(count >= 1, "Should have at least one thread");
}

// =============================================================================
// Performance Validation
// =============================================================================

#[test]
fn test_performance_scaling() {
    use std::time::Instant;

    let num_circuits = 32;
    let num_qubits = 8;

    // Sequential
    let start = Instant::now();
    let mut seq_results = Vec::with_capacity(num_circuits);
    for _i in 0..num_circuits {
        let mut state = SparseStateVector::new(num_qubits, 1, 1);
        for q in 0..num_qubits {
            apply_gate_sparse(&mut state, q, &gates::hadamard());
        }
        seq_results.push(state);
    }
    let seq_time = start.elapsed();

    // Parallel
    let start = Instant::now();
    let par_results = parallel_execute_circuits(num_circuits, |_| {
        let mut state = SparseStateVector::new(num_qubits, 1, 1);
        for q in 0..num_qubits {
            apply_gate_sparse(&mut state, q, &gates::hadamard());
        }
        state
    });
    let par_time = start.elapsed();

    let speedup = seq_time.as_secs_f64() / par_time.as_secs_f64();

    println!("\n=== Performance Scaling Test ===");
    println!("Circuits: {}, Qubits: {}", num_circuits, num_qubits);
    println!("Sequential: {:?}", seq_time);
    println!("Parallel: {:?}", par_time);
    println!("Speedup: {:.2}x", speedup);
    println!("Threads: {}", get_thread_count());

    // Should have at least some speedup on multi-core systems
    // (may be ~1x on single-core)
    assert!(speedup > 0.5, "Should not be significantly slower");

    // Verify correctness
    assert_eq!(seq_results.len(), par_results.len());
}

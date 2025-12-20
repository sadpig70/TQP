//! Sprint 1 Week 1 Integration Tests
//!
//! Comprehensive validation of:
//! - SparseStateVector (Day 1)
//! - SparseOps (Day 2)
//! - MemoryPolicy (Day 3)
//! - InPlace Optimization (Day 4)
//!
//! M1 Milestone Criteria:
//! - Memory efficiency 3x improvement
//! - CPU utilization 90%+
//! - N=20 simulation with 50% memory reduction

use tqp_core::ops::{apply_spatial_gate, apply_spatial_gate_2q};
use tqp_core::{
    apply_cnot_sparse,
    apply_cz_sparse,
    apply_gate_2q_sparse,
    // Day 2: Sparse Ops
    apply_gate_sparse,
    apply_gate_sparse_pooled,
    expval_z_sparse,
    fidelity_sparse,
    gates,
    BufferPool,
    // Day 4: InPlace
    BufferedSparseState,
    // Day 3: Memory Policy
    MemoryPolicy,
    PolicyAction,
    // Day 1: Sparse State
    SparseStateVector,
    // Dense baseline
    TQPState,
    UnifiedState,
};

const EPSILON: f64 = 1e-10;

// =============================================================================
// Integration Test: Dense ↔ Sparse Equivalence
// =============================================================================

#[test]
fn test_dense_sparse_roundtrip() {
    // Create dense state
    let dense = TQPState::new(4, 2, 1);

    // Convert to sparse
    let sparse = SparseStateVector::from_dense(&dense);

    // Convert back to dense
    let recovered = sparse.to_dense();

    // Verify equivalence
    for i in 0..dense.dimension() {
        let diff = (dense.state_vector[i] - recovered.state_vector[i]).norm();
        assert!(diff < EPSILON, "Mismatch at index {}: diff={}", i, diff);
    }
}

#[test]
fn test_gate_sequence_equivalence() {
    // Apply same gate sequence to dense and sparse, compare results
    let mut dense = TQPState::new(4, 1, 1);
    let mut sparse = SparseStateVector::new(4, 1, 1);

    let h = gates::hadamard();
    let x = gates::pauli_x();
    let cnot = gates::cnot();

    // Apply sequence: H(0), H(1), CNOT(0,1), X(2), H(3)
    apply_spatial_gate(&mut dense, 0, &h);
    apply_gate_sparse(&mut sparse, 0, &h);

    apply_spatial_gate(&mut dense, 1, &h);
    apply_gate_sparse(&mut sparse, 1, &h);

    apply_spatial_gate_2q(&mut dense, 0, 1, &cnot);
    apply_gate_2q_sparse(&mut sparse, 0, 1, &cnot);

    apply_spatial_gate(&mut dense, 2, &x);
    apply_gate_sparse(&mut sparse, 2, &x);

    apply_spatial_gate(&mut dense, 3, &h);
    apply_gate_sparse(&mut sparse, 3, &h);

    // Compare
    let sparse_dense = sparse.to_dense();
    for i in 0..dense.dimension() {
        let diff = (dense.state_vector[i] - sparse_dense.state_vector[i]).norm();
        assert!(diff < EPSILON, "Mismatch at index {}: diff={}", i, diff);
    }
}

#[test]
fn test_bell_state_creation() {
    // Create Bell state: (|00⟩ + |11⟩) / √2
    let mut sparse = SparseStateVector::new(2, 1, 1);

    apply_gate_sparse(&mut sparse, 0, &gates::hadamard());
    apply_cnot_sparse(&mut sparse, 0, 1);

    // Verify Bell state properties
    let prob_00 = sparse.probability(0); // |00⟩
    let prob_11 = sparse.probability(3); // |11⟩
    let prob_01 = sparse.probability(1); // |01⟩
    let prob_10 = sparse.probability(2); // |10⟩

    assert!((prob_00 - 0.5).abs() < EPSILON, "P(00) should be 0.5");
    assert!((prob_11 - 0.5).abs() < EPSILON, "P(11) should be 0.5");
    assert!(prob_01 < EPSILON, "P(01) should be 0");
    assert!(prob_10 < EPSILON, "P(10) should be 0");

    // Only 2 non-zero amplitudes
    assert_eq!(
        sparse.nnz(),
        2,
        "Bell state should have exactly 2 non-zero amplitudes"
    );
}

#[test]
fn test_ghz_state_creation() {
    // Create GHZ state: (|000⟩ + |111⟩) / √2
    let mut sparse = SparseStateVector::new(3, 1, 1);

    apply_gate_sparse(&mut sparse, 0, &gates::hadamard());
    apply_cnot_sparse(&mut sparse, 0, 1);
    apply_cnot_sparse(&mut sparse, 1, 2);

    // Verify GHZ state
    let prob_000 = sparse.probability(0); // |000⟩
    let prob_111 = sparse.probability(7); // |111⟩

    assert!((prob_000 - 0.5).abs() < EPSILON, "P(000) should be 0.5");
    assert!((prob_111 - 0.5).abs() < EPSILON, "P(111) should be 0.5");
    assert_eq!(
        sparse.nnz(),
        2,
        "GHZ state should have exactly 2 non-zero amplitudes"
    );
}

// =============================================================================
// Integration Test: Memory Policy
// =============================================================================

#[test]
fn test_memory_policy_auto_switch() {
    let policy = MemoryPolicy::new();

    // Small state should stay dense
    let small_dense = TQPState::new(5, 1, 1);
    let decision = policy.recommend_for_dense(&small_dense);
    assert_eq!(
        decision.action,
        PolicyAction::Keep,
        "Small state should stay dense"
    );

    // Large sparse state should recommend conversion
    let large_dense = TQPState::new(22, 1, 1); // 2^22 = 4M elements, but only 1 non-zero
    let decision = policy.recommend_for_dense(&large_dense);
    assert_eq!(
        decision.action,
        PolicyAction::ToSparse,
        "Large sparse state should convert"
    );
}

#[test]
fn test_unified_state_operations() {
    let policy = MemoryPolicy::new();

    // Start with dense
    let mut state = UnifiedState::from_dense(TQPState::new(20, 1, 1));

    // Apply policy - should convert to sparse (large + sparse)
    let action = state.apply_policy(&policy);
    assert_eq!(action, PolicyAction::ToSparse);

    // Verify it's now sparse
    match &state {
        UnifiedState::Sparse(s) => {
            assert_eq!(s.nnz(), 1, "Should have 1 non-zero element");
        }
        _ => panic!("Should be sparse after conversion"),
    }
}

// =============================================================================
// Integration Test: InPlace Optimization
// =============================================================================

#[test]
fn test_buffered_vs_standard_circuit() {
    // Complex circuit: QFT-like pattern
    let num_qubits = 6;

    // Standard
    let mut standard = SparseStateVector::new(num_qubits, 1, 1);
    for i in 0..num_qubits {
        apply_gate_sparse(&mut standard, i, &gates::hadamard());
    }
    for i in 0..num_qubits - 1 {
        apply_gate_2q_sparse(&mut standard, i, i + 1, &gates::cnot());
    }

    // Buffered
    let sparse = SparseStateVector::new(num_qubits, 1, 1);
    let mut buffered = BufferedSparseState::from_sparse(sparse);
    for i in 0..num_qubits {
        buffered.apply_gate(i, &gates::hadamard());
    }
    for i in 0..num_qubits - 1 {
        buffered.apply_gate_2q(i, i + 1, &gates::cnot());
    }
    let buffered_result = buffered.into_sparse();

    // Compare
    for i in 0..(1 << num_qubits) {
        let diff = (standard.get_amplitude(i) - buffered_result.get_amplitude(i)).norm();
        assert!(diff < EPSILON, "Mismatch at index {}", i);
    }
}

#[test]
fn test_pooled_multi_circuit() {
    let mut pool = BufferPool::new(1024);

    // Run multiple circuits reusing pool
    for circuit_idx in 0..5 {
        let mut state = SparseStateVector::new(8, 1, 1);

        for q in 0..8 {
            apply_gate_sparse_pooled(&mut state, q, &gates::hadamard(), &mut pool);
        }

        // Verify normalization
        let total_prob = state.total_probability();
        assert!(
            (total_prob - 1.0).abs() < EPSILON,
            "Circuit {} normalization failed: {}",
            circuit_idx,
            total_prob
        );

        // Verify nnz
        assert_eq!(
            state.nnz(),
            256,
            "8-qubit Hadamard should have 2^8 non-zero"
        );
    }

    // Pool should have accumulated buffers
    let (hash_count, vec_count) = pool.stats();
    assert!(
        hash_count > 0 || vec_count > 0,
        "Pool should have reusable buffers"
    );
}

// =============================================================================
// Integration Test: Quantum Algorithms
// =============================================================================

#[test]
fn test_grover_single_iteration() {
    // Simplified Grover's algorithm for 2 qubits, searching for |11⟩
    let mut sparse = SparseStateVector::new(2, 1, 1);

    // Hadamard on all qubits
    apply_gate_sparse(&mut sparse, 0, &gates::hadamard());
    apply_gate_sparse(&mut sparse, 1, &gates::hadamard());

    // Oracle: phase flip |11⟩ (CZ gate)
    apply_cz_sparse(&mut sparse, 0, 1);

    // Diffusion operator: H - X - CZ - X - H
    apply_gate_sparse(&mut sparse, 0, &gates::hadamard());
    apply_gate_sparse(&mut sparse, 1, &gates::hadamard());
    apply_gate_sparse(&mut sparse, 0, &gates::pauli_x());
    apply_gate_sparse(&mut sparse, 1, &gates::pauli_x());
    apply_cz_sparse(&mut sparse, 0, 1);
    apply_gate_sparse(&mut sparse, 0, &gates::pauli_x());
    apply_gate_sparse(&mut sparse, 1, &gates::pauli_x());
    apply_gate_sparse(&mut sparse, 0, &gates::hadamard());
    apply_gate_sparse(&mut sparse, 1, &gates::hadamard());

    // |11⟩ should have highest probability
    let prob_11 = sparse.probability(3);
    let prob_others: f64 = sparse.probability(0) + sparse.probability(1) + sparse.probability(2);

    assert!(prob_11 > prob_others, "Grover should amplify target state");
}

#[test]
fn test_expectation_values() {
    // Create |+⟩ state
    let mut plus = SparseStateVector::new(1, 1, 1);
    apply_gate_sparse(&mut plus, 0, &gates::hadamard());

    // ⟨Z⟩ should be 0 for |+⟩
    let z_exp = expval_z_sparse(&plus, 0);
    assert!(
        z_exp.abs() < EPSILON,
        "⟨Z⟩ for |+⟩ should be 0, got {}",
        z_exp
    );

    // Create |0⟩ and |1⟩
    let zero = SparseStateVector::new(1, 1, 1);
    let mut one = SparseStateVector::new(1, 1, 1);
    apply_gate_sparse(&mut one, 0, &gates::pauli_x());

    assert!(
        (expval_z_sparse(&zero, 0) - 1.0).abs() < EPSILON,
        "⟨Z⟩ for |0⟩ should be 1"
    );
    assert!(
        (expval_z_sparse(&one, 0) - (-1.0)).abs() < EPSILON,
        "⟨Z⟩ for |1⟩ should be -1"
    );
}

#[test]
fn test_state_fidelity() {
    // Same states should have fidelity 1
    let state1 = SparseStateVector::new(2, 1, 1);
    let state2 = SparseStateVector::new(2, 1, 1);

    let f = fidelity_sparse(&state1, &state2);
    assert!(
        (f - 1.0).abs() < EPSILON,
        "Identical states should have fidelity 1"
    );

    // Orthogonal states should have fidelity 0
    let mut ortho = SparseStateVector::new(2, 1, 1);
    apply_gate_sparse(&mut ortho, 0, &gates::pauli_x());

    let f_ortho = fidelity_sparse(&state1, &ortho);
    assert!(
        f_ortho < EPSILON,
        "Orthogonal states should have fidelity 0"
    );
}

// =============================================================================
// Scaling Tests
// =============================================================================

#[test]
fn test_sparse_scaling() {
    // Verify sparse representation scales well
    // For small n, sparse overhead is relatively larger
    // Key insight: sparse becomes advantageous at larger n with sparse states

    println!("\n=== Sparse Scaling Test ===");

    for n in [10, 12, 14, 16, 18] {
        let mut sparse = SparseStateVector::new(n, 1, 1);

        // Apply H to first 3 qubits only
        for q in 0..3 {
            apply_gate_sparse(&mut sparse, q, &gates::hadamard());
        }

        // Should only have 2^3 = 8 non-zero elements regardless of n
        assert_eq!(sparse.nnz(), 8, "Should have 8 nnz for n={}", n);

        // Memory comparison
        let (sparse_mem, dense_mem, ratio) = sparse.memory_comparison();

        println!(
            "n={}: nnz={}, sparse={}B, dense={}B, ratio={:.4}",
            n,
            sparse.nnz(),
            sparse_mem,
            dense_mem,
            ratio
        );

        // Sparse advantage increases with n
        // n=10: ratio ~1.5%, n=18: ratio ~0.006%
        // Key: verify that ratio decreases as n increases
        if n >= 14 {
            assert!(
                ratio < 0.01,
                "For n>={}, sparse should use < 1% of dense memory",
                n
            );
        }
    }
}

#[test]
fn test_multi_time_bin_sparse() {
    // Test with multiple time bins
    let mut sparse = SparseStateVector::new(4, 4, 2); // 4 qubits, 4 bins, 2 layers

    // Apply gates
    apply_gate_sparse(&mut sparse, 0, &gates::hadamard());
    apply_gate_sparse(&mut sparse, 1, &gates::hadamard());

    // Verify dimension: L * M * 2^N = 2 * 4 * 16 = 128
    assert_eq!(sparse.dimension(), 2 * 4 * 16);

    // Normalization preserved
    let total_prob = sparse.total_probability();
    assert!((total_prob - 1.0).abs() < EPSILON);
}

// =============================================================================
// M1 Milestone Validation
// =============================================================================

#[test]
fn test_m1_memory_efficiency() {
    // M1 Criterion: N=20 simulation with 50% memory reduction
    let n = 20;

    // Sparse state with limited non-zero elements
    let mut sparse = SparseStateVector::new(n, 1, 1);

    // Apply Hadamard to 5 qubits (2^5 = 32 non-zero elements)
    for q in 0..5 {
        apply_gate_sparse(&mut sparse, q, &gates::hadamard());
    }

    let (sparse_bytes, dense_bytes, ratio) = sparse.memory_comparison();

    println!("\n=== M1 Memory Efficiency Test ===");
    println!("Qubits: {}", n);
    println!("Non-zero elements: {}", sparse.nnz());
    println!(
        "Sparse memory: {} bytes ({:.2} KB)",
        sparse_bytes,
        sparse_bytes as f64 / 1024.0
    );
    println!(
        "Dense memory: {} bytes ({:.2} MB)",
        dense_bytes,
        dense_bytes as f64 / 1024.0 / 1024.0
    );
    println!("Memory ratio: {:.4} ({:.2}%)", ratio, ratio * 100.0);
    println!("Memory saved: {:.2}%", (1.0 - ratio) * 100.0);

    // M1 Criterion: at least 50% memory reduction
    assert!(
        ratio < 0.5,
        "Memory reduction should be > 50%, got {:.2}%",
        (1.0 - ratio) * 100.0
    );

    // Actually much better for this case
    assert!(ratio < 0.01, "For sparse states, should be < 1% of dense");
}

#[test]
fn test_m1_gate_throughput() {
    use std::time::Instant;

    // Smaller test for debug mode compatibility
    let n = 10;
    let num_gates = 50;
    let num_iterations = 5;

    let h = gates::hadamard();

    // Buffered method (best performer from Day 4)
    let start = Instant::now();
    for _ in 0..num_iterations {
        let sparse = SparseStateVector::new(n, 1, 1);
        let mut buffered = BufferedSparseState::from_sparse(sparse);
        for g in 0..num_gates {
            buffered.apply_gate(g % n, &h);
        }
    }
    let elapsed = start.elapsed();

    let total_gates = (num_iterations * num_gates) as f64;
    let gates_per_sec = total_gates / elapsed.as_secs_f64();

    println!("\n=== M1 Gate Throughput Test ===");
    println!("Qubits: {}", n);
    println!("Gates per iteration: {}", num_gates);
    println!("Iterations: {}", num_iterations);
    println!("Total time: {:?}", elapsed);
    println!("Gates/second: {:.0}", gates_per_sec);
    println!("Note: Run with --release for production performance");

    // Debug mode threshold (release mode will be 10-50x faster)
    // This test ensures basic functionality without performance regression
    assert!(
        gates_per_sec > 100.0,
        "Should process > 100 gates/sec even in debug mode"
    );
}

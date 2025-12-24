//! Week 2 Day 3: SIMD AVX2 Integration Tests
//!
//! Validates:
//! - CPU feature detection
//! - SIMD gate operations correctness
//! - Performance comparison with scalar

// use num_complex::Complex64;
use tqp_core::ops::{apply_spatial_gate, apply_spatial_gate_2q};
use tqp_core::{
    apply_gate_1q_simd, apply_gate_2q_simd, apply_gates_batch_simd, gates, has_avx2, has_avx512,
    total_probability_simd, SimdLevel, TQPState,
};

const EPSILON: f64 = 1e-10;

// =============================================================================
// Feature Detection Tests
// =============================================================================

#[test]
fn test_simd_level_detection() {
    let level = SimdLevel::detect();

    println!("=== SIMD Feature Detection ===");
    println!("Detected level: {:?}", level);
    println!("AVX2 available: {}", has_avx2());
    println!("AVX-512 available: {}", has_avx512());
    println!("Vector width: {} f64", level.vector_width());

    // Should at least have scalar
    assert!(level.vector_width() >= 1);
}

// =============================================================================
// 1Q Gate Tests
// =============================================================================

#[test]
fn test_simd_hadamard_correctness() {
    let mut state = TQPState::new(8, 1, 1);
    let h = gates::hadamard();

    // Apply H to qubit 0
    apply_gate_1q_simd(&mut state, 0, &h);

    // Verify superposition
    let amp0 = state.state_vector[0].re;
    let amp1 = state.state_vector[1].re;
    let expected = 1.0 / 2.0_f64.sqrt();

    assert!((amp0 - expected).abs() < EPSILON, "amp[0] wrong");
    assert!((amp1 - expected).abs() < EPSILON, "amp[1] wrong");
}

#[test]
fn test_simd_pauli_gates() {
    // Test X gate
    let mut state_x = TQPState::new(4, 1, 1);
    apply_gate_1q_simd(&mut state_x, 0, &gates::pauli_x());
    assert!((state_x.state_vector[1].re - 1.0).abs() < EPSILON);

    // Test Z gate on |+⟩
    let mut state_z = TQPState::new(4, 1, 1);
    apply_gate_1q_simd(&mut state_z, 0, &gates::hadamard());
    apply_gate_1q_simd(&mut state_z, 0, &gates::pauli_z());
    // Should give |−⟩ = (|0⟩ - |1⟩)/√2
    let expected = 1.0 / 2.0_f64.sqrt();
    assert!((state_z.state_vector[0].re - expected).abs() < EPSILON);
    assert!((state_z.state_vector[1].re + expected).abs() < EPSILON);
}

#[test]
fn test_simd_vs_scalar_extensive() {
    // Test on various qubit indices and state sizes
    for num_qubits in [4, 6, 8, 10] {
        for target_qubit in 0..num_qubits {
            let mut simd_state = TQPState::new(num_qubits, 2, 2);
            let mut scalar_state = simd_state.clone();

            let h = gates::hadamard();

            apply_gate_1q_simd(&mut simd_state, target_qubit, &h);
            apply_spatial_gate(&mut scalar_state, target_qubit, &h);

            for i in 0..simd_state.dimension() {
                let diff = (simd_state.state_vector[i] - scalar_state.state_vector[i]).norm();
                assert!(
                    diff < EPSILON,
                    "Mismatch: n={}, q={}, idx={}",
                    num_qubits,
                    target_qubit,
                    i
                );
            }
        }
    }
}

// =============================================================================
// 2Q Gate Tests
// =============================================================================

#[test]
fn test_simd_cnot_bell_state() {
    let mut state = TQPState::new(4, 1, 1);

    // Create Bell state
    apply_gate_1q_simd(&mut state, 0, &gates::hadamard());
    apply_gate_2q_simd(&mut state, 0, 1, &gates::cnot());

    // Verify (|00⟩ + |11⟩)/√2
    let expected = 1.0 / 2.0_f64.sqrt();
    assert!((state.state_vector[0].re - expected).abs() < EPSILON); // |00⟩
    assert!(state.state_vector[1].norm() < EPSILON); // |01⟩
    assert!(state.state_vector[2].norm() < EPSILON); // |10⟩
    assert!((state.state_vector[3].re - expected).abs() < EPSILON); // |11⟩
}

#[test]
fn test_simd_2q_vs_scalar() {
    let mut simd_state = TQPState::new(6, 2, 2);
    let mut scalar_state = simd_state.clone();

    // Setup
    apply_gate_1q_simd(&mut simd_state, 0, &gates::hadamard());
    apply_gate_1q_simd(&mut simd_state, 2, &gates::hadamard());
    apply_spatial_gate(&mut scalar_state, 0, &gates::hadamard());
    apply_spatial_gate(&mut scalar_state, 2, &gates::hadamard());

    // Apply 2Q gate
    apply_gate_2q_simd(&mut simd_state, 0, 2, &gates::cnot());
    apply_spatial_gate_2q(&mut scalar_state, 0, 2, &gates::cnot());

    // Compare
    for i in 0..simd_state.dimension() {
        let diff = (simd_state.state_vector[i] - scalar_state.state_vector[i]).norm();
        assert!(diff < EPSILON, "2Q gate mismatch at idx {}", i);
    }
}

// =============================================================================
// Probability Tests
// =============================================================================

#[test]
fn test_simd_probability_conservation() {
    let mut state = TQPState::new(10, 2, 2);

    // Apply many gates
    for q in 0..10 {
        apply_gate_1q_simd(&mut state, q, &gates::hadamard());
    }

    let prob = total_probability_simd(&state);
    assert!(
        (prob - 1.0).abs() < EPSILON,
        "Probability not conserved: {}",
        prob
    );
}

#[test]
fn test_simd_probability_vs_direct() {
    let mut state = TQPState::new(8, 2, 2);

    apply_gate_1q_simd(&mut state, 0, &gates::hadamard());
    apply_gate_1q_simd(&mut state, 3, &gates::hadamard());

    let simd_prob = total_probability_simd(&state);
    let direct_prob: f64 = state.state_vector.iter().map(|c| c.norm_sqr()).sum();

    assert!((simd_prob - direct_prob).abs() < EPSILON);
}

// =============================================================================
// Batch Operations
// =============================================================================

#[test]
fn test_batch_gates_correctness() {
    let mut state = TQPState::new(6, 1, 1);

    let gates_list = vec![
        (0, gates::hadamard()),
        (1, gates::pauli_x()),
        (2, gates::hadamard()),
        (3, gates::pauli_z()),
    ];

    apply_gates_batch_simd(&mut state, &gates_list);

    // Verify normalization preserved
    let prob = total_probability_simd(&state);
    assert!((prob - 1.0).abs() < EPSILON);
}

// =============================================================================
// Performance Tests
// =============================================================================

#[test]
fn test_simd_performance() {
    use std::time::Instant;

    let num_qubits = 12;
    let num_iterations = 50;
    let h = gates::hadamard();

    // SIMD
    let mut simd_state = TQPState::new(num_qubits, 1, 1);
    let start = Instant::now();
    for _ in 0..num_iterations {
        for q in 0..num_qubits {
            apply_gate_1q_simd(&mut simd_state, q, &h);
        }
    }
    let simd_time = start.elapsed();

    // Scalar
    let mut scalar_state = TQPState::new(num_qubits, 1, 1);
    let start = Instant::now();
    for _ in 0..num_iterations {
        for q in 0..num_qubits {
            apply_spatial_gate(&mut scalar_state, q, &h);
        }
    }
    let scalar_time = start.elapsed();

    let speedup = scalar_time.as_secs_f64() / simd_time.as_secs_f64();

    println!("\n=== SIMD Performance Test ===");
    println!("Qubits: {}", num_qubits);
    println!("Iterations: {}", num_iterations);
    println!("Total gates: {}", num_qubits * num_iterations);
    println!("SIMD level: {:?}", SimdLevel::detect());
    println!("Scalar: {:?}", scalar_time);
    println!("SIMD: {:?}", simd_time);
    println!("Speedup: {:.2}x", speedup);

    // Should not be significantly slower
    assert!(speedup > 0.5, "SIMD should not be 2x slower than scalar");
}

// =============================================================================
// Circuit Tests
// =============================================================================

#[test]
fn test_simd_qft_like_circuit() {
    let num_qubits = 5;
    let mut state = TQPState::new(num_qubits, 1, 1);

    // QFT-like pattern: H on all, then controlled phases
    for q in 0..num_qubits {
        apply_gate_1q_simd(&mut state, q, &gates::hadamard());
    }

    for q in 0..num_qubits - 1 {
        apply_gate_2q_simd(&mut state, q, q + 1, &gates::cz());
    }

    // Verify normalization
    let prob = total_probability_simd(&state);
    assert!((prob - 1.0).abs() < EPSILON);

    // Verify we created entanglement (many non-zero amplitudes)
    let nnz = state
        .state_vector
        .iter()
        .filter(|c| c.norm_sqr() > 1e-14)
        .count();
    assert!(nnz > 1, "Should have multiple non-zero amplitudes");
}

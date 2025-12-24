//! Week 3 Day 1: Parameter-Shift Autodiff Integration Tests
//!
//! Validates:
//! - Parameterized gate matrices
//! - Variational circuit execution
//! - Parameter-shift gradient computation
//! - Gradient verification against finite differences

use std::f64::consts::PI;
use tqp_core::{
    compute_expectation, compute_expectation_and_gradient, compute_gradient, rx, ry, rz, u3,
    verify_gradient, GateType, Hamiltonian, ParameterizedGate, PauliObservable, VariationalCircuit,
    PARAMETER_SHIFT,
};

const EPSILON: f64 = 1e-8;
const GRADIENT_TOL: f64 = 1e-4;

// =============================================================================
// Gate Matrix Tests
// =============================================================================

#[test]
fn test_rotation_gates_identity() {
    // All rotation gates at θ=0 should be identity
    let rx_0 = rx(0.0);
    let ry_0 = ry(0.0);
    let rz_0 = rz(0.0);

    for gate in [rx_0, ry_0, rz_0] {
        assert!((gate[[0, 0]].re - 1.0).abs() < EPSILON);
        assert!((gate[[1, 1]].re - 1.0).abs() < EPSILON);
        assert!(gate[[0, 1]].norm() < EPSILON);
        assert!(gate[[1, 0]].norm() < EPSILON);
    }
}

#[test]
fn test_rotation_gates_pi() {
    // RX(π), RY(π), RZ(π) have specific forms
    let rx_pi = rx(PI);
    let ry_pi = ry(PI);
    let rz_pi = rz(PI);

    // RX(π) ≈ -iX (off-diagonal should be non-zero)
    assert!(rx_pi[[0, 0]].norm() < EPSILON);
    assert!(rx_pi[[0, 1]].norm() > 0.9);

    // RY(π) should flip qubit
    assert!(ry_pi[[0, 0]].norm() < EPSILON);
    assert!(ry_pi[[0, 1]].norm() > 0.9);

    // RZ(π) is diagonal
    assert!(rz_pi[[0, 1]].norm() < EPSILON);
    assert!(rz_pi[[1, 0]].norm() < EPSILON);
}

#[test]
fn test_u3_gate() {
    // U3(0, 0, 0) = I
    let u3_0 = u3(0.0, 0.0, 0.0);
    assert!((u3_0[[0, 0]].re - 1.0).abs() < EPSILON);
    assert!((u3_0[[1, 1]].re - 1.0).abs() < EPSILON);

    // U3(π, 0, π) = X
    let u3_x = u3(PI, 0.0, PI);
    assert!(u3_x[[0, 0]].norm() < EPSILON);
    assert!(u3_x[[0, 1]].norm() > 0.9);
    assert!(u3_x[[1, 0]].norm() > 0.9);
    assert!(u3_x[[1, 1]].norm() < EPSILON);
}

// =============================================================================
// Parameterized Gate Tests
// =============================================================================

#[test]
fn test_parameterized_gate_types() {
    let rx_gate = ParameterizedGate::rx(0, 0);
    let ry_gate = ParameterizedGate::ry(1, 1);
    let rz_gate = ParameterizedGate::rz(2, 2);

    assert_eq!(rx_gate.gate_type, GateType::RX);
    assert_eq!(ry_gate.gate_type, GateType::RY);
    assert_eq!(rz_gate.gate_type, GateType::RZ);

    assert_eq!(rx_gate.qubits, vec![0]);
    assert_eq!(ry_gate.qubits, vec![1]);
    assert_eq!(rz_gate.qubits, vec![2]);
}

#[test]
fn test_parameterized_gate_uses_param() {
    let gate = ParameterizedGate::rx(0, 3);

    assert!(gate.uses_param(3));
    assert!(!gate.uses_param(0));
    assert!(!gate.uses_param(1));
}

#[test]
fn test_parameterized_gate_num_params() {
    let rx = ParameterizedGate::rx(0, 0);
    let ry = ParameterizedGate::ry(0, 0);
    let rz = ParameterizedGate::rz(0, 0);

    assert_eq!(rx.num_params(), 1);
    assert_eq!(ry.num_params(), 1);
    assert_eq!(rz.num_params(), 1);
}

// =============================================================================
// Variational Circuit Tests
// =============================================================================

#[test]
fn test_circuit_construction() {
    let mut circuit = VariationalCircuit::new(4);

    let p0 = circuit.ry(0);
    let p1 = circuit.ry(1);
    let p2 = circuit.rz(2);
    circuit.h(3);
    circuit.cnot(0, 1);

    assert_eq!(p0, 0);
    assert_eq!(p1, 1);
    assert_eq!(p2, 2);
    assert_eq!(circuit.num_params(), 3);
}

#[test]
fn test_circuit_execution_ground_state() {
    // Circuit with θ=0 should preserve |0⟩
    let mut circuit = VariationalCircuit::new(2);
    circuit.ry(0);
    circuit.ry(1);

    let params = vec![0.0, 0.0];
    let state = circuit.execute(&params);

    // Should be |00⟩
    assert!((state.state_vector[0].re - 1.0).abs() < EPSILON);
    for i in 1..4 {
        assert!(state.state_vector[i].norm() < EPSILON);
    }
}

#[test]
fn test_circuit_execution_superposition() {
    // RY(π/2)|0⟩ = (|0⟩ + |1⟩)/√2
    let mut circuit = VariationalCircuit::new(1);
    circuit.ry(0);

    let params = vec![PI / 2.0];
    let state = circuit.execute(&params);

    let expected = 1.0 / 2.0_f64.sqrt();
    assert!((state.state_vector[0].re - expected).abs() < EPSILON);
    assert!((state.state_vector[1].re - expected).abs() < EPSILON);
}

#[test]
fn test_circuit_execution_bell_state() {
    // RY(π/2)|0⟩ on qubit 0, then CNOT(0,1) gives Bell state
    let mut circuit = VariationalCircuit::new(2);
    circuit.ry(0);
    circuit.cnot(0, 1);

    let params = vec![PI / 2.0];
    let state = circuit.execute(&params);

    // Should be (|00⟩ + |11⟩)/√2
    let expected = 1.0 / 2.0_f64.sqrt();
    assert!((state.state_vector[0].re - expected).abs() < EPSILON);
    assert!(state.state_vector[1].norm() < EPSILON);
    assert!(state.state_vector[2].norm() < EPSILON);
    assert!((state.state_vector[3].re - expected).abs() < EPSILON);
}

// =============================================================================
// Observable Tests
// =============================================================================

#[test]
fn test_z_observable_ground_state() {
    use tqp_core::TQPState;

    // ⟨0|Z|0⟩ = 1
    let state = TQPState::new(1, 1, 1);
    let z = PauliObservable::z(0);

    assert!((z.expectation(&state) - 1.0).abs() < EPSILON);
}

#[test]
fn test_z_observable_excited_state() {
    use tqp_core::gates;
    use tqp_core::ops::apply_spatial_gate;
    use tqp_core::TQPState;

    // ⟨1|Z|1⟩ = -1
    let mut state = TQPState::new(1, 1, 1);
    apply_spatial_gate(&mut state, 0, &gates::pauli_x()); // |0⟩ -> |1⟩

    let z = PauliObservable::z(0);
    assert!((z.expectation(&state) - (-1.0)).abs() < EPSILON);
}

#[test]
fn test_zz_observable() {
    use tqp_core::TQPState;

    // ⟨00|Z₀Z₁|00⟩ = 1
    let state = TQPState::new(2, 1, 1);
    let zz = PauliObservable::zz(0, 1);

    assert!((zz.expectation(&state) - 1.0).abs() < EPSILON);
}

#[test]
fn test_hamiltonian_all_z() {
    use tqp_core::TQPState;

    let state = TQPState::new(3, 1, 1);
    let h = Hamiltonian::all_z(3);

    // ⟨000|Z₀+Z₁+Z₂|000⟩ = 3
    assert!((h.expectation(&state) - 3.0).abs() < EPSILON);
}

#[test]
fn test_hamiltonian_ising() {
    use tqp_core::TQPState;

    let state = TQPState::new(4, 1, 1);
    let h = Hamiltonian::ising(4);

    // ⟨0000|Z₀Z₁+Z₁Z₂+Z₂Z₃|0000⟩ = 3
    assert!((h.expectation(&state) - 3.0).abs() < EPSILON);
}

// =============================================================================
// Gradient Tests
// =============================================================================

#[test]
fn test_gradient_single_rx() {
    // For RX(θ)|0⟩, ⟨Z⟩ = cos(θ)
    // ∂⟨Z⟩/∂θ = -sin(θ)
    let mut circuit = VariationalCircuit::new(1);
    circuit.rx(0);

    let h = Hamiltonian::all_z(1);

    // Test at θ = π/3
    let theta = PI / 3.0;
    let params = vec![theta];
    let gradient = compute_gradient(&circuit, &params, &h);

    let expected = -theta.sin();
    assert!(
        (gradient[0] - expected).abs() < GRADIENT_TOL,
        "Gradient mismatch: got {} expected {}",
        gradient[0],
        expected
    );
}

#[test]
fn test_gradient_single_ry() {
    // For RY(θ)|0⟩, ⟨Z⟩ = cos(θ)
    // ∂⟨Z⟩/∂θ = -sin(θ)
    let mut circuit = VariationalCircuit::new(1);
    circuit.ry(0);

    let h = Hamiltonian::all_z(1);

    let theta = PI / 4.0;
    let params = vec![theta];
    let gradient = compute_gradient(&circuit, &params, &h);

    let expected = -theta.sin();
    assert!(
        (gradient[0] - expected).abs() < GRADIENT_TOL,
        "Gradient mismatch: got {} expected {}",
        gradient[0],
        expected
    );
}

#[test]
fn test_gradient_at_extrema() {
    // At θ=0, cos(θ) has zero gradient
    // At θ=π/2, cos(θ) has maximum gradient magnitude
    let mut circuit = VariationalCircuit::new(1);
    circuit.ry(0);

    let h = Hamiltonian::all_z(1);

    // At θ=0: gradient should be 0
    let grad_0 = compute_gradient(&circuit, &[0.0], &h);
    assert!(grad_0[0].abs() < GRADIENT_TOL);

    // At θ=π/2: gradient should be -1
    let grad_pi2 = compute_gradient(&circuit, &[PI / 2.0], &h);
    assert!((grad_pi2[0] - (-1.0)).abs() < GRADIENT_TOL);
}

#[test]
fn test_gradient_multi_param() {
    let mut circuit = VariationalCircuit::new(2);
    circuit.ry(0);
    circuit.ry(1);

    let h = Hamiltonian::all_z(2);
    let params = vec![0.5, 1.0];

    let gradients = compute_gradient(&circuit, &params, &h);

    assert_eq!(gradients.len(), 2);
    // Both gradients should be non-zero for these parameter values
    assert!(gradients[0].abs() > 0.1);
    assert!(gradients[1].abs() > 0.1);
}

#[test]
fn test_gradient_verification_simple() {
    let mut circuit = VariationalCircuit::new(1);
    circuit.ry(0);

    let h = Hamiltonian::all_z(1);
    let params = vec![0.7];

    let (analytic, numeric, max_diff) = verify_gradient(&circuit, &params, &h, 1e-6);

    assert!(
        max_diff < GRADIENT_TOL,
        "Gradient verification failed: analytic={:?}, numeric={:?}, diff={}",
        analytic,
        numeric,
        max_diff
    );
}

#[test]
fn test_gradient_verification_complex_circuit() {
    let mut circuit = VariationalCircuit::new(3);
    circuit.ry(0);
    circuit.ry(1);
    circuit.ry(2);
    circuit.cnot(0, 1);
    circuit.cnot(1, 2);
    circuit.rz(0);
    circuit.rz(1);

    let h = Hamiltonian::ising(3);
    let params = vec![0.3, 0.7, 1.2, 0.5, 0.9];

    let (_analytic, _numeric, max_diff) = verify_gradient(&circuit, &params, &h, 1e-6);

    assert!(
        max_diff < GRADIENT_TOL,
        "Complex circuit gradient verification failed: max_diff={}",
        max_diff
    );
}

// =============================================================================
// Combined Function Tests
// =============================================================================

#[test]
fn test_expectation_computation() {
    let mut circuit = VariationalCircuit::new(1);
    circuit.ry(0);

    let h = Hamiltonian::all_z(1);

    // At θ=0: ⟨Z⟩ = cos(0) = 1
    let exp_0 = compute_expectation(&circuit, &[0.0], &h);
    assert!((exp_0 - 1.0).abs() < EPSILON);

    // At θ=π: ⟨Z⟩ = cos(π) = -1
    let exp_pi = compute_expectation(&circuit, &[PI], &h);
    assert!((exp_pi - (-1.0)).abs() < EPSILON);

    // At θ=π/2: ⟨Z⟩ = cos(π/2) = 0
    let exp_pi2 = compute_expectation(&circuit, &[PI / 2.0], &h);
    assert!(exp_pi2.abs() < EPSILON);
}

#[test]
fn test_expectation_and_gradient_combined() {
    let mut circuit = VariationalCircuit::new(2);
    circuit.ry(0);
    circuit.cnot(0, 1);
    circuit.ry(1);

    let h = Hamiltonian::all_z(2);
    let params = vec![PI / 4.0, PI / 3.0];

    let (exp, grad) = compute_expectation_and_gradient(&circuit, &params, &h);

    // Verify consistency with separate calls
    let exp_separate = compute_expectation(&circuit, &params, &h);
    let grad_separate = compute_gradient(&circuit, &params, &h);

    assert!((exp - exp_separate).abs() < EPSILON);
    for i in 0..grad.len() {
        assert!((grad[i] - grad_separate[i]).abs() < EPSILON);
    }
}

#[test]
fn test_parameter_shift_constant() {
    // Verify PARAMETER_SHIFT = π/2
    assert!((PARAMETER_SHIFT - PI / 2.0).abs() < EPSILON);
}

// =============================================================================
// VQE-like Scenario Tests
// =============================================================================

#[test]
fn test_vqe_ansatz() {
    // Hardware-efficient ansatz: RY layer + entangling layer
    let num_qubits = 4;
    let mut circuit = VariationalCircuit::new(num_qubits);

    // Layer 1: RY on all qubits
    for q in 0..num_qubits {
        circuit.ry(q);
    }

    // Entangling layer
    for q in 0..(num_qubits - 1) {
        circuit.cnot(q, q + 1);
    }

    // Layer 2: RY on all qubits
    for q in 0..num_qubits {
        circuit.ry(q);
    }

    assert_eq!(circuit.num_params(), 2 * num_qubits);

    let h = Hamiltonian::ising(num_qubits);
    let params: Vec<f64> = (0..2 * num_qubits).map(|i| 0.1 * i as f64).collect();

    let (exp, grad) = compute_expectation_and_gradient(&circuit, &params, &h);

    // Expectation should be in valid range
    assert!(exp >= -(num_qubits as f64 - 1.0));
    assert!(exp <= (num_qubits as f64 - 1.0));

    // All gradients should be computed
    assert_eq!(grad.len(), 2 * num_qubits);
}

#[test]
fn test_gradient_descent_step() {
    // Verify that following the negative gradient decreases energy
    let mut circuit = VariationalCircuit::new(2);
    circuit.ry(0);
    circuit.ry(1);
    circuit.cnot(0, 1);

    let h = Hamiltonian::all_z(2);

    // Start from random-ish parameters
    let params = vec![1.0, 0.5];
    let (exp_0, grad) = compute_expectation_and_gradient(&circuit, &params, &h);

    // Take gradient descent step
    let lr = 0.1;
    let new_params: Vec<f64> = params
        .iter()
        .zip(grad.iter())
        .map(|(p, g)| p - lr * g)
        .collect();

    let exp_1 = compute_expectation(&circuit, &new_params, &h);

    // Energy should decrease (or stay same if at minimum)
    assert!(
        exp_1 <= exp_0 + 1e-6,
        "Energy should decrease: {} -> {}",
        exp_0,
        exp_1
    );
}

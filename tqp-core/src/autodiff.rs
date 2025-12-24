//! Automatic Differentiation for Variational Quantum Circuits
//!
//! Implements the Parameter-Shift Rule for computing gradients of
//! quantum expectation values with respect to circuit parameters.
//!
//! # Parameter-Shift Rule
//!
//! For a parameterized gate U(θ) = exp(-iθG/2) where G is a generator
//! with eigenvalues ±1, the gradient is:
//!
//! ```text
//! ∂⟨O⟩/∂θ = (⟨O⟩_{θ+π/2} - ⟨O⟩_{θ-π/2}) / 2
//! ```
//!
//! This provides exact gradients without approximation errors.
//!
//! # Supported Gates
//!
//! - RX(θ): Rotation around X-axis
//! - RY(θ): Rotation around Y-axis
//! - RZ(θ): Rotation around Z-axis
//! - CNOT: Not parameterized (gradient = 0)

use ndarray::Array2;
use num_complex::Complex64;
use std::f64::consts::PI;

use crate::ops::{apply_spatial_gate, apply_spatial_gate_2q};
use crate::sparse_ops::gates;
use crate::state::TQPState;

// =============================================================================
// Constants
// =============================================================================

/// Shift amount for parameter-shift rule (π/2)
pub const PARAMETER_SHIFT: f64 = PI / 2.0;

/// Small epsilon for numerical stability
pub const GRADIENT_EPSILON: f64 = 1e-10;

// =============================================================================
// Parameterized Gates
// =============================================================================

/// Rotation around X-axis: RX(θ) = exp(-iθX/2)
pub fn rx(theta: f64) -> Array2<Complex64> {
    let c = Complex64::new((theta / 2.0).cos(), 0.0);
    let s = Complex64::new(0.0, -(theta / 2.0).sin());
    Array2::from_shape_vec((2, 2), vec![c, s, s, c]).unwrap()
}

/// Rotation around Y-axis: RY(θ) = exp(-iθY/2)
pub fn ry(theta: f64) -> Array2<Complex64> {
    let c = Complex64::new((theta / 2.0).cos(), 0.0);
    let s = Complex64::new((theta / 2.0).sin(), 0.0);
    Array2::from_shape_vec((2, 2), vec![c, -s, s, c]).unwrap()
}

/// Rotation around Z-axis: RZ(θ) = exp(-iθZ/2)
pub fn rz(theta: f64) -> Array2<Complex64> {
    let e_neg = Complex64::new(0.0, -theta / 2.0).exp();
    let e_pos = Complex64::new(0.0, theta / 2.0).exp();
    Array2::from_shape_vec(
        (2, 2),
        vec![
            e_neg,
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            e_pos,
        ],
    )
    .unwrap()
}

/// General rotation: U(θ, φ, λ) (U3 gate)
pub fn u3(theta: f64, phi: f64, lambda: f64) -> Array2<Complex64> {
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();

    let u00 = Complex64::new(c, 0.0);
    let u01 = -Complex64::new(0.0, lambda).exp() * s;
    let u10 = Complex64::new(0.0, phi).exp() * s;
    let u11 = Complex64::new(0.0, phi + lambda).exp() * c;

    Array2::from_shape_vec((2, 2), vec![u00, u01, u10, u11]).unwrap()
}

// =============================================================================
// Parameterized Gate Description
// =============================================================================

/// Type of parameterized gate
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GateType {
    RX,
    RY,
    RZ,
    U3,  // Has 3 parameters
    CRX, // Controlled-RX (2-qubit)
    CRY, // Controlled-RY (2-qubit)
    CRZ, // Controlled-RZ (2-qubit)
}

/// A parameterized gate in a circuit
#[derive(Debug, Clone)]
pub struct ParameterizedGate {
    /// Gate type
    pub gate_type: GateType,
    /// Target qubit(s)
    pub qubits: Vec<usize>,
    /// Parameter index(es) in the parameter vector
    pub param_indices: Vec<usize>,
}

impl ParameterizedGate {
    pub fn rx(qubit: usize, param_idx: usize) -> Self {
        ParameterizedGate {
            gate_type: GateType::RX,
            qubits: vec![qubit],
            param_indices: vec![param_idx],
        }
    }

    pub fn ry(qubit: usize, param_idx: usize) -> Self {
        ParameterizedGate {
            gate_type: GateType::RY,
            qubits: vec![qubit],
            param_indices: vec![param_idx],
        }
    }

    pub fn rz(qubit: usize, param_idx: usize) -> Self {
        ParameterizedGate {
            gate_type: GateType::RZ,
            qubits: vec![qubit],
            param_indices: vec![param_idx],
        }
    }

    /// Check if this gate affects a given parameter
    pub fn uses_param(&self, param_idx: usize) -> bool {
        self.param_indices.contains(&param_idx)
    }

    /// Get the gate matrix for given parameters
    pub fn matrix(&self, params: &[f64]) -> Array2<Complex64> {
        match self.gate_type {
            GateType::RX => rx(params[self.param_indices[0]]),
            GateType::RY => ry(params[self.param_indices[0]]),
            GateType::RZ => rz(params[self.param_indices[0]]),
            GateType::U3 => u3(
                params[self.param_indices[0]],
                params[self.param_indices[1]],
                params[self.param_indices[2]],
            ),
            _ => unimplemented!("Controlled gates not yet implemented"),
        }
    }

    /// Number of parameters this gate uses
    pub fn num_params(&self) -> usize {
        match self.gate_type {
            GateType::U3 => 3,
            _ => 1,
        }
    }
}

// =============================================================================
// Variational Circuit
// =============================================================================

/// A variational quantum circuit with parameterized gates
#[derive(Debug, Clone)]
pub struct VariationalCircuit {
    /// Number of qubits
    pub num_qubits: usize,
    /// List of gates (parameterized and fixed)
    gates: Vec<CircuitGate>,
    /// Number of parameters
    num_params: usize,
}

/// A gate in the circuit (parameterized or fixed)
#[derive(Debug, Clone)]
enum CircuitGate {
    Parameterized(ParameterizedGate),
    Fixed1Q {
        qubit: usize,
        matrix: Array2<Complex64>,
    },
    Fixed2Q {
        qubit1: usize,
        qubit2: usize,
        matrix: Array2<Complex64>,
    },
}

impl VariationalCircuit {
    pub fn new(num_qubits: usize) -> Self {
        VariationalCircuit {
            num_qubits,
            gates: Vec::new(),
            num_params: 0,
        }
    }

    /// Add a parameterized RX gate
    pub fn rx(&mut self, qubit: usize) -> usize {
        let param_idx = self.num_params;
        self.gates
            .push(CircuitGate::Parameterized(ParameterizedGate::rx(
                qubit, param_idx,
            )));
        self.num_params += 1;
        param_idx
    }

    /// Add a parameterized RY gate
    pub fn ry(&mut self, qubit: usize) -> usize {
        let param_idx = self.num_params;
        self.gates
            .push(CircuitGate::Parameterized(ParameterizedGate::ry(
                qubit, param_idx,
            )));
        self.num_params += 1;
        param_idx
    }

    /// Add a parameterized RZ gate
    pub fn rz(&mut self, qubit: usize) -> usize {
        let param_idx = self.num_params;
        self.gates
            .push(CircuitGate::Parameterized(ParameterizedGate::rz(
                qubit, param_idx,
            )));
        self.num_params += 1;
        param_idx
    }

    /// Add a fixed Hadamard gate
    pub fn h(&mut self, qubit: usize) {
        self.gates.push(CircuitGate::Fixed1Q {
            qubit,
            matrix: gates::hadamard(),
        });
    }

    /// Add a fixed CNOT gate
    pub fn cnot(&mut self, control: usize, target: usize) {
        self.gates.push(CircuitGate::Fixed2Q {
            qubit1: control,
            qubit2: target,
            matrix: gates::cnot(),
        });
    }

    /// Add a fixed CZ gate
    pub fn cz(&mut self, qubit1: usize, qubit2: usize) {
        self.gates.push(CircuitGate::Fixed2Q {
            qubit1,
            qubit2,
            matrix: gates::cz(),
        });
    }

    /// Get number of parameters
    pub fn num_params(&self) -> usize {
        self.num_params
    }

    /// Execute circuit with given parameters
    pub fn execute(&self, params: &[f64]) -> TQPState {
        assert_eq!(params.len(), self.num_params, "Parameter count mismatch");

        let mut state = TQPState::new(self.num_qubits, 1, 1);

        for gate in &self.gates {
            match gate {
                CircuitGate::Parameterized(pg) => {
                    let matrix = pg.matrix(params);
                    apply_spatial_gate(&mut state, pg.qubits[0], &matrix);
                }
                CircuitGate::Fixed1Q { qubit, matrix } => {
                    apply_spatial_gate(&mut state, *qubit, matrix);
                }
                CircuitGate::Fixed2Q {
                    qubit1,
                    qubit2,
                    matrix,
                } => {
                    apply_spatial_gate_2q(&mut state, *qubit1, *qubit2, matrix);
                }
            }
        }

        state
    }

    /// Execute with shifted parameter (for parameter-shift rule)
    fn execute_with_shift(&self, params: &[f64], param_idx: usize, shift: f64) -> TQPState {
        let mut shifted_params = params.to_vec();
        shifted_params[param_idx] += shift;
        self.execute(&shifted_params)
    }
}

// =============================================================================
// Observable (Hamiltonian)
// =============================================================================

/// A Pauli string observable (e.g., Z0, X0X1, Z0Z1Z2)
#[derive(Debug, Clone)]
pub struct PauliObservable {
    /// Coefficient
    pub coeff: f64,
    /// Pauli operators: (qubit, pauli_type) where pauli_type is 'I', 'X', 'Y', 'Z'
    pub terms: Vec<(usize, char)>,
}

impl PauliObservable {
    /// Create Z observable on a single qubit
    pub fn z(qubit: usize) -> Self {
        PauliObservable {
            coeff: 1.0,
            terms: vec![(qubit, 'Z')],
        }
    }

    /// Create X observable on a single qubit
    pub fn x(qubit: usize) -> Self {
        PauliObservable {
            coeff: 1.0,
            terms: vec![(qubit, 'X')],
        }
    }

    /// Create ZZ observable on two qubits
    pub fn zz(qubit1: usize, qubit2: usize) -> Self {
        PauliObservable {
            coeff: 1.0,
            terms: vec![(qubit1, 'Z'), (qubit2, 'Z')],
        }
    }

    /// Compute expectation value ⟨ψ|O|ψ⟩
    pub fn expectation(&self, state: &TQPState) -> f64 {
        let dim = state.dimension();
        let mut result = 0.0;

        // For each basis state |i⟩
        for i in 0..dim {
            let amplitude = state.state_vector[i];
            let prob = amplitude.norm_sqr();

            if prob < 1e-16 {
                continue;
            }

            // Compute eigenvalue of Pauli string for this basis state
            let mut eigenvalue = 1.0;
            for &(qubit, pauli) in &self.terms {
                let bit = (i >> qubit) & 1;
                match pauli {
                    'Z' => {
                        // Z eigenvalue: +1 for |0⟩, -1 for |1⟩
                        eigenvalue *= if bit == 0 { 1.0 } else { -1.0 };
                    }
                    'I' => {
                        // Identity: eigenvalue = 1
                    }
                    'X' | 'Y' => {
                        // X and Y are more complex - need to handle off-diagonal
                        // For now, only support diagonal observables
                        unimplemented!("X/Y observables require off-diagonal handling");
                    }
                    _ => panic!("Unknown Pauli operator"),
                }
            }

            result += prob * eigenvalue * self.coeff;
        }

        result
    }
}

/// A Hamiltonian as a sum of Pauli observables
#[derive(Debug, Clone)]
pub struct Hamiltonian {
    pub terms: Vec<PauliObservable>,
}

impl Hamiltonian {
    pub fn new() -> Self {
        Hamiltonian { terms: Vec::new() }
    }

    pub fn add_term(&mut self, term: PauliObservable) {
        self.terms.push(term);
    }

    /// Compute total expectation value
    pub fn expectation(&self, state: &TQPState) -> f64 {
        self.terms.iter().map(|t| t.expectation(state)).sum()
    }

    /// Simple Z-only Hamiltonian: H = Σ Z_i
    pub fn all_z(num_qubits: usize) -> Self {
        let mut h = Hamiltonian::new();
        for q in 0..num_qubits {
            h.add_term(PauliObservable::z(q));
        }
        h
    }

    /// Ising model: H = Σ Z_i Z_{i+1}
    pub fn ising(num_qubits: usize) -> Self {
        let mut h = Hamiltonian::new();
        for q in 0..(num_qubits - 1) {
            h.add_term(PauliObservable::zz(q, q + 1));
        }
        h
    }
}

impl Default for Hamiltonian {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Parameter-Shift Gradient Computation
// =============================================================================

/// Compute gradient of ⟨H⟩ with respect to θ using parameter-shift rule
///
/// ∂⟨H⟩/∂θ_i = (⟨H⟩_{θ_i + π/2} - ⟨H⟩_{θ_i - π/2}) / 2
pub fn compute_gradient(
    circuit: &VariationalCircuit,
    params: &[f64],
    hamiltonian: &Hamiltonian,
) -> Vec<f64> {
    let num_params = circuit.num_params();
    let mut gradients = vec![0.0; num_params];

    for (i, grad) in gradients.iter_mut().enumerate() {
        // θ + π/2
        let state_plus = circuit.execute_with_shift(params, i, PARAMETER_SHIFT);
        let exp_plus = hamiltonian.expectation(&state_plus);

        // θ - π/2
        let state_minus = circuit.execute_with_shift(params, i, -PARAMETER_SHIFT);
        let exp_minus = hamiltonian.expectation(&state_minus);

        // Gradient = (f(θ+π/2) - f(θ-π/2)) / 2
        *grad = (exp_plus - exp_minus) / 2.0;
    }

    gradients
}

/// Compute expectation value for given parameters
pub fn compute_expectation(
    circuit: &VariationalCircuit,
    params: &[f64],
    hamiltonian: &Hamiltonian,
) -> f64 {
    let state = circuit.execute(params);
    hamiltonian.expectation(&state)
}

/// Compute both expectation and gradient in one pass (more efficient)
pub fn compute_expectation_and_gradient(
    circuit: &VariationalCircuit,
    params: &[f64],
    hamiltonian: &Hamiltonian,
) -> (f64, Vec<f64>) {
    let expectation = compute_expectation(circuit, params, hamiltonian);
    let gradient = compute_gradient(circuit, params, hamiltonian);
    (expectation, gradient)
}

// =============================================================================
// Gradient Verification
// =============================================================================

/// Verify gradient using finite differences
pub fn verify_gradient(
    circuit: &VariationalCircuit,
    params: &[f64],
    hamiltonian: &Hamiltonian,
    epsilon: f64,
) -> (Vec<f64>, Vec<f64>, f64) {
    let analytic = compute_gradient(circuit, params, hamiltonian);

    let mut numeric = vec![0.0; params.len()];
    for i in 0..params.len() {
        let mut params_plus = params.to_vec();
        let mut params_minus = params.to_vec();
        params_plus[i] += epsilon;
        params_minus[i] -= epsilon;

        let exp_plus = compute_expectation(circuit, &params_plus, hamiltonian);
        let exp_minus = compute_expectation(circuit, &params_minus, hamiltonian);

        numeric[i] = (exp_plus - exp_minus) / (2.0 * epsilon);
    }

    // Compute max difference
    let max_diff = analytic
        .iter()
        .zip(numeric.iter())
        .map(|(a, n)| (a - n).abs())
        .fold(0.0_f64, f64::max);

    (analytic, numeric, max_diff)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const EPSILON: f64 = 1e-8;

    #[test]
    fn test_rx_gate() {
        let rx_0 = rx(0.0);
        // RX(0) = I
        assert!((rx_0[[0, 0]].re - 1.0).abs() < EPSILON);
        assert!(rx_0[[0, 1]].norm() < EPSILON);

        let rx_pi = rx(PI);
        // RX(π) = -iX
        assert!(rx_pi[[0, 0]].norm() < EPSILON);
        assert!((rx_pi[[0, 1]].im - (-1.0)).abs() < EPSILON);
    }

    #[test]
    fn test_ry_gate() {
        let ry_0 = ry(0.0);
        // RY(0) = I
        assert!((ry_0[[0, 0]].re - 1.0).abs() < EPSILON);

        let ry_pi = ry(PI);
        // RY(π) ≈ iY (up to global phase)
        assert!(ry_pi[[0, 0]].norm() < EPSILON);
    }

    #[test]
    fn test_rz_gate() {
        let rz_0 = rz(0.0);
        // RZ(0) = I
        assert!((rz_0[[0, 0]].re - 1.0).abs() < EPSILON);
        assert!((rz_0[[1, 1]].re - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_circuit_execution() {
        let mut circuit = VariationalCircuit::new(2);
        circuit.ry(0);
        circuit.cnot(0, 1);

        let params = vec![PI / 2.0];
        let state = circuit.execute(&params);

        // RY(π/2)|0⟩ = (|0⟩ + |1⟩)/√2, then CNOT gives Bell state
        let prob_00 = state.state_vector[0].norm_sqr();
        let prob_11 = state.state_vector[3].norm_sqr();

        assert!((prob_00 - 0.5).abs() < EPSILON);
        assert!((prob_11 - 0.5).abs() < EPSILON);
    }

    #[test]
    fn test_z_expectation() {
        // |0⟩ state: ⟨Z⟩ = 1
        let state = TQPState::new(1, 1, 1);
        let z = PauliObservable::z(0);
        assert!((z.expectation(&state) - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_gradient_rx() {
        // For RX(θ)|0⟩, ⟨Z⟩ = cos(θ)
        // So ∂⟨Z⟩/∂θ = -sin(θ)
        let mut circuit = VariationalCircuit::new(1);
        circuit.rx(0);

        let h = Hamiltonian::all_z(1);

        for theta in [0.0, PI / 4.0, PI / 2.0, PI] {
            let params = vec![theta];
            let gradient = compute_gradient(&circuit, &params, &h);
            let expected = -theta.sin();

            assert!(
                (gradient[0] - expected).abs() < 1e-6,
                "At θ={}: got {} expected {}",
                theta,
                gradient[0],
                expected
            );
        }
    }

    #[test]
    fn test_gradient_ry() {
        // For RY(θ)|0⟩, ⟨Z⟩ = cos(θ)
        // So ∂⟨Z⟩/∂θ = -sin(θ)
        let mut circuit = VariationalCircuit::new(1);
        circuit.ry(0);

        let h = Hamiltonian::all_z(1);

        for theta in [0.0, PI / 4.0, PI / 2.0, PI] {
            let params = vec![theta];
            let gradient = compute_gradient(&circuit, &params, &h);
            let expected = -theta.sin();

            assert!(
                (gradient[0] - expected).abs() < 1e-6,
                "At θ={}: got {} expected {}",
                theta,
                gradient[0],
                expected
            );
        }
    }

    #[test]
    fn test_gradient_verification() {
        let mut circuit = VariationalCircuit::new(2);
        circuit.ry(0);
        circuit.ry(1);
        circuit.cnot(0, 1);

        let h = Hamiltonian::all_z(2);
        let params = vec![0.5, 1.0];

        let (analytic, numeric, max_diff) = verify_gradient(&circuit, &params, &h, 1e-6);

        println!("Analytic: {:?}", analytic);
        println!("Numeric: {:?}", numeric);
        println!("Max diff: {}", max_diff);

        assert!(max_diff < 1e-4, "Gradient mismatch: {}", max_diff);
    }

    #[test]
    fn test_multi_param_gradient() {
        let mut circuit = VariationalCircuit::new(3);
        circuit.ry(0);
        circuit.ry(1);
        circuit.ry(2);
        circuit.cnot(0, 1);
        circuit.cnot(1, 2);

        let h = Hamiltonian::ising(3);
        let params = vec![0.3, 0.7, 1.2];

        let (_analytic, _numeric, max_diff) = verify_gradient(&circuit, &params, &h, 1e-6);

        assert!(max_diff < 1e-4, "Gradient mismatch: {}", max_diff);
    }

    #[test]
    fn test_expectation_and_gradient() {
        let mut circuit = VariationalCircuit::new(2);
        circuit.ry(0);
        circuit.cnot(0, 1);

        let h = Hamiltonian::all_z(2);
        let params = vec![PI / 4.0];

        let (exp, grad) = compute_expectation_and_gradient(&circuit, &params, &h);

        // Verify expectation is in valid range [-2, 2]
        assert!(exp >= -2.0 && exp <= 2.0);

        // Verify gradient is non-zero for non-extremal point
        assert!(grad[0].abs() > 0.1);
    }

    #[test]
    fn test_parameterized_gate_matrix() {
        let pg = ParameterizedGate::rx(0, 0);
        let params = vec![PI / 2.0];
        let matrix = pg.matrix(&params);

        // RX(π/2) should have specific form
        let expected_diag = (PI / 4.0).cos();
        assert!((matrix[[0, 0]].re - expected_diag).abs() < EPSILON);
    }
}

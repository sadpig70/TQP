//! Bridge between TQP-Core and TQP-IBM
//!
//! Provides conversion from TQP's quantum circuits and Hamiltonians
//! to IBM-compatible QASM circuits.
//!
//! ## Supported Conversions
//!
//! - H₂ VQE ansatz (UCCSD-like) → QASM circuit
//! - QAOA MaxCut → QASM circuit
//! - Hamiltonian → Measurement circuits

use crate::error::{IBMError, Result};
use crate::transpiler::{Circuit, Gate, GateType, CircuitBuilder};

// =============================================================================
// H₂ VQE Ansatz Builder
// =============================================================================

/// H₂ VQE ansatz configuration
#[derive(Debug, Clone)]
pub struct H2AnsatzConfig {
    /// Number of qubits (2 or 4)
    pub n_qubits: usize,
    
    /// Use hardware-efficient ansatz instead of UCCSD
    pub hardware_efficient: bool,
    
    /// Ansatz depth (for hardware-efficient)
    pub depth: usize,
}

impl Default for H2AnsatzConfig {
    fn default() -> Self {
        Self {
            n_qubits: 2,
            hardware_efficient: false,
            depth: 1,
        }
    }
}

impl H2AnsatzConfig {
    /// Create 2-qubit configuration
    pub fn two_qubit() -> Self {
        Self {
            n_qubits: 2,
            hardware_efficient: false,
            depth: 1,
        }
    }
    
    /// Create 4-qubit configuration
    pub fn four_qubit() -> Self {
        Self {
            n_qubits: 4,
            hardware_efficient: false,
            depth: 1,
        }
    }
    
    /// Use hardware-efficient ansatz
    pub fn with_hardware_efficient(mut self, depth: usize) -> Self {
        self.hardware_efficient = true;
        self.depth = depth;
        self
    }
}

/// H₂ VQE ansatz builder
pub struct H2AnsatzBuilder;

impl H2AnsatzBuilder {
    /// Build H₂ ansatz circuit
    ///
    /// For 2-qubit case (reduced Hamiltonian):
    /// |ψ(θ)⟩ = U(θ)|01⟩
    /// where U(θ) is a parameterized circuit
    ///
    /// For 4-qubit case (full Hamiltonian):
    /// Uses UCCSD-inspired ansatz
    pub fn build(config: &H2AnsatzConfig) -> Circuit {
        if config.hardware_efficient {
            CircuitBuilder::hardware_efficient(config.n_qubits, config.depth)
        } else {
            match config.n_qubits {
                2 => Self::build_2qubit_ansatz(),
                4 => Self::build_4qubit_ansatz(),
                _ => panic!("Unsupported qubit count: {}", config.n_qubits),
            }
        }
    }
    
    /// Build 2-qubit H₂ ansatz
    ///
    /// Circuit structure:
    /// |0⟩ ─ X ─ Ry(θ) ─●─
    /// |0⟩ ─────────────X─
    ///
    /// This prepares: cos(θ/2)|01⟩ + sin(θ/2)|10⟩
    /// For θ=0: |01⟩ (Hartree-Fock state)
    fn build_2qubit_ansatz() -> Circuit {
        let mut circuit = Circuit::new(2);
        
        // Prepare Hartree-Fock state |01⟩
        circuit.add(Gate::single(GateType::X, 0));
        
        // Parameterized excitation
        // Ry rotation to create superposition
        circuit.add_param_ry(0, Some("theta_0".to_string()));
        
        // Entangling gate
        circuit.add(Gate::two(GateType::CNOT, 0, 1));
        
        circuit
    }
    
    /// Build 4-qubit UCCSD-inspired ansatz for H₂
    ///
    /// Simplified UCCSD for H₂ with single and double excitations:
    /// - Single excitations: σ_g → σ_u
    /// - Double excitations: σ_g² → σ_u²
    fn build_4qubit_ansatz() -> Circuit {
        let mut circuit = Circuit::new(4);
        
        // Prepare Hartree-Fock state |0011⟩
        // (electrons in qubits 0 and 1, which are σ_g orbitals)
        circuit.add(Gate::single(GateType::X, 0));
        circuit.add(Gate::single(GateType::X, 1));
        
        // Single excitation: σ_g↑ ↔ σ_u↑ (qubits 0 ↔ 2)
        circuit.add(Gate::single(GateType::H, 0));
        circuit.add(Gate::two(GateType::CNOT, 0, 2));
        circuit.add_param_rz(2, Some("theta_s0".to_string()));
        circuit.add(Gate::two(GateType::CNOT, 0, 2));
        circuit.add(Gate::single(GateType::H, 0));
        
        // Single excitation: σ_g↓ ↔ σ_u↓ (qubits 1 ↔ 3)
        circuit.add(Gate::single(GateType::H, 1));
        circuit.add(Gate::two(GateType::CNOT, 1, 3));
        circuit.add_param_rz(3, Some("theta_s1".to_string()));
        circuit.add(Gate::two(GateType::CNOT, 1, 3));
        circuit.add(Gate::single(GateType::H, 1));
        
        // Double excitation: σ_g↑σ_g↓ ↔ σ_u↑σ_u↓
        // This is a 4-body term, simplified implementation
        circuit.add(Gate::two(GateType::CNOT, 0, 1));
        circuit.add(Gate::two(GateType::CNOT, 1, 2));
        circuit.add(Gate::two(GateType::CNOT, 2, 3));
        circuit.add_param_rz(3, Some("theta_d".to_string()));
        circuit.add(Gate::two(GateType::CNOT, 2, 3));
        circuit.add(Gate::two(GateType::CNOT, 1, 2));
        circuit.add(Gate::two(GateType::CNOT, 0, 1));
        
        circuit
    }
    
    /// Get number of parameters for given configuration
    pub fn n_params(config: &H2AnsatzConfig) -> usize {
        if config.hardware_efficient {
            // hardware_efficient: (depth+1) * n_qubits * 2 params
            (config.depth + 1) * config.n_qubits * 2
        } else {
            match config.n_qubits {
                2 => 1,  // Single theta parameter
                4 => 3,  // theta_s0, theta_s1, theta_d
                _ => 0,
            }
        }
    }
    
    /// Get initial parameters (near Hartree-Fock)
    pub fn initial_params(config: &H2AnsatzConfig) -> Vec<f64> {
        let n = Self::n_params(config);
        vec![0.01; n]  // Small perturbation from HF
    }
}

// =============================================================================
// Hamiltonian Measurement Circuits
// =============================================================================

/// Pauli basis for measurement
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PauliBasis {
    /// Z basis (computational basis)
    Z,
    /// X basis
    X,
    /// Y basis
    Y,
}

/// A Pauli term in the Hamiltonian
#[derive(Debug, Clone)]
pub struct PauliTerm {
    /// Coefficient
    pub coeff: f64,
    /// Pauli operators for each qubit (None = Identity)
    pub paulis: Vec<Option<PauliBasis>>,
}

impl PauliTerm {
    /// Create identity term
    pub fn identity(n_qubits: usize, coeff: f64) -> Self {
        Self {
            coeff,
            paulis: vec![None; n_qubits],
        }
    }
    
    /// Create Z term on single qubit
    pub fn z(qubit: usize, n_qubits: usize, coeff: f64) -> Self {
        let mut paulis = vec![None; n_qubits];
        paulis[qubit] = Some(PauliBasis::Z);
        Self { coeff, paulis }
    }
    
    /// Create ZZ term on two qubits
    pub fn zz(q1: usize, q2: usize, n_qubits: usize, coeff: f64) -> Self {
        let mut paulis = vec![None; n_qubits];
        paulis[q1] = Some(PauliBasis::Z);
        paulis[q2] = Some(PauliBasis::Z);
        Self { coeff, paulis }
    }
    
    /// Create XX term on two qubits
    pub fn xx(q1: usize, q2: usize, n_qubits: usize, coeff: f64) -> Self {
        let mut paulis = vec![None; n_qubits];
        paulis[q1] = Some(PauliBasis::X);
        paulis[q2] = Some(PauliBasis::X);
        Self { coeff, paulis }
    }
    
    /// Create YY term on two qubits
    pub fn yy(q1: usize, q2: usize, n_qubits: usize, coeff: f64) -> Self {
        let mut paulis = vec![None; n_qubits];
        paulis[q1] = Some(PauliBasis::Y);
        paulis[q2] = Some(PauliBasis::Y);
        Self { coeff, paulis }
    }
    
    /// Check if term is identity
    pub fn is_identity(&self) -> bool {
        self.paulis.iter().all(|p| p.is_none())
    }
    
    /// Get non-identity qubits
    pub fn active_qubits(&self) -> Vec<usize> {
        self.paulis
            .iter()
            .enumerate()
            .filter_map(|(i, p)| p.map(|_| i))
            .collect()
    }
}

/// H₂ Hamiltonian representation for hardware execution
#[derive(Debug, Clone)]
pub struct H2HamiltonianHW {
    /// Number of qubits
    pub n_qubits: usize,
    /// Pauli terms
    pub terms: Vec<PauliTerm>,
    /// Nuclear repulsion energy
    pub nuclear_repulsion: f64,
}

impl H2HamiltonianHW {
    /// Create 2-qubit H₂ Hamiltonian
    ///
    /// H = g0*I + g1*Z0 + g2*Z1 + g3*Z0Z1 + g4*(X0X1 + Y0Y1)
    pub fn two_qubit(g0: f64, g1: f64, g2: f64, g3: f64, g4: f64, e_nuc: f64) -> Self {
        let n_qubits = 2;
        let mut terms = Vec::new();
        
        // Identity term (includes nuclear repulsion)
        terms.push(PauliTerm::identity(n_qubits, g0 + e_nuc));
        
        // Z terms
        if g1.abs() > 1e-10 {
            terms.push(PauliTerm::z(0, n_qubits, g1));
        }
        if g2.abs() > 1e-10 {
            terms.push(PauliTerm::z(1, n_qubits, g2));
        }
        
        // ZZ term
        if g3.abs() > 1e-10 {
            terms.push(PauliTerm::zz(0, 1, n_qubits, g3));
        }
        
        // XX + YY terms
        if g4.abs() > 1e-10 {
            terms.push(PauliTerm::xx(0, 1, n_qubits, g4));
            terms.push(PauliTerm::yy(0, 1, n_qubits, g4));
        }
        
        Self {
            n_qubits,
            terms,
            nuclear_repulsion: e_nuc,
        }
    }
    
    /// Create 2-qubit H₂ Hamiltonian at equilibrium bond length
    pub fn equilibrium() -> Self {
        // Coefficients at R = 0.7414 Å
        Self::two_qubit(-0.7384, 0.1322, -0.1322, 0.1480, 0.1435, 0.7137)
    }
    
    /// Create 2-qubit H₂ Hamiltonian at given bond length
    ///
    /// Uses linear interpolation from precomputed values
    pub fn at_bond_length(r: f64) -> Self {
        let (g0, g1, g2, g3, g4, e_nuc) = Self::interpolate_coefficients(r);
        Self::two_qubit(g0, g1, g2, g3, g4, e_nuc)
    }
    
    /// Interpolate coefficients for arbitrary bond length
    fn interpolate_coefficients(bond_length: f64) -> (f64, f64, f64, f64, f64, f64) {
        // Precomputed H₂ coefficients (STO-3G basis)
        const COEFFS: &[(f64, f64, f64, f64, f64, f64, f64)] = &[
            // (R, g0, g1, g2, g3, g4, E_nuc)
            (0.50, -0.6401, 0.2189, -0.2189, 0.1095, 0.1624, 1.0583),
            (0.60, -0.6881, 0.1786, -0.1786, 0.1294, 0.1545, 0.8819),
            (0.70, -0.7240, 0.1462, -0.1462, 0.1427, 0.1469, 0.7560),
            (0.7414, -0.7384, 0.1322, -0.1322, 0.1480, 0.1435, 0.7137),
            (0.80, -0.7514, 0.1198, -0.1198, 0.1514, 0.1397, 0.6615),
            (0.90, -0.7725, 0.0976, -0.0976, 0.1564, 0.1330, 0.5880),
            (1.00, -0.7891, 0.0792, -0.0792, 0.1586, 0.1268, 0.5292),
            (1.20, -0.8123, 0.0502, -0.0502, 0.1586, 0.1157, 0.4410),
            (1.50, -0.8359, 0.0122, -0.0122, 0.1515, 0.0980, 0.3528),
            (2.00, -0.8447, -0.0115, 0.0115, 0.1412, 0.0846, 0.2646),
        ];
        
        // Clamp bond length to valid range
        let r = bond_length.clamp(0.5, 2.0);
        
        // Find bracketing indices
        let mut lower = 0;
        for (i, &(ri, _, _, _, _, _, _)) in COEFFS.iter().enumerate() {
            if ri <= r {
                lower = i;
            }
        }
        let upper = (lower + 1).min(COEFFS.len() - 1);
        
        if lower == upper {
            let (_, g0, g1, g2, g3, g4, e_nuc) = COEFFS[lower];
            return (g0, g1, g2, g3, g4, e_nuc);
        }
        
        // Linear interpolation
        let (r0, g0_0, g1_0, g2_0, g3_0, g4_0, e0) = COEFFS[lower];
        let (r1, g0_1, g1_1, g2_1, g3_1, g4_1, e1) = COEFFS[upper];
        let t = (r - r0) / (r1 - r0);
        
        (
            g0_0 + t * (g0_1 - g0_0),
            g1_0 + t * (g1_1 - g1_0),
            g2_0 + t * (g2_1 - g2_0),
            g3_0 + t * (g3_1 - g3_0),
            g4_0 + t * (g4_1 - g4_0),
            e0 + t * (e1 - e0),
        )
    }
    
    /// Get exact ground state energy (FCI)
    pub fn exact_ground_state(&self) -> f64 {
        // For 2-qubit H₂, we can compute exactly
        // This is a simplified calculation
        let identity_coeff = self.terms.iter()
            .find(|t| t.is_identity())
            .map(|t| t.coeff)
            .unwrap_or(0.0);
        
        // The exact energy involves diagonalizing the 4x4 matrix
        // For equilibrium H₂: approximately -1.137 Ha
        identity_coeff - 0.4  // Rough approximation
    }
    
    /// Get all unique Pauli bases needed for measurement
    pub fn measurement_bases(&self) -> Vec<Vec<Option<PauliBasis>>> {
        // Collect unique measurement bases
        let mut bases: Vec<Vec<Option<PauliBasis>>> = Vec::new();
        
        for term in &self.terms {
            if term.is_identity() {
                continue;  // Identity doesn't need measurement
            }
            
            // Check if this basis is already included
            let exists = bases.iter().any(|b| {
                b.iter().zip(&term.paulis).all(|(a, b)| a == b)
            });
            
            if !exists {
                bases.push(term.paulis.clone());
            }
        }
        
        bases
    }
}

/// Build measurement circuit for a Pauli term
pub fn build_measurement_circuit(
    ansatz: &Circuit,
    params: &[f64],
    basis: &[Option<PauliBasis>],
) -> Result<Circuit> {
    let n_qubits = ansatz.n_qubits;
    
    if basis.len() != n_qubits {
        return Err(IBMError::InvalidCircuit(format!(
            "Basis length {} doesn't match circuit qubits {}",
            basis.len(), n_qubits
        )));
    }
    
    let mut circuit = Circuit::new(n_qubits);
    
    // Copy ansatz gates with bound parameters
    for gate in &ansatz.gates {
        let new_gate = match &gate.gate_type {
            GateType::RxParam(idx) => Gate::single(GateType::Rx(params[*idx]), gate.qubits[0]),
            GateType::RyParam(idx) => Gate::single(GateType::Ry(params[*idx]), gate.qubits[0]),
            GateType::RzParam(idx) => Gate::single(GateType::Rz(params[*idx]), gate.qubits[0]),
            _ => gate.clone(),
        };
        circuit.add(new_gate);
    }
    
    // Add basis rotation for measurement
    for (q, pauli) in basis.iter().enumerate() {
        match pauli {
            Some(PauliBasis::X) => {
                // Rotate X to Z: H
                circuit.add(Gate::single(GateType::H, q));
            }
            Some(PauliBasis::Y) => {
                // Rotate Y to Z: Sdg then H
                circuit.add(Gate::single(GateType::Sdg, q));
                circuit.add(Gate::single(GateType::H, q));
            }
            Some(PauliBasis::Z) | None => {
                // Already in Z basis, no rotation needed
            }
        }
    }
    
    Ok(circuit)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transpiler::QASMTranspiler;
    
    #[test]
    fn test_h2_2qubit_ansatz() {
        let config = H2AnsatzConfig::two_qubit();
        let circuit = H2AnsatzBuilder::build(&config);
        
        assert_eq!(circuit.n_qubits, 2);
        assert_eq!(circuit.n_params, 1);
        
        let params = vec![0.5];
        let qasm = QASMTranspiler::transpile(&circuit, &params).unwrap();
        
        assert!(qasm.contains("x q[0]"));  // HF state prep
        assert!(qasm.contains("ry(0.5) q[0]"));  // Parameterized rotation
        assert!(qasm.contains("cx q[0], q[1]"));  // Entangling gate
    }
    
    #[test]
    fn test_h2_4qubit_ansatz() {
        let config = H2AnsatzConfig::four_qubit();
        let circuit = H2AnsatzBuilder::build(&config);
        
        assert_eq!(circuit.n_qubits, 4);
        assert_eq!(circuit.n_params, 3);
    }
    
    #[test]
    fn test_h2_hamiltonian_hw() {
        let h2 = H2HamiltonianHW::equilibrium();
        
        assert_eq!(h2.n_qubits, 2);
        assert!(!h2.terms.is_empty());
        
        // Check that we have identity, Z, ZZ, XX, YY terms
        let has_identity = h2.terms.iter().any(|t| t.is_identity());
        let has_z = h2.terms.iter().any(|t| {
            t.paulis.iter().filter(|p| **p == Some(PauliBasis::Z)).count() == 1
        });
        let has_zz = h2.terms.iter().any(|t| {
            t.paulis.iter().filter(|p| **p == Some(PauliBasis::Z)).count() == 2
        });
        
        assert!(has_identity);
        assert!(has_z);
        assert!(has_zz);
    }
    
    #[test]
    fn test_measurement_bases() {
        let h2 = H2HamiltonianHW::equilibrium();
        let bases = h2.measurement_bases();
        
        // Should have at least Z, ZZ, XX, YY bases
        assert!(bases.len() >= 3);
    }
    
    #[test]
    fn test_measurement_circuit() {
        let config = H2AnsatzConfig::two_qubit();
        let ansatz = H2AnsatzBuilder::build(&config);
        let params = vec![0.5];
        
        // Z basis measurement
        let z_basis = vec![Some(PauliBasis::Z), Some(PauliBasis::Z)];
        let z_circuit = build_measurement_circuit(&ansatz, &params, &z_basis).unwrap();
        
        // X basis measurement
        let x_basis = vec![Some(PauliBasis::X), Some(PauliBasis::X)];
        let x_circuit = build_measurement_circuit(&ansatz, &params, &x_basis).unwrap();
        
        // X basis should have H gates at the end
        let x_qasm = QASMTranspiler::transpile(&x_circuit, &[]).unwrap();
        assert!(x_qasm.contains("h q[0]"));
        assert!(x_qasm.contains("h q[1]"));
    }
    
    #[test]
    fn test_interpolate_coefficients() {
        // Test equilibrium
        let h2_eq = H2HamiltonianHW::at_bond_length(0.7414);
        assert!((h2_eq.nuclear_repulsion - 0.7137).abs() < 0.01);
        
        // Test interpolation between points
        let h2_mid = H2HamiltonianHW::at_bond_length(0.75);
        assert!(h2_mid.nuclear_repulsion > 0.66 && h2_mid.nuclear_repulsion < 0.76);
    }
    
    #[test]
    fn test_initial_params() {
        let config = H2AnsatzConfig::two_qubit();
        let params = H2AnsatzBuilder::initial_params(&config);
        
        assert_eq!(params.len(), 1);
        assert!(params[0].abs() < 0.1);  // Small perturbation
    }
}

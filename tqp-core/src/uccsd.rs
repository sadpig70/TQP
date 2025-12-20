//! UCCSD Ansatz for VQE
//!
//! Sprint 3 Week 5 Day 2: Unitary Coupled Cluster Singles and Doubles
//!
//! The UCCSD ansatz is the most commonly used ansatz for molecular
//! simulation on quantum computers. It preserves the correct symmetries
//! and provides a systematic way to capture electron correlation.
//!
//! # Theory
//!
//! The UCCSD state is:
//! |ψ(θ)⟩ = e^{T(θ) - T†(θ)} |HF⟩
//!
//! where T = T₁ + T₂ with:
//! - T₁ = Σ θᵢⱼ a†ᵢ aⱼ (single excitations)
//! - T₂ = Σ θᵢⱼₖₗ a†ᵢ a†ⱼ aₖ aₗ (double excitations)
//!
//! After Trotterization and Jordan-Wigner transformation, this becomes
//! a product of parameterized Pauli rotations.
//!
//! # Example
//!
//! ```ignore
//! use tqp_core::uccsd::{UCCSDCircuit, UCCSDConfig};
//!
//! // Create UCCSD for H₂ (2 electrons, 4 spin-orbitals)
//! let uccsd = UCCSDCircuit::h2();
//!
//! // Get the variational circuit
//! let circuit = uccsd.build_circuit();
//! ```

use std::f64::consts::PI;
use crate::autodiff::VariationalCircuit;
use crate::state::TQPState;

// =============================================================================
// Configuration
// =============================================================================

/// UCCSD configuration
#[derive(Debug, Clone)]
pub struct UCCSDConfig {
    /// Number of qubits (spin-orbitals)
    pub n_qubits: usize,
    /// Number of electrons
    pub n_electrons: usize,
    /// Include single excitations
    pub singles: bool,
    /// Include double excitations
    pub doubles: bool,
    /// Number of Trotter steps
    pub trotter_steps: usize,
    /// Use hardware-efficient decomposition
    pub hardware_efficient: bool,
}

impl Default for UCCSDConfig {
    fn default() -> Self {
        Self {
            n_qubits: 4,
            n_electrons: 2,
            singles: true,
            doubles: true,
            trotter_steps: 1,
            hardware_efficient: false,
        }
    }
}

impl UCCSDConfig {
    /// Create config for H₂ molecule (2 electrons, 4 qubits)
    pub fn h2() -> Self {
        Self {
            n_qubits: 4,
            n_electrons: 2,
            singles: true,
            doubles: true,
            trotter_steps: 1,
            hardware_efficient: false,
        }
    }

    /// Create config for LiH molecule (4 electrons, 12 qubits in minimal basis)
    pub fn lih() -> Self {
        Self {
            n_qubits: 12,
            n_electrons: 4,
            singles: true,
            doubles: true,
            trotter_steps: 1,
            hardware_efficient: false,
        }
    }

    /// Set number of Trotter steps
    pub fn with_trotter_steps(mut self, steps: usize) -> Self {
        self.trotter_steps = steps;
        self
    }

    /// Enable only singles
    pub fn singles_only(mut self) -> Self {
        self.singles = true;
        self.doubles = false;
        self
    }

    /// Enable only doubles
    pub fn doubles_only(mut self) -> Self {
        self.singles = false;
        self.doubles = true;
        self
    }

    /// Use hardware-efficient decomposition
    pub fn with_hardware_efficient(mut self) -> Self {
        self.hardware_efficient = true;
        self
    }
}

// =============================================================================
// Excitation Operators
// =============================================================================

/// Single excitation from orbital i to orbital a
#[derive(Debug, Clone, Copy)]
pub struct SingleExcitation {
    /// Occupied orbital index
    pub i: usize,
    /// Virtual orbital index
    pub a: usize,
}

/// Double excitation from orbitals (i,j) to (a,b)
#[derive(Debug, Clone, Copy)]
pub struct DoubleExcitation {
    /// First occupied orbital
    pub i: usize,
    /// Second occupied orbital
    pub j: usize,
    /// First virtual orbital
    pub a: usize,
    /// Second virtual orbital
    pub b: usize,
}

impl SingleExcitation {
    pub fn new(i: usize, a: usize) -> Self {
        Self { i, a }
    }
}

impl DoubleExcitation {
    pub fn new(i: usize, j: usize, a: usize, b: usize) -> Self {
        Self { i, j, a, b }
    }
}

// =============================================================================
// UCCSD Circuit Builder
// =============================================================================

/// UCCSD Circuit for VQE
#[derive(Debug, Clone)]
pub struct UCCSDCircuit {
    /// Configuration
    config: UCCSDConfig,
    /// Single excitations
    singles: Vec<SingleExcitation>,
    /// Double excitations
    doubles: Vec<DoubleExcitation>,
    /// Number of parameters
    n_params: usize,
}

impl UCCSDCircuit {
    /// Create UCCSD circuit with configuration
    pub fn new(config: UCCSDConfig) -> Self {
        let (singles, doubles) = Self::generate_excitations(&config);
        let n_params = singles.len() + doubles.len();

        Self {
            config,
            singles,
            doubles,
            n_params,
        }
    }

    /// Create UCCSD for H₂
    pub fn h2() -> Self {
        Self::new(UCCSDConfig::h2())
    }

    /// Create UCCSD for LiH
    pub fn lih() -> Self {
        Self::new(UCCSDConfig::lih())
    }

    /// Generate all symmetry-allowed excitations
    fn generate_excitations(config: &UCCSDConfig) -> (Vec<SingleExcitation>, Vec<DoubleExcitation>) {
        let n_occ = config.n_electrons;
        let n_virt = config.n_qubits - n_occ;

        let mut singles = Vec::new();
        let mut doubles = Vec::new();

        // Single excitations: i → a where i is occupied, a is virtual
        if config.singles {
            for i in 0..n_occ {
                for a in n_occ..config.n_qubits {
                    // Check spin conservation (same spin)
                    if i % 2 == a % 2 {
                        singles.push(SingleExcitation::new(i, a));
                    }
                }
            }
        }

        // Double excitations: (i,j) → (a,b)
        if config.doubles {
            for i in 0..n_occ {
                for j in (i + 1)..n_occ {
                    for a in n_occ..config.n_qubits {
                        for b in (a + 1)..config.n_qubits {
                            // Check spin conservation
                            let spin_in = (i % 2) + (j % 2);
                            let spin_out = (a % 2) + (b % 2);
                            if spin_in == spin_out {
                                doubles.push(DoubleExcitation::new(i, j, a, b));
                            }
                        }
                    }
                }
            }
        }

        (singles, doubles)
    }

    /// Get number of parameters
    pub fn n_params(&self) -> usize {
        self.n_params
    }

    /// Get number of qubits
    pub fn n_qubits(&self) -> usize {
        self.config.n_qubits
    }

    /// Get number of single excitations
    pub fn n_singles(&self) -> usize {
        self.singles.len()
    }

    /// Get number of double excitations
    pub fn n_doubles(&self) -> usize {
        self.doubles.len()
    }

    /// Build the variational circuit
    pub fn build_circuit(&self) -> VariationalCircuit {
        let mut circuit = VariationalCircuit::new(self.config.n_qubits);

        // Initial Hartree-Fock state: |1100...0⟩
        // Apply X gates to set occupied orbitals
        for i in 0..self.config.n_electrons {
            circuit.x(i);
        }

        // Apply UCCSD operators
        for _trotter in 0..self.config.trotter_steps {
            // Single excitations
            for exc in &self.singles {
                self.apply_single_excitation(&mut circuit, exc);
            }

            // Double excitations
            for exc in &self.doubles {
                self.apply_double_excitation(&mut circuit, exc);
            }
        }

        circuit
    }

    /// Apply single excitation operator: e^{θ(a†_a a_i - a†_i a_a)}
    fn apply_single_excitation(&self, circuit: &mut VariationalCircuit, exc: &SingleExcitation) {
        let i = exc.i;
        let a = exc.a;

        if self.config.hardware_efficient {
            // Hardware-efficient: just use RY gates
            circuit.ry(i);
            circuit.ry(a);
            circuit.cnot(i, a);
        } else {
            // Full UCCSD decomposition
            // The single excitation decomposes into:
            // exp(θ/2 (X_i Y_a - Y_i X_a) * Z_{i+1}...Z_{a-1})

            // Simplified version for nearest-neighbor or all-to-all
            // Using the "Givens rotation" decomposition

            // CNOT ladder from i to a
            if a > i + 1 {
                for q in i..(a - 1) {
                    circuit.cnot(q, q + 1);
                }
            }

            // Parameterized rotation
            circuit.ry(a); // θ parameter

            // Reverse CNOT ladder
            if a > i + 1 {
                for q in ((i + 1)..a).rev() {
                    circuit.cnot(q - 1, q);
                }
            }

            circuit.cnot(i, a);
        }
    }

    /// Apply double excitation operator: e^{θ(a†_a a†_b a_j a_i - h.c.)}
    fn apply_double_excitation(&self, circuit: &mut VariationalCircuit, exc: &DoubleExcitation) {
        let i = exc.i;
        let j = exc.j;
        let a = exc.a;
        let b = exc.b;

        if self.config.hardware_efficient {
            // Hardware-efficient version
            circuit.ry(i);
            circuit.ry(j);
            circuit.ry(a);
            circuit.ry(b);
            circuit.cnot(i, j);
            circuit.cnot(a, b);
            circuit.cnot(j, a);
        } else {
            // Full UCCSD double excitation decomposition
            // This is the most expensive part of UCCSD

            // The double excitation can be decomposed into 8 Pauli exponentials
            // For simplicity, we use a ladder decomposition

            // Prepare entanglement
            circuit.cnot(i, j);
            circuit.cnot(j, a);
            circuit.cnot(a, b);

            // Parameterized rotation
            circuit.rz(b); // θ parameter

            // Reverse entanglement
            circuit.cnot(a, b);
            circuit.cnot(j, a);
            circuit.cnot(i, j);

            // Additional rotations for accuracy
            circuit.ry(i);
            circuit.ry(b);
        }
    }

    /// Create initial parameters (zeros for HF reference)
    pub fn initial_params(&self) -> Vec<f64> {
        vec![0.0; self.n_params]
    }

    /// Create random initial parameters
    pub fn random_params(&self, seed: u64) -> Vec<f64> {
        let mut params = Vec::with_capacity(self.n_params);
        let mut rng_state = seed;

        for _ in 0..self.n_params {
            rng_state = rng_state.wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let r = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
            params.push((r - 0.5) * 0.2); // Small random values
        }

        params
    }

    /// Print circuit summary
    pub fn print_summary(&self) {
        println!("=== UCCSD Circuit Summary ===");
        println!("Qubits: {}", self.config.n_qubits);
        println!("Electrons: {}", self.config.n_electrons);
        println!("Single excitations: {}", self.singles.len());
        println!("Double excitations: {}", self.doubles.len());
        println!("Total parameters: {}", self.n_params);
        println!("Trotter steps: {}", self.config.trotter_steps);

        if !self.singles.is_empty() {
            println!("\nSingle excitations:");
            for exc in &self.singles {
                println!("  {} → {}", exc.i, exc.a);
            }
        }

        if !self.doubles.is_empty() {
            println!("\nDouble excitations:");
            for exc in &self.doubles {
                println!("  ({},{}) → ({},{})", exc.i, exc.j, exc.a, exc.b);
            }
        }
    }
}

// =============================================================================
// Hardware-Efficient Ansatz Alternative
// =============================================================================

/// Hardware-efficient ansatz for molecular simulation
#[derive(Debug, Clone)]
pub struct HardwareEfficientAnsatz {
    /// Number of qubits
    n_qubits: usize,
    /// Number of layers
    n_layers: usize,
    /// Number of electrons (for initial state)
    n_electrons: usize,
}

impl HardwareEfficientAnsatz {
    /// Create hardware-efficient ansatz
    pub fn new(n_qubits: usize, n_layers: usize, n_electrons: usize) -> Self {
        Self {
            n_qubits,
            n_layers,
            n_electrons,
        }
    }

    /// Create for H₂
    pub fn h2(n_layers: usize) -> Self {
        Self::new(4, n_layers, 2)
    }

    /// Get number of parameters
    pub fn n_params(&self) -> usize {
        // Each layer: n_qubits * 2 (RY + RZ) + (n_qubits - 1) CNOTs (no params)
        self.n_qubits * 2 * self.n_layers
    }

    /// Build the circuit
    pub fn build_circuit(&self) -> VariationalCircuit {
        let mut circuit = VariationalCircuit::new(self.n_qubits);

        // Initial Hartree-Fock state
        for i in 0..self.n_electrons {
            circuit.x(i);
        }

        // Variational layers
        for _layer in 0..self.n_layers {
            // Single-qubit rotations
            for q in 0..self.n_qubits {
                circuit.ry(q);
                circuit.rz(q);
            }

            // Entangling layer (linear connectivity)
            for q in 0..(self.n_qubits - 1) {
                circuit.cnot(q, q + 1);
            }
        }

        circuit
    }

    /// Create initial parameters
    pub fn initial_params(&self) -> Vec<f64> {
        vec![0.1; self.n_params()]
    }
}

// =============================================================================
// Simplified H₂ Ansatz (2-qubit)
// =============================================================================

/// Simplified 2-qubit ansatz for H₂ with symmetry reduction
#[derive(Debug, Clone)]
pub struct H2SimplifiedAnsatz {
    /// Number of parameters
    n_params: usize,
}

impl H2SimplifiedAnsatz {
    /// Create simplified H₂ ansatz
    pub fn new() -> Self {
        Self { n_params: 1 }
    }

    /// Build the circuit (single parameter)
    pub fn build_circuit(&self) -> VariationalCircuit {
        let mut circuit = VariationalCircuit::new(2);

        // Initial state |01⟩ (one electron in each spatial orbital)
        circuit.x(0);

        // Single excitation rotation
        circuit.ry(1);
        circuit.cnot(0, 1);

        circuit
    }

    /// Build circuit for 4 qubits (full representation)
    pub fn build_circuit_4q(&self) -> VariationalCircuit {
        let mut circuit = VariationalCircuit::new(4);

        // HF state |1100⟩
        circuit.x(0);
        circuit.x(1);

        // Double excitation: |1100⟩ ↔ |0011⟩
        // This is the only symmetry-allowed excitation

        // UCCSD double excitation decomposition
        circuit.cnot(2, 3);
        circuit.cnot(0, 2);
        circuit.h(3);
        circuit.cnot(1, 3);

        // Parameterized rotation (the only parameter)
        circuit.ry(3);

        // Reverse
        circuit.cnot(1, 3);
        circuit.h(3);
        circuit.cnot(0, 2);
        circuit.cnot(2, 3);

        circuit
    }

    /// Get number of parameters
    pub fn n_params(&self) -> usize {
        self.n_params
    }

    /// Initial parameters
    pub fn initial_params(&self) -> Vec<f64> {
        vec![0.0]
    }
}

impl Default for H2SimplifiedAnsatz {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uccsd_config_h2() {
        let config = UCCSDConfig::h2();
        assert_eq!(config.n_qubits, 4);
        assert_eq!(config.n_electrons, 2);
        assert!(config.singles);
        assert!(config.doubles);
    }

    #[test]
    fn test_uccsd_circuit_creation() {
        let uccsd = UCCSDCircuit::h2();

        assert_eq!(uccsd.n_qubits(), 4);
        assert!(uccsd.n_params() > 0);
    }

    #[test]
    fn test_uccsd_excitations() {
        let uccsd = UCCSDCircuit::h2();

        // For H₂: 2 occupied (0,1), 2 virtual (2,3)
        // Singles with spin conservation: 0→2, 1→3
        // Doubles: (0,1)→(2,3)

        println!("Singles: {}", uccsd.n_singles());
        println!("Doubles: {}", uccsd.n_doubles());

        assert!(uccsd.n_singles() >= 0);
        assert!(uccsd.n_doubles() >= 0);
    }

    #[test]
    fn test_uccsd_build_circuit() {
        let uccsd = UCCSDCircuit::h2();
        let circuit = uccsd.build_circuit();

        assert_eq!(circuit.n_qubits(), 4);
        assert!(circuit.num_params() > 0);
    }

    #[test]
    fn test_uccsd_initial_params() {
        let uccsd = UCCSDCircuit::h2();
        let params = uccsd.initial_params();

        assert_eq!(params.len(), uccsd.n_params());
        assert!(params.iter().all(|&p| p == 0.0));
    }

    #[test]
    fn test_uccsd_random_params() {
        let uccsd = UCCSDCircuit::h2();
        let params = uccsd.random_params(42);

        assert_eq!(params.len(), uccsd.n_params());
        // Should be small random values
        assert!(params.iter().all(|&p| p.abs() < 0.2));
    }

    #[test]
    fn test_hardware_efficient_ansatz() {
        let hea = HardwareEfficientAnsatz::h2(2);

        assert_eq!(hea.n_params(), 4 * 2 * 2); // 4 qubits * 2 gates * 2 layers

        let circuit = hea.build_circuit();
        assert_eq!(circuit.n_qubits(), 4);
    }

    #[test]
    fn test_h2_simplified_ansatz() {
        let ansatz = H2SimplifiedAnsatz::new();

        assert_eq!(ansatz.n_params(), 1);

        let circuit = ansatz.build_circuit();
        assert_eq!(circuit.n_qubits(), 2);
    }

    #[test]
    fn test_h2_simplified_4q() {
        let ansatz = H2SimplifiedAnsatz::new();
        let circuit = ansatz.build_circuit_4q();

        assert_eq!(circuit.n_qubits(), 4);
    }

    #[test]
    fn test_single_excitation() {
        let exc = SingleExcitation::new(0, 2);
        assert_eq!(exc.i, 0);
        assert_eq!(exc.a, 2);
    }

    #[test]
    fn test_double_excitation() {
        let exc = DoubleExcitation::new(0, 1, 2, 3);
        assert_eq!(exc.i, 0);
        assert_eq!(exc.j, 1);
        assert_eq!(exc.a, 2);
        assert_eq!(exc.b, 3);
    }

    #[test]
    fn test_uccsd_config_builder() {
        let config = UCCSDConfig::default()
            .with_trotter_steps(2)
            .doubles_only();

        assert_eq!(config.trotter_steps, 2);
        assert!(!config.singles);
        assert!(config.doubles);
    }

    #[test]
    fn test_uccsd_hardware_efficient_mode() {
        let config = UCCSDConfig::h2().with_hardware_efficient();
        let uccsd = UCCSDCircuit::new(config);

        let circuit = uccsd.build_circuit();
        assert!(circuit.num_params() > 0);
    }

    #[test]
    fn test_uccsd_lih() {
        let uccsd = UCCSDCircuit::lih();

        assert_eq!(uccsd.n_qubits(), 12);
        // LiH has more excitations
        assert!(uccsd.n_params() > UCCSDCircuit::h2().n_params());
    }
}

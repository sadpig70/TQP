//! QAOA Ansatz Implementation
//!
//! Sprint 3 Week 6 Day 2: Quantum Approximate Optimization Algorithm
//!
//! QAOA is a variational quantum algorithm for solving combinatorial
//! optimization problems. It alternates between:
//! - Cost layer: e^{-iγC} where C is the problem Hamiltonian
//! - Mixer layer: e^{-iβB} where B = Σ X_i is the transverse field
//!
//! # Theory
//!
//! The QAOA state is:
//! |ψ(γ,β)⟩ = U_B(β_p) U_C(γ_p) ... U_B(β_1) U_C(γ_1) |+⟩^n
//!
//! where:
//! - U_C(γ) = e^{-iγC} = Π_{ij} e^{-iγ w_{ij} Z_i Z_j}
//! - U_B(β) = e^{-iβB} = Π_i e^{-iβ X_i}
//!
//! # Example
//!
//! ```ignore
//! use tqp_core::qaoa::{QAOAAnsatz, QAOAConfig};
//! use tqp_core::maxcut::instances;
//!
//! let problem = instances::triangle();
//! let qaoa = QAOAAnsatz::new(problem, 2); // p=2 layers
//!
//! let circuit = qaoa.build_circuit();
//! ```

use std::f64::consts::PI;
use crate::autodiff::{VariationalCircuit, Hamiltonian, PauliObservable};
use crate::maxcut::{MaxCutProblem, Graph, Edge};

// =============================================================================
// Constants
// =============================================================================

/// Default number of QAOA layers
pub const DEFAULT_QAOA_LAYERS: usize = 1;

/// Initial gamma range for parameter initialization
pub const GAMMA_INIT_RANGE: f64 = PI / 4.0;

/// Initial beta range for parameter initialization
pub const BETA_INIT_RANGE: f64 = PI / 4.0;

// =============================================================================
// QAOA Configuration
// =============================================================================

/// QAOA configuration
#[derive(Debug, Clone)]
pub struct QAOAConfig {
    /// Number of QAOA layers (p)
    pub p: usize,
    /// Use RZZ decomposition for cost layer
    pub decompose_rzz: bool,
    /// Initial state type
    pub init_state: InitState,
    /// Mixer type
    pub mixer: MixerType,
}

/// Initial state for QAOA
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitState {
    /// Uniform superposition |+⟩^n (standard QAOA)
    Plus,
    /// Computational basis |0⟩^n
    Zero,
    /// Custom (requires manual preparation)
    Custom,
}

/// Mixer Hamiltonian type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixerType {
    /// Standard X mixer: B = Σ X_i
    X,
    /// XY mixer for constrained problems
    XY,
    /// Grover-style mixer
    Grover,
}

impl Default for QAOAConfig {
    fn default() -> Self {
        Self {
            p: DEFAULT_QAOA_LAYERS,
            decompose_rzz: true,
            init_state: InitState::Plus,
            mixer: MixerType::X,
        }
    }
}

impl QAOAConfig {
    /// Set number of layers
    pub fn with_layers(mut self, p: usize) -> Self {
        self.p = p;
        self
    }

    /// Use native RZZ gates (if available)
    pub fn with_native_rzz(mut self) -> Self {
        self.decompose_rzz = false;
        self
    }

    /// Set initial state
    pub fn with_init_state(mut self, state: InitState) -> Self {
        self.init_state = state;
        self
    }

    /// Set mixer type
    pub fn with_mixer(mut self, mixer: MixerType) -> Self {
        self.mixer = mixer;
        self
    }
}

// =============================================================================
// QAOA Ansatz
// =============================================================================

/// QAOA Ansatz for MaxCut
#[derive(Debug, Clone)]
pub struct QAOAAnsatz {
    /// Number of qubits
    n_qubits: usize,
    /// Number of layers (p)
    p: usize,
    /// Configuration
    config: QAOAConfig,
    /// Edges from the problem graph
    edges: Vec<Edge>,
    /// Total number of parameters (2p: gamma and beta for each layer)
    n_params: usize,
}

impl QAOAAnsatz {
    /// Create QAOA ansatz for MaxCut problem
    pub fn new(problem: &MaxCutProblem, p: usize) -> Self {
        Self::with_config(problem, QAOAConfig::default().with_layers(p))
    }

    /// Create QAOA ansatz with configuration
    pub fn with_config(problem: &MaxCutProblem, config: QAOAConfig) -> Self {
        let n_qubits = problem.n_qubits();
        let edges = problem.graph().edges().to_vec();
        
        // Each layer has: n_edges RZ gates (cost) + n_qubits RX gates (mixer)
        let params_per_layer = edges.len() + n_qubits;
        let n_params = config.p * params_per_layer;

        Self {
            n_qubits,
            p: config.p,
            config,
            edges,
            n_params,
        }
    }

    /// Create QAOA directly from graph
    pub fn from_graph(graph: &Graph, p: usize) -> Self {
        let n_qubits = graph.n_vertices();
        let edges = graph.edges().to_vec();
        let config = QAOAConfig::default().with_layers(p);
        
        let params_per_layer = edges.len() + n_qubits;
        let n_params = p * params_per_layer;

        Self {
            n_qubits,
            p,
            config,
            edges,
            n_params,
        }
    }

    /// Get number of qubits
    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// Get number of layers
    pub fn n_layers(&self) -> usize {
        self.p
    }

    /// Get total number of parameters
    pub fn n_params(&self) -> usize {
        self.n_params
    }

    /// Build the QAOA circuit
    ///
    /// Parameters are ordered as: [γ_1, β_1, γ_2, β_2, ..., γ_p, β_p]
    pub fn build_circuit(&self) -> VariationalCircuit {
        let mut circuit = VariationalCircuit::new(self.n_qubits);

        // Initial state: |+⟩^n
        match self.config.init_state {
            InitState::Plus => {
                for q in 0..self.n_qubits {
                    circuit.h(q);
                }
            }
            InitState::Zero => {
                // Already |0⟩^n, do nothing
            }
            InitState::Custom => {
                // User will prepare state externally
            }
        }

        // Apply p layers of Cost + Mixer
        for _layer in 0..self.p {
            // Cost layer: U_C(γ) = exp(-iγC) where C = Σ w_{ij} Z_i Z_j
            self.apply_cost_layer(&mut circuit);

            // Mixer layer: U_B(β) = exp(-iβB) where B = Σ X_i
            self.apply_mixer_layer(&mut circuit);
        }

        circuit
    }

    /// Apply cost unitary for one layer
    /// U_C(γ) = Π_{(i,j)∈E} exp(-iγ w_{ij} Z_i Z_j)
    fn apply_cost_layer(&self, circuit: &mut VariationalCircuit) {
        for edge in &self.edges {
            if self.config.decompose_rzz {
                // Decompose RZZ(θ) = CNOT · RZ(θ) · CNOT
                // exp(-iγ Z_i Z_j) = exp(-iθ/2 Z_i Z_j) with θ = 2γ·w
                circuit.cnot(edge.u, edge.v);
                circuit.rz(edge.v); // This is the γ parameter (shared per layer)
                circuit.cnot(edge.u, edge.v);
            } else {
                // Native RZZ would go here
                // For now, use decomposition
                circuit.cnot(edge.u, edge.v);
                circuit.rz(edge.v);
                circuit.cnot(edge.u, edge.v);
            }
        }
    }

    /// Apply mixer unitary for one layer
    /// U_B(β) = Π_i exp(-iβ X_i) = Π_i RX(2β)
    fn apply_mixer_layer(&self, circuit: &mut VariationalCircuit) {
        match self.config.mixer {
            MixerType::X => {
                for q in 0..self.n_qubits {
                    circuit.rx(q); // This is the β parameter (shared per layer)
                }
            }
            MixerType::XY => {
                // XY mixer: exp(-iβ(X_i X_j + Y_i Y_j))
                // Simplified: just use RX for now
                for q in 0..self.n_qubits {
                    circuit.rx(q);
                }
            }
            MixerType::Grover => {
                // Grover mixer: 2|+⟩⟨+| - I
                // Approximated with X mixer
                for q in 0..self.n_qubits {
                    circuit.rx(q);
                }
            }
        }
    }

    /// Generate initial parameters
    ///
    /// Strategy: small random values centered around known good regions
    pub fn initial_params(&self, seed: u64) -> Vec<f64> {
        let mut params = Vec::with_capacity(self.n_params);
        let mut rng_state = seed;

        for _ in 0..self.n_params {
            rng_state = rng_state.wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let r = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
            // Small initial values for better convergence
            params.push((r - 0.5) * 0.2);
        }

        params
    }

    /// Generate grid of parameters for optimization landscape analysis
    /// Note: Only practical for very small problems
    pub fn param_grid(&self, points_per_param: usize) -> Vec<Vec<f64>> {
        if self.n_params > 4 {
            // Too many parameters for grid search
            return vec![self.initial_params(42)];
        }

        let mut grid = Vec::new();
        let total = points_per_param.pow(self.n_params as u32);

        for i in 0..total.min(10000) {
            let mut params = Vec::with_capacity(self.n_params);
            let mut idx = i;
            for _ in 0..self.n_params {
                let val = (idx % points_per_param) as f64 / points_per_param as f64 * std::f64::consts::PI;
                params.push(val);
                idx /= points_per_param;
            }
            grid.push(params);
        }

        grid
    }

    /// Get recommended parameter bounds
    pub fn param_bounds(&self) -> Vec<(f64, f64)> {
        vec![(0.0, std::f64::consts::PI); self.n_params]
    }

    /// Print ansatz summary
    pub fn print_summary(&self) {
        println!("=== QAOA Ansatz ===");
        println!("Qubits: {}", self.n_qubits);
        println!("Layers (p): {}", self.p);
        println!("Parameters: {}", self.n_params);
        println!("Edges: {}", self.edges.len());
        println!("Init state: {:?}", self.config.init_state);
        println!("Mixer: {:?}", self.config.mixer);
    }
}

// =============================================================================
// QAOA for General Hamiltonians
// =============================================================================

/// General QAOA for arbitrary cost Hamiltonians
#[derive(Debug, Clone)]
pub struct GeneralQAOA {
    /// Number of qubits
    n_qubits: usize,
    /// Number of layers
    p: usize,
    /// Cost Hamiltonian terms (ZZ interactions)
    zz_terms: Vec<(usize, usize, f64)>,
    /// Single Z terms
    z_terms: Vec<(usize, f64)>,
}

impl GeneralQAOA {
    /// Create from Hamiltonian
    pub fn from_hamiltonian(n_qubits: usize, hamiltonian: &Hamiltonian, p: usize) -> Self {
        let mut zz_terms = Vec::new();
        let mut z_terms = Vec::new();

        for term in hamiltonian.terms() {
            let pauli_terms = &term.terms;

            // Count Z operators
            let z_positions: Vec<usize> = pauli_terms.iter()
                .filter(|(_, p)| *p == 'Z')
                .map(|(i, _)| *i)
                .collect();

            match z_positions.len() {
                1 => {
                    z_terms.push((z_positions[0], term.coeff));
                }
                2 => {
                    zz_terms.push((z_positions[0], z_positions[1], term.coeff));
                }
                _ => {
                    // Higher order terms not supported
                }
            }
        }

        Self {
            n_qubits,
            p,
            zz_terms,
            z_terms,
        }
    }

    /// Build circuit
    pub fn build_circuit(&self) -> VariationalCircuit {
        let mut circuit = VariationalCircuit::new(self.n_qubits);

        // Initial state |+⟩^n
        for q in 0..self.n_qubits {
            circuit.h(q);
        }

        for _layer in 0..self.p {
            // ZZ terms
            for (i, j, _weight) in &self.zz_terms {
                circuit.cnot(*i, *j);
                circuit.rz(*j);
                circuit.cnot(*i, *j);
            }

            // Z terms
            for (i, _weight) in &self.z_terms {
                circuit.rz(*i);
            }

            // Mixer
            for q in 0..self.n_qubits {
                circuit.rx(q);
            }
        }

        circuit
    }

    /// Get number of parameters
    pub fn n_params(&self) -> usize {
        self.p * (self.zz_terms.len() + self.z_terms.len() + self.n_qubits)
    }
}

// =============================================================================
// Warm-Start QAOA
// =============================================================================

/// Warm-start QAOA with classical solution
pub struct WarmStartQAOA {
    /// Base QAOA ansatz
    base: QAOAAnsatz,
    /// Classical solution for warm start
    classical_solution: Vec<bool>,
}

impl WarmStartQAOA {
    /// Create warm-start QAOA
    pub fn new(problem: &MaxCutProblem, p: usize, classical_solution: Vec<bool>) -> Self {
        let base = QAOAAnsatz::new(problem, p);
        Self {
            base,
            classical_solution,
        }
    }

    /// Build circuit with warm start
    pub fn build_circuit(&self) -> VariationalCircuit {
        let mut circuit = VariationalCircuit::new(self.base.n_qubits);

        // Prepare classical solution state with rotation
        for (q, &bit) in self.classical_solution.iter().enumerate() {
            // Rotate towards classical solution
            // |0⟩ → cos(ε)|0⟩ + sin(ε)|1⟩ if bit=1
            // This is a simplified warm start
            circuit.h(q);
            if bit {
                circuit.rz(q); // Small rotation towards |1⟩
            }
        }

        // Continue with standard QAOA layers
        for _layer in 0..self.base.p {
            self.base.apply_cost_layer(&mut circuit);
            self.base.apply_mixer_layer(&mut circuit);
        }

        circuit
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::maxcut::instances;

    #[test]
    fn test_qaoa_config_default() {
        let config = QAOAConfig::default();
        assert_eq!(config.p, 1);
        assert!(config.decompose_rzz);
        assert_eq!(config.init_state, InitState::Plus);
        assert_eq!(config.mixer, MixerType::X);
    }

    #[test]
    fn test_qaoa_config_builder() {
        let config = QAOAConfig::default()
            .with_layers(3)
            .with_native_rzz()
            .with_mixer(MixerType::XY);

        assert_eq!(config.p, 3);
        assert!(!config.decompose_rzz);
        assert_eq!(config.mixer, MixerType::XY);
    }

    #[test]
    fn test_qaoa_ansatz_creation() {
        let problem = instances::triangle();
        let qaoa = QAOAAnsatz::new(&problem, 2);

        assert_eq!(qaoa.n_qubits(), 3);
        assert_eq!(qaoa.n_layers(), 2);
        // Triangle: 3 edges + 3 qubits = 6 params per layer, p=2 -> 12
        assert_eq!(qaoa.n_params(), 12);
    }

    #[test]
    fn test_qaoa_build_circuit() {
        let problem = instances::triangle();
        let qaoa = QAOAAnsatz::new(&problem, 1);

        let circuit = qaoa.build_circuit();
        assert_eq!(circuit.n_qubits(), 3);
        assert!(circuit.num_params() > 0);
    }

    #[test]
    fn test_qaoa_initial_params() {
        let problem = instances::square();
        let qaoa = QAOAAnsatz::new(&problem, 2);

        let params = qaoa.initial_params(42);
        // Square: 4 edges + 4 qubits = 8 params per layer, p=2 → 16
        assert_eq!(params.len(), qaoa.n_params());

        // Should be small values
        for &p in &params {
            assert!(p.abs() < PI);
        }
    }

    #[test]
    fn test_qaoa_param_grid() {
        let problem = instances::triangle();
        let qaoa = QAOAAnsatz::new(&problem, 1);

        let grid = qaoa.param_grid(5);
        // Should have some entries (limited by n_params)
        assert!(!grid.is_empty());
    }

    #[test]
    fn test_qaoa_param_bounds() {
        let problem = instances::triangle();
        let qaoa = QAOAAnsatz::new(&problem, 1);

        let bounds = qaoa.param_bounds();
        // Triangle: 3 edges + 3 qubits = 6 params per layer
        assert_eq!(bounds.len(), qaoa.n_params());

        for (low, high) in bounds {
            assert_eq!(low, 0.0);
            assert!((high - PI).abs() < 0.001);
        }
    }

    #[test]
    fn test_qaoa_from_graph() {
        let graph = Graph::complete(4);
        let qaoa = QAOAAnsatz::from_graph(&graph, 2);

        assert_eq!(qaoa.n_qubits(), 4);
        assert_eq!(qaoa.n_layers(), 2);
    }

    #[test]
    fn test_qaoa_different_layers() {
        let problem = instances::triangle();
        let params_per_layer = 6; // 3 edges + 3 qubits

        for p in 1..=5 {
            let qaoa = QAOAAnsatz::new(&problem, p);
            assert_eq!(qaoa.n_params(), params_per_layer * p);

            let circuit = qaoa.build_circuit();
            assert!(circuit.num_params() > 0);
        }
    }

    #[test]
    fn test_qaoa_with_zero_init() {
        let problem = instances::triangle();
        let config = QAOAConfig::default()
            .with_init_state(InitState::Zero);

        let qaoa = QAOAAnsatz::with_config(&problem, config);
        let circuit = qaoa.build_circuit();

        assert_eq!(circuit.n_qubits(), 3);
    }

    #[test]
    fn test_general_qaoa() {
        let problem = instances::triangle();
        let ham = problem.cost_hamiltonian();

        let qaoa = GeneralQAOA::from_hamiltonian(3, ham, 1);
        let circuit = qaoa.build_circuit();

        assert_eq!(circuit.n_qubits(), 3);
    }

    #[test]
    fn test_warm_start_qaoa() {
        let problem = instances::triangle();
        let classical = vec![true, false, false]; // A valid cut

        let qaoa = WarmStartQAOA::new(&problem, 1, classical);
        let circuit = qaoa.build_circuit();

        assert_eq!(circuit.n_qubits(), 3);
    }

    #[test]
    fn test_qaoa_square() {
        let problem = instances::square();
        let qaoa = QAOAAnsatz::new(&problem, 2);

        assert_eq!(qaoa.n_qubits(), 4);
        assert_eq!(qaoa.edges.len(), 4);

        let circuit = qaoa.build_circuit();
        assert_eq!(circuit.n_qubits(), 4);
    }

    #[test]
    fn test_qaoa_k5() {
        let problem = instances::k5();
        let qaoa = QAOAAnsatz::new(&problem, 1);

        assert_eq!(qaoa.n_qubits(), 5);
        assert_eq!(qaoa.edges.len(), 10); // C(5,2) = 10

        let circuit = qaoa.build_circuit();
        assert_eq!(circuit.n_qubits(), 5);
    }
}

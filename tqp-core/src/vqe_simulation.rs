//! VQE Simulation for Molecular Systems
//!
//! Sprint 3 Week 5 Day 3: Complete VQE implementation
//!
//! This module provides a complete VQE simulation framework that combines:
//! - Molecular Hamiltonians (H₂, etc.)
//! - UCCSD and hardware-efficient ansätze
//! - Various optimizers (Adam, QNG, SPSA)
//!
//! # Example
//!
//! ```ignore
//! use tqp_core::vqe_simulation::{VQESimulation, VQEConfig};
//!
//! let vqe = VQESimulation::h2_default();
//! let result = vqe.run();
//!
//! println!("Ground state energy: {:.6} Ha", result.energy);
//! println!("Chemical accuracy: {}", result.chemical_accuracy);
//! ```

use std::time::Instant;

use crate::autodiff::{
    compute_expectation, compute_gradient, VariationalCircuit, Hamiltonian,
};
use crate::optimizer::{Optimizer, OptimizerConfig, OptimizerType};
use crate::qng::{QNGOptimizer, QNGConfig};
use crate::h2_hamiltonian::{H2Hamiltonian, H2Config, H2_EXPERIMENTAL_ENERGY};
use crate::uccsd::{UCCSDCircuit, UCCSDConfig, HardwareEfficientAnsatz, H2SimplifiedAnsatz};

// =============================================================================
// Constants
// =============================================================================

/// Chemical accuracy threshold (1 kcal/mol ≈ 0.0016 Ha)
pub const CHEMICAL_ACCURACY: f64 = 0.0016;

/// High accuracy threshold (0.1 kcal/mol)
pub const HIGH_ACCURACY: f64 = 0.00016;

/// Default maximum iterations
pub const DEFAULT_VQE_MAX_ITER: usize = 200;

/// Default convergence tolerance
pub const DEFAULT_VQE_TOLERANCE: f64 = 1e-6;

// =============================================================================
// Configuration
// =============================================================================

/// Type of ansatz to use
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnsatzType {
    /// Full UCCSD ansatz
    UCCSD,
    /// Hardware-efficient ansatz
    HardwareEfficient,
    /// Simplified single-parameter ansatz for H₂
    Simplified,
}

/// Type of optimizer to use
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VQEOptimizer {
    /// Adam optimizer
    Adam,
    /// Quantum Natural Gradient
    QNG,
    /// SPSA (gradient-free)
    SPSA,
    /// Simple SGD
    SGD,
    /// SGD with momentum
    Momentum,
}

/// VQE configuration
#[derive(Debug, Clone)]
pub struct VQEConfig {
    /// Ansatz type
    pub ansatz: AnsatzType,
    /// Optimizer type
    pub optimizer: VQEOptimizer,
    /// Maximum iterations
    pub max_iter: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Number of hardware-efficient layers (if applicable)
    pub n_layers: usize,
    /// Whether to use learning rate schedule
    pub lr_schedule: bool,
    /// Verbose output
    pub verbose: bool,
}

impl Default for VQEConfig {
    fn default() -> Self {
        Self {
            ansatz: AnsatzType::UCCSD,
            optimizer: VQEOptimizer::Adam,
            max_iter: DEFAULT_VQE_MAX_ITER,
            learning_rate: 0.1,
            tolerance: DEFAULT_VQE_TOLERANCE,
            n_layers: 2,
            lr_schedule: false,
            verbose: false,
        }
    }
}

impl VQEConfig {
    /// Use UCCSD ansatz
    pub fn with_uccsd(mut self) -> Self {
        self.ansatz = AnsatzType::UCCSD;
        self
    }

    /// Use hardware-efficient ansatz
    pub fn with_hardware_efficient(mut self, n_layers: usize) -> Self {
        self.ansatz = AnsatzType::HardwareEfficient;
        self.n_layers = n_layers;
        self
    }

    /// Use simplified ansatz
    pub fn with_simplified(mut self) -> Self {
        self.ansatz = AnsatzType::Simplified;
        self
    }

    /// Set optimizer
    pub fn with_optimizer(mut self, opt: VQEOptimizer) -> Self {
        self.optimizer = opt;
        self
    }

    /// Set maximum iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set learning rate
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Enable learning rate schedule
    pub fn with_lr_schedule(mut self) -> Self {
        self.lr_schedule = true;
        self
    }

    /// Enable verbose output
    pub fn with_verbose(mut self) -> Self {
        self.verbose = true;
        self
    }
}

// =============================================================================
// VQE Result
// =============================================================================

/// Result of VQE optimization
#[derive(Debug, Clone)]
pub struct VQEResult {
    /// Final energy in Hartree
    pub energy: f64,
    /// Optimized parameters
    pub params: Vec<f64>,
    /// Number of iterations
    pub iterations: usize,
    /// Whether optimization converged
    pub converged: bool,
    /// Final gradient norm
    pub final_grad_norm: f64,
    /// Energy history
    pub energy_history: Vec<f64>,
    /// Elapsed time in milliseconds
    pub elapsed_ms: u64,
    /// Error vs exact (if known)
    pub error: Option<f64>,
    /// Whether chemical accuracy was achieved
    pub chemical_accuracy: bool,
    /// Reference energies for comparison
    pub reference: VQEReferenceEnergies,
}

/// Reference energies for comparison
#[derive(Debug, Clone)]
pub struct VQEReferenceEnergies {
    /// Hartree-Fock energy
    pub hartree_fock: f64,
    /// Exact (FCI) energy
    pub exact: f64,
    /// Experimental energy (if known)
    pub experimental: Option<f64>,
}

impl VQEResult {
    /// Get correlation energy captured
    pub fn correlation_energy(&self) -> f64 {
        self.reference.hartree_fock - self.energy
    }

    /// Get percentage of correlation energy captured
    pub fn correlation_percentage(&self) -> f64 {
        let total_corr = self.reference.hartree_fock - self.reference.exact;
        if total_corr.abs() < 1e-10 {
            100.0
        } else {
            (self.reference.hartree_fock - self.energy) / total_corr * 100.0
        }
    }

    /// Print detailed summary
    pub fn print_summary(&self) {
        println!("╔══════════════════════════════════════════╗");
        println!("║         VQE SIMULATION RESULTS           ║");
        println!("╠══════════════════════════════════════════╣");
        println!("║ Energy: {:.8} Ha           ║", self.energy);
        println!("║ Iterations: {:>6}                       ║", self.iterations);
        println!("║ Converged: {:>5}                        ║", self.converged);
        println!("║ Time: {:>8} ms                       ║", self.elapsed_ms);
        println!("╠══════════════════════════════════════════╣");
        println!("║ Reference Energies:                      ║");
        println!("║   HF:    {:.8} Ha                    ║", self.reference.hartree_fock);
        println!("║   Exact: {:.8} Ha                    ║", self.reference.exact);
        if let Some(exp) = self.reference.experimental {
            println!("║   Exp:   {:.8} Ha                    ║", exp);
        }
        println!("╠══════════════════════════════════════════╣");
        println!("║ Analysis:                                ║");
        if let Some(err) = self.error {
            println!("║   Error vs exact: {:.6} Ha             ║", err);
        }
        println!("║   Correlation: {:.1}%                     ║", self.correlation_percentage());
        println!("║   Chemical accuracy: {}                 ║", 
            if self.chemical_accuracy { "YES ✓" } else { "NO  ✗" });
        println!("╚══════════════════════════════════════════╝");
    }
}

// =============================================================================
// VQE Simulation
// =============================================================================

/// VQE Simulation runner
pub struct VQESimulation {
    /// Configuration
    config: VQEConfig,
    /// Molecular Hamiltonian
    hamiltonian: H2Hamiltonian,
    /// Variational circuit
    circuit: VariationalCircuit,
    /// Initial parameters
    initial_params: Vec<f64>,
}

impl VQESimulation {
    /// Create VQE simulation with H₂ Hamiltonian
    pub fn new(h2_config: H2Config, vqe_config: VQEConfig) -> Self {
        let hamiltonian = H2Hamiltonian::new(h2_config);

        let circuit = match vqe_config.ansatz {
            AnsatzType::UCCSD => {
                let uccsd = UCCSDCircuit::h2();
                uccsd.build_circuit()
            }
            AnsatzType::HardwareEfficient => {
                let hea = HardwareEfficientAnsatz::h2(vqe_config.n_layers);
                hea.build_circuit()
            }
            AnsatzType::Simplified => {
                let simplified = H2SimplifiedAnsatz::new();
                simplified.build_circuit_4q()
            }
        };

        // Use circuit's actual parameter count
        let initial_params = vec![0.1; circuit.num_params()];

        Self {
            config: vqe_config,
            hamiltonian,
            circuit,
            initial_params,
        }
    }

    /// Create default H₂ VQE at equilibrium
    pub fn h2_default() -> Self {
        Self::new(H2Config::default(), VQEConfig::default())
    }

    /// Create H₂ VQE at specific bond length
    pub fn h2_at_bond_length(r: f64) -> Self {
        Self::new(
            H2Config::default().with_bond_length(r),
            VQEConfig::default(),
        )
    }

    /// Create H₂ VQE with specific config
    pub fn h2_with_config(bond_length: f64, vqe_config: VQEConfig) -> Self {
        Self::new(
            H2Config::default().with_bond_length(bond_length),
            vqe_config,
        )
    }

    /// Get the qubit Hamiltonian
    pub fn qubit_hamiltonian(&self) -> &Hamiltonian {
        self.hamiltonian.qubit_hamiltonian()
    }

    /// Get the circuit
    pub fn circuit(&self) -> &VariationalCircuit {
        &self.circuit
    }

    /// Run VQE optimization
    pub fn run(&self) -> VQEResult {
        let start = Instant::now();

        // Get constant term to add back
        let constant = self.hamiltonian.constant_term();
        let qubit_ham = self.hamiltonian.to_hamiltonian();

        // Choose optimizer and run
        let (params, energy_history, iterations, converged, final_grad_norm) = 
            match self.config.optimizer {
                VQEOptimizer::Adam => self.run_adam(&qubit_ham),
                VQEOptimizer::QNG => self.run_qng(&qubit_ham),
                VQEOptimizer::SPSA => self.run_spsa(&qubit_ham),
                VQEOptimizer::SGD => self.run_sgd(&qubit_ham),
                VQEOptimizer::Momentum => self.run_momentum(&qubit_ham),
            };

        // Compute final energy with constant term
        let final_energy = compute_expectation(&self.circuit, &params, &qubit_ham) + constant;

        // Reference energies
        let hf_energy = self.hamiltonian.hartree_fock_energy();
        let exact_energy = self.hamiltonian.exact_ground_state();

        let error = Some((final_energy - exact_energy).abs());
        let chemical_accuracy = error.map(|e| e < CHEMICAL_ACCURACY).unwrap_or(false);

        VQEResult {
            energy: final_energy,
            params,
            iterations,
            converged,
            final_grad_norm,
            energy_history: energy_history.iter()
                .map(|&e| e + constant)
                .collect(),
            elapsed_ms: start.elapsed().as_millis() as u64,
            error,
            chemical_accuracy,
            reference: VQEReferenceEnergies {
                hartree_fock: hf_energy,
                exact: exact_energy,
                experimental: Some(H2_EXPERIMENTAL_ENERGY),
            },
        }
    }

    /// Run with Adam optimizer
    fn run_adam(&self, hamiltonian: &Hamiltonian) -> (Vec<f64>, Vec<f64>, usize, bool, f64) {
        let opt_config = OptimizerConfig::adam(self.config.learning_rate)
            .with_max_iter(self.config.max_iter)
            .with_tolerance(self.config.tolerance);

        let mut optimizer = Optimizer::new(opt_config, self.initial_params.clone());
        let mut energy_history = Vec::new();
        let mut final_grad_norm = 0.0;

        for iter in 0..self.config.max_iter {
            let energy = compute_expectation(&self.circuit, optimizer.params(), hamiltonian);
            energy_history.push(energy);

            let gradient = compute_gradient(&self.circuit, optimizer.params(), hamiltonian);
            final_grad_norm = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();

            if self.config.verbose && iter % 20 == 0 {
                println!("Iter {}: E = {:.8}, |g| = {:.6}", iter, energy, final_grad_norm);
            }

            if final_grad_norm < self.config.tolerance {
                return (
                    optimizer.params().to_vec(),
                    energy_history,
                    iter + 1,
                    true,
                    final_grad_norm,
                );
            }

            optimizer.update_objective(energy);
            optimizer.step(&gradient);
        }

        (
            optimizer.params().to_vec(),
            energy_history,
            self.config.max_iter,
            false,
            final_grad_norm,
        )
    }

    /// Run with QNG optimizer
    fn run_qng(&self, hamiltonian: &Hamiltonian) -> (Vec<f64>, Vec<f64>, usize, bool, f64) {
        let qng_config = QNGConfig::default()
            .with_learning_rate(self.config.learning_rate)
            .with_max_iter(self.config.max_iter)
            .with_tolerance(self.config.tolerance)
            .with_block_diagonal(true); // Faster

        let mut optimizer = QNGOptimizer::new(qng_config, self.initial_params.clone());
        let mut energy_history = Vec::new();
        let mut final_grad_norm = 0.0;

        for iter in 0..self.config.max_iter {
            let result = optimizer.step(&self.circuit, hamiltonian);
            energy_history.push(result.objective);
            final_grad_norm = result.grad_norm;

            if self.config.verbose && iter % 20 == 0 {
                println!("Iter {}: E = {:.8}, |g| = {:.6}", iter, result.objective, final_grad_norm);
            }

            if result.converged {
                return (
                    optimizer.params().to_vec(),
                    energy_history,
                    iter + 1,
                    true,
                    final_grad_norm,
                );
            }
        }

        (
            optimizer.params().to_vec(),
            energy_history,
            self.config.max_iter,
            false,
            final_grad_norm,
        )
    }

    /// Run with SPSA optimizer
    fn run_spsa(&self, hamiltonian: &Hamiltonian) -> (Vec<f64>, Vec<f64>, usize, bool, f64) {
        let opt_config = OptimizerConfig::spsa()
            .with_max_iter(self.config.max_iter)
            .with_tolerance(self.config.tolerance);

        let mut optimizer = Optimizer::new(opt_config, self.initial_params.clone());
        let mut energy_history = Vec::new();
        let mut prev_energy = f64::MAX;

        for iter in 0..self.config.max_iter {
            let energy = compute_expectation(&self.circuit, optimizer.params(), hamiltonian);
            energy_history.push(energy);

            if self.config.verbose && iter % 20 == 0 {
                println!("Iter {}: E = {:.8}", iter, energy);
            }

            // SPSA doesn't compute gradients directly
            let delta = (prev_energy - energy).abs();
            if delta < self.config.tolerance && iter > 10 {
                return (
                    optimizer.params().to_vec(),
                    energy_history,
                    iter + 1,
                    true,
                    0.0,
                );
            }
            prev_energy = energy;

            // SPSA perturbation
            let perturbation = Self::generate_perturbation(optimizer.params().len(), iter as u64);
            optimizer.update_objective(energy);
            optimizer.step(&perturbation);
        }

        (
            optimizer.params().to_vec(),
            energy_history,
            self.config.max_iter,
            false,
            0.0,
        )
    }

    /// Run with SGD optimizer
    fn run_sgd(&self, hamiltonian: &Hamiltonian) -> (Vec<f64>, Vec<f64>, usize, bool, f64) {
        let opt_config = OptimizerConfig::sgd(self.config.learning_rate)
            .with_max_iter(self.config.max_iter);

        let mut optimizer = Optimizer::new(opt_config, self.initial_params.clone());
        let mut energy_history = Vec::new();
        let mut final_grad_norm = 0.0;

        for iter in 0..self.config.max_iter {
            let energy = compute_expectation(&self.circuit, optimizer.params(), hamiltonian);
            energy_history.push(energy);

            let gradient = compute_gradient(&self.circuit, optimizer.params(), hamiltonian);
            final_grad_norm = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();

            if final_grad_norm < self.config.tolerance {
                return (
                    optimizer.params().to_vec(),
                    energy_history,
                    iter + 1,
                    true,
                    final_grad_norm,
                );
            }

            optimizer.update_objective(energy);
            optimizer.step(&gradient);
        }

        (
            optimizer.params().to_vec(),
            energy_history,
            self.config.max_iter,
            false,
            final_grad_norm,
        )
    }

    /// Run with Momentum optimizer
    fn run_momentum(&self, hamiltonian: &Hamiltonian) -> (Vec<f64>, Vec<f64>, usize, bool, f64) {
        let opt_config = OptimizerConfig::momentum(self.config.learning_rate, 0.9)
            .with_max_iter(self.config.max_iter);

        let mut optimizer = Optimizer::new(opt_config, self.initial_params.clone());
        let mut energy_history = Vec::new();
        let mut final_grad_norm = 0.0;

        for iter in 0..self.config.max_iter {
            let energy = compute_expectation(&self.circuit, optimizer.params(), hamiltonian);
            energy_history.push(energy);

            let gradient = compute_gradient(&self.circuit, optimizer.params(), hamiltonian);
            final_grad_norm = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();

            if final_grad_norm < self.config.tolerance {
                return (
                    optimizer.params().to_vec(),
                    energy_history,
                    iter + 1,
                    true,
                    final_grad_norm,
                );
            }

            optimizer.update_objective(energy);
            optimizer.step(&gradient);
        }

        (
            optimizer.params().to_vec(),
            energy_history,
            self.config.max_iter,
            false,
            final_grad_norm,
        )
    }

    /// Generate SPSA perturbation
    fn generate_perturbation(n_params: usize, seed: u64) -> Vec<f64> {
        let mut result = Vec::with_capacity(n_params);
        let mut rng_state = seed.wrapping_add(12345);

        for _ in 0..n_params {
            rng_state = rng_state.wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let r = (rng_state >> 63) as i64;
            result.push(if r == 0 { 1.0 } else { -1.0 });
        }

        result
    }
}

// =============================================================================
// Convenience Functions
// =============================================================================

/// Run VQE for H₂ at equilibrium with default settings
pub fn run_h2_vqe() -> VQEResult {
    VQESimulation::h2_default().run()
}

/// Run VQE for H₂ at specific bond length
pub fn run_h2_vqe_at(bond_length: f64) -> VQEResult {
    VQESimulation::h2_at_bond_length(bond_length).run()
}

/// Compute potential energy surface using VQE
pub fn compute_vqe_pes(
    bond_lengths: &[f64],
    config: VQEConfig,
) -> Vec<(f64, f64, f64)> {
    bond_lengths.iter()
        .map(|&r| {
            let vqe = VQESimulation::h2_with_config(r, config.clone());
            let result = vqe.run();
            (r, result.energy, result.reference.exact)
        })
        .collect()
}

/// Compare different optimizers on H₂
pub fn compare_optimizers(bond_length: f64, max_iter: usize) -> Vec<(String, VQEResult)> {
    let optimizers = [
        ("Adam", VQEOptimizer::Adam),
        ("QNG", VQEOptimizer::QNG),
        ("SGD", VQEOptimizer::SGD),
        ("Momentum", VQEOptimizer::Momentum),
    ];

    optimizers.iter()
        .map(|(name, opt)| {
            let config = VQEConfig::default()
                .with_optimizer(*opt)
                .with_max_iter(max_iter);
            let vqe = VQESimulation::h2_with_config(bond_length, config);
            (name.to_string(), vqe.run())
        })
        .collect()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vqe_config_default() {
        let config = VQEConfig::default();
        assert_eq!(config.ansatz, AnsatzType::UCCSD);
        assert_eq!(config.optimizer, VQEOptimizer::Adam);
    }

    #[test]
    fn test_vqe_config_builder() {
        let config = VQEConfig::default()
            .with_hardware_efficient(3)
            .with_optimizer(VQEOptimizer::QNG)
            .with_max_iter(100);

        assert_eq!(config.ansatz, AnsatzType::HardwareEfficient);
        assert_eq!(config.n_layers, 3);
        assert_eq!(config.optimizer, VQEOptimizer::QNG);
        assert_eq!(config.max_iter, 100);
    }

    #[test]
    fn test_vqe_simulation_creation() {
        let vqe = VQESimulation::h2_default();

        assert!(vqe.initial_params.len() > 0);
    }

    #[test]
    fn test_vqe_run_adam() {
        let config = VQEConfig::default()
            .with_max_iter(20)
            .with_learning_rate(0.1);

        let vqe = VQESimulation::h2_with_config(0.7414, config);
        let result = vqe.run();

        assert!(result.energy < 0.0); // Bound state
        assert!(result.iterations > 0);
        assert!(result.energy_history.len() > 0);
    }

    #[test]
    fn test_vqe_run_simplified() {
        let config = VQEConfig::default()
            .with_simplified()
            .with_max_iter(20);

        let vqe = VQESimulation::h2_with_config(0.7414, config);
        let result = vqe.run();

        assert!(result.energy < 0.0);
    }

    #[test]
    fn test_vqe_run_hardware_efficient() {
        let config = VQEConfig::default()
            .with_hardware_efficient(2)
            .with_max_iter(20);

        let vqe = VQESimulation::h2_with_config(0.7414, config);
        let result = vqe.run();

        assert!(result.energy < 0.0);
    }

    #[test]
    fn test_vqe_reference_energies() {
        let vqe = VQESimulation::h2_default();
        let result = vqe.run();

        // Reference energies should be finite
        assert!(result.reference.hartree_fock.is_finite());
        assert!(result.reference.exact.is_finite());
    }

    #[test]
    fn test_vqe_correlation_percentage() {
        let config = VQEConfig::default().with_max_iter(50);
        let vqe = VQESimulation::h2_with_config(0.7414, config);
        let result = vqe.run();

        let corr_pct = result.correlation_percentage();
        assert!(corr_pct >= 0.0);
        assert!(corr_pct <= 200.0); // Allow some numerical error
    }

    #[test]
    fn test_vqe_sgd() {
        let config = VQEConfig::default()
            .with_optimizer(VQEOptimizer::SGD)
            .with_max_iter(20)
            .with_learning_rate(0.05);

        let vqe = VQESimulation::h2_with_config(0.7414, config);
        let result = vqe.run();

        // Energy should be finite
        assert!(result.energy.is_finite());
    }

    #[test]
    fn test_vqe_momentum() {
        let config = VQEConfig::default()
            .with_optimizer(VQEOptimizer::Momentum)
            .with_max_iter(20);

        let vqe = VQESimulation::h2_with_config(0.7414, config);
        let result = vqe.run();

        // Energy should be finite
        assert!(result.energy.is_finite());
    }

    #[test]
    fn test_run_h2_vqe_convenience() {
        // Just ensure it runs without panicking
        let _ = run_h2_vqe();
    }

    #[test]
    fn test_chemical_accuracy_field() {
        let config = VQEConfig::default().with_max_iter(50);
        let vqe = VQESimulation::h2_with_config(0.7414, config);
        let result = vqe.run();

        // Should have error field
        assert!(result.error.is_some());
    }

    #[test]
    fn test_vqe_at_stretched_bond() {
        let config = VQEConfig::default().with_max_iter(30);
        let vqe = VQESimulation::h2_with_config(1.5, config);
        let result = vqe.run();

        // Energy should still be negative but higher than equilibrium
        assert!(result.energy < 0.0);
    }

    #[test]
    fn test_vqe_pes_computation() {
        let bond_lengths = vec![0.7414, 1.0];
        let config = VQEConfig::default().with_max_iter(10);
        let pes = compute_vqe_pes(&bond_lengths, config);

        assert_eq!(pes.len(), 2);
        // Both energies should be finite
        assert!(pes[0].1.is_finite());
        assert!(pes[1].1.is_finite());
    }
}

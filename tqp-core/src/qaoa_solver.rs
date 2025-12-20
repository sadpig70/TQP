//! QAOA Solver
//!
//! Sprint 3 Week 6 Day 3: Complete QAOA optimization pipeline
//!
//! This module provides a complete QAOA solver that:
//! 1. Builds the cost Hamiltonian from the problem
//! 2. Constructs the QAOA circuit
//! 3. Optimizes parameters using classical optimizers
//! 4. Samples solutions from the optimized state
//!
//! # Example
//!
//! ```ignore
//! use tqp_core::qaoa_solver::{QAOASolver, QAOASolverConfig};
//! use tqp_core::maxcut::instances;
//!
//! let problem = instances::triangle();
//! let solver = QAOASolver::new(problem, 2); // p=2 layers
//!
//! let result = solver.solve();
//! println!("Best cut: {}", result.best_cut_value);
//! println!("Approximation ratio: {:.2}", result.approximation_ratio);
//! ```

use std::time::Instant;

use crate::autodiff::{compute_expectation, compute_gradient, Hamiltonian};
use crate::optimizer::{Optimizer, OptimizerConfig};
use crate::maxcut::{MaxCutProblem, Graph};
use crate::qaoa_ansatz::{QAOAAnsatz, QAOAConfig};
use crate::state::TQPState;

// =============================================================================
// Constants
// =============================================================================

/// Default maximum iterations for QAOA optimization
pub const DEFAULT_QAOA_MAX_ITER: usize = 100;

/// Default learning rate for QAOA
pub const DEFAULT_QAOA_LR: f64 = 0.1;

/// Default number of measurement samples
pub const DEFAULT_SAMPLES: usize = 1000;

// =============================================================================
// Solver Configuration
// =============================================================================

/// QAOA optimizer type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QAOAOptimizer {
    /// Adam optimizer
    Adam,
    /// Gradient descent with momentum
    Momentum,
    /// Simple SGD
    SGD,
    /// COBYLA (gradient-free, not implemented)
    COBYLA,
}

/// QAOA solver configuration
#[derive(Debug, Clone)]
pub struct QAOASolverConfig {
    /// Number of QAOA layers (p)
    pub p: usize,
    /// Optimizer type
    pub optimizer: QAOAOptimizer,
    /// Maximum iterations
    pub max_iter: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Number of measurement samples
    pub n_samples: usize,
    /// Random seed
    pub seed: u64,
    /// Verbose output
    pub verbose: bool,
}

impl Default for QAOASolverConfig {
    fn default() -> Self {
        Self {
            p: 1,
            optimizer: QAOAOptimizer::Adam,
            max_iter: DEFAULT_QAOA_MAX_ITER,
            learning_rate: DEFAULT_QAOA_LR,
            tolerance: 1e-6,
            n_samples: DEFAULT_SAMPLES,
            seed: 42,
            verbose: false,
        }
    }
}

impl QAOASolverConfig {
    /// Set number of layers
    pub fn with_layers(mut self, p: usize) -> Self {
        self.p = p;
        self
    }

    /// Set optimizer
    pub fn with_optimizer(mut self, opt: QAOAOptimizer) -> Self {
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

    /// Set number of samples
    pub fn with_samples(mut self, n: usize) -> Self {
        self.n_samples = n;
        self
    }

    /// Enable verbose output
    pub fn with_verbose(mut self) -> Self {
        self.verbose = true;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

// =============================================================================
// QAOA Result
// =============================================================================

/// Result of QAOA optimization
#[derive(Debug, Clone)]
pub struct QAOAResult {
    /// Best bitstring found
    pub best_bitstring: Vec<bool>,
    /// Best cut value
    pub best_cut_value: f64,
    /// Optimal parameters [γ_1, β_1, ..., γ_p, β_p]
    pub optimal_params: Vec<f64>,
    /// Final expectation value ⟨C⟩
    pub expectation_value: f64,
    /// Approximation ratio (if optimal known)
    pub approximation_ratio: Option<f64>,
    /// Number of iterations
    pub iterations: usize,
    /// Whether optimization converged
    pub converged: bool,
    /// Energy history during optimization
    pub energy_history: Vec<f64>,
    /// Elapsed time in milliseconds
    pub elapsed_ms: u64,
    /// Sample statistics
    pub sample_stats: SampleStats,
}

/// Statistics from sampling
#[derive(Debug, Clone)]
pub struct SampleStats {
    /// Number of samples
    pub n_samples: usize,
    /// Mean cut value
    pub mean_cut: f64,
    /// Standard deviation
    pub std_cut: f64,
    /// Best cut from samples
    pub best_cut: f64,
    /// Probability of best solution
    pub best_prob: f64,
    /// Top 5 solutions with probabilities
    pub top_solutions: Vec<(Vec<bool>, f64, f64)>, // (bitstring, cut_value, probability)
}

impl QAOAResult {
    /// Print detailed summary
    pub fn print_summary(&self) {
        println!("╔══════════════════════════════════════════╗");
        println!("║           QAOA SOLVER RESULTS            ║");
        println!("╠══════════════════════════════════════════╣");
        println!("║ Best cut value: {:.4}                    ", self.best_cut_value);
        println!("║ Expectation ⟨C⟩: {:.6}                 ", self.expectation_value);
        if let Some(ratio) = self.approximation_ratio {
            println!("║ Approx. ratio: {:.4}                    ", ratio);
        }
        println!("║ Iterations: {}                          ", self.iterations);
        println!("║ Converged: {}                           ", self.converged);
        println!("║ Time: {} ms                             ", self.elapsed_ms);
        println!("╠══════════════════════════════════════════╣");
        println!("║ Optimal parameters:                      ║");
        for (i, &p) in self.optimal_params.iter().enumerate() {
            let name = if i % 2 == 0 { "γ" } else { "β" };
            let layer = i / 2 + 1;
            println!("║   {}_{}: {:.6}                         ", name, layer, p);
        }
        println!("╠══════════════════════════════════════════╣");
        println!("║ Best bitstring: {:?}                    ", self.best_bitstring);
        println!("╠══════════════════════════════════════════╣");
        println!("║ Sample Statistics:                       ║");
        println!("║   Mean cut: {:.4}                       ", self.sample_stats.mean_cut);
        println!("║   Std cut: {:.4}                        ", self.sample_stats.std_cut);
        println!("║   Best prob: {:.4}                      ", self.sample_stats.best_prob);
        println!("╚══════════════════════════════════════════╝");
    }
}

// =============================================================================
// QAOA Solver
// =============================================================================

/// QAOA Solver for MaxCut
pub struct QAOASolver {
    /// Problem instance
    problem: MaxCutProblem,
    /// Configuration
    config: QAOASolverConfig,
    /// QAOA ansatz
    ansatz: QAOAAnsatz,
}

impl QAOASolver {
    /// Create QAOA solver for MaxCut problem
    pub fn new(problem: MaxCutProblem, p: usize) -> Self {
        Self::with_config(problem, QAOASolverConfig::default().with_layers(p))
    }

    /// Create solver with configuration
    pub fn with_config(problem: MaxCutProblem, config: QAOASolverConfig) -> Self {
        let ansatz = QAOAAnsatz::new(&problem, config.p);

        Self {
            problem,
            config,
            ansatz,
        }
    }

    /// Create solver from graph
    pub fn from_graph(graph: Graph, p: usize) -> Self {
        let problem = MaxCutProblem::new(graph);
        Self::new(problem, p)
    }

    /// Get the problem
    pub fn problem(&self) -> &MaxCutProblem {
        &self.problem
    }

    /// Run QAOA optimization
    pub fn solve(&mut self) -> QAOAResult {
        let start = Instant::now();

        // Build circuit and get Hamiltonian
        let circuit = self.ansatz.build_circuit();
        let hamiltonian = self.problem.cost_hamiltonian().clone();
        let constant_offset = self.problem.constant_offset();

        // Initialize parameters
        let initial_params = self.ansatz.initial_params(self.config.seed);

        // Run optimization
        let (params, energy_history, iterations, converged) = 
            self.run_optimization(&circuit, &hamiltonian, initial_params);

        // Compute final expectation value
        let expectation = compute_expectation(&circuit, &params, &hamiltonian);
        let final_energy = constant_offset + expectation;

        // Sample solutions
        let sample_stats = self.sample_solutions(&circuit, &params);

        // Find best solution
        let (best_bitstring, best_cut) = self.find_best_solution(&sample_stats);

        // Compute approximation ratio if we can compute optimal
        let approximation_ratio = if self.problem.graph().n_vertices() <= 20 {
            let optimal = self.problem.optimal_cut_value();
            Some(best_cut / optimal)
        } else {
            None
        };

        QAOAResult {
            best_bitstring,
            best_cut_value: best_cut,
            optimal_params: params,
            expectation_value: final_energy,
            approximation_ratio,
            iterations,
            converged,
            energy_history,
            elapsed_ms: start.elapsed().as_millis() as u64,
            sample_stats,
        }
    }

    /// Run the optimization loop
    fn run_optimization(
        &self,
        circuit: &crate::autodiff::VariationalCircuit,
        hamiltonian: &Hamiltonian,
        initial_params: Vec<f64>,
    ) -> (Vec<f64>, Vec<f64>, usize, bool) {
        let opt_config = match self.config.optimizer {
            QAOAOptimizer::Adam => OptimizerConfig::adam(self.config.learning_rate),
            QAOAOptimizer::Momentum => OptimizerConfig::momentum(self.config.learning_rate, 0.9),
            QAOAOptimizer::SGD => OptimizerConfig::sgd(self.config.learning_rate),
            QAOAOptimizer::COBYLA => OptimizerConfig::sgd(self.config.learning_rate), // Fallback
        };

        let opt_config = opt_config.with_max_iter(self.config.max_iter);
        let mut optimizer = Optimizer::new(opt_config, initial_params);

        let mut energy_history = Vec::new();
        let mut converged = false;

        for iter in 0..self.config.max_iter {
            // Compute expectation value (we minimize, so negate for maximization)
            let energy = compute_expectation(circuit, optimizer.params(), hamiltonian);
            energy_history.push(energy);

            // Compute gradient
            let gradient = compute_gradient(circuit, optimizer.params(), hamiltonian);
            let grad_norm: f64 = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();

            if self.config.verbose && iter % 20 == 0 {
                println!("Iter {}: E = {:.6}, |∇| = {:.6}", iter, energy, grad_norm);
            }

            // Check convergence
            if grad_norm < self.config.tolerance {
                converged = true;
                break;
            }

            // Update parameters (minimize negative expectation = maximize expectation)
            optimizer.update_objective(energy);
            optimizer.step(&gradient);
        }

        let iterations = energy_history.len();
        (
            optimizer.params().to_vec(),
            energy_history,
            iterations,
            converged,
        )
    }

    /// Sample solutions from the optimized QAOA state
    fn sample_solutions(
        &self,
        circuit: &crate::autodiff::VariationalCircuit,
        params: &[f64],
    ) -> SampleStats {
        let n_qubits = self.ansatz.n_qubits();

        // Execute circuit to get final state
        let state = circuit.execute(params);

        // Get probability distribution
        let probs: Vec<f64> = state.state_vector.iter()
            .map(|a| a.norm_sqr())
            .collect();

        // Sample and compute cut values
        let mut samples = Vec::with_capacity(self.config.n_samples);
        let mut rng_state = self.config.seed;

        for _ in 0..self.config.n_samples {
            // Sample from distribution
            rng_state = rng_state.wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let r = (rng_state >> 33) as f64 / (1u64 << 31) as f64;

            let mut cumsum = 0.0;
            let mut sampled_idx = 0;
            for (idx, &p) in probs.iter().enumerate() {
                cumsum += p;
                if cumsum > r {
                    sampled_idx = idx;
                    break;
                }
            }

            // Convert to bitstring
            let bitstring: Vec<bool> = (0..n_qubits)
                .map(|q| (sampled_idx >> q) & 1 == 1)
                .collect();

            let cut_value = self.problem.evaluate_cut(&bitstring);
            samples.push((bitstring, cut_value));
        }

        // Compute statistics
        let cut_values: Vec<f64> = samples.iter().map(|(_, c)| *c).collect();
        let mean_cut = cut_values.iter().sum::<f64>() / cut_values.len() as f64;
        let variance = cut_values.iter()
            .map(|c| (c - mean_cut).powi(2))
            .sum::<f64>() / cut_values.len() as f64;
        let std_cut = variance.sqrt();

        // Find top solutions
        let mut solution_counts: std::collections::HashMap<Vec<bool>, usize> = 
            std::collections::HashMap::new();
        for (bs, _) in &samples {
            *solution_counts.entry(bs.clone()).or_insert(0) += 1;
        }

        let mut solutions_with_cuts: Vec<_> = solution_counts.into_iter()
            .map(|(bs, count)| {
                let cut = self.problem.evaluate_cut(&bs);
                let prob = count as f64 / self.config.n_samples as f64;
                (bs, cut, prob)
            })
            .collect();

        solutions_with_cuts.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top_solutions: Vec<_> = solutions_with_cuts.iter().take(5).cloned().collect();

        let best_cut = samples.iter().map(|(_, c)| *c).fold(0.0, f64::max);
        let best_prob = top_solutions.first().map(|(_, _, p)| *p).unwrap_or(0.0);

        SampleStats {
            n_samples: self.config.n_samples,
            mean_cut,
            std_cut,
            best_cut,
            best_prob,
            top_solutions,
        }
    }

    /// Find best solution from samples
    fn find_best_solution(&self, stats: &SampleStats) -> (Vec<bool>, f64) {
        if let Some((bs, cut, _)) = stats.top_solutions.first() {
            (bs.clone(), *cut)
        } else {
            // Fallback: random
            let (bs, cut) = self.problem.random_cut(self.config.seed);
            (bs, cut)
        }
    }

    /// Perform grid search over parameters
    /// Note: Only practical for small problems with few parameters
    pub fn grid_search(&mut self, points_per_param: usize) -> QAOAResult {
        let start = Instant::now();
        let circuit = self.ansatz.build_circuit();
        let hamiltonian = self.problem.cost_hamiltonian().clone();
        let constant_offset = self.problem.constant_offset();

        let grid = self.ansatz.param_grid(points_per_param);

        let mut best_params = self.ansatz.initial_params(self.config.seed);
        let mut best_energy = f64::MAX;
        let mut energy_history = Vec::new();

        for params in &grid {
            let energy = compute_expectation(&circuit, params, &hamiltonian);
            energy_history.push(energy);

            if energy < best_energy {
                best_energy = energy;
                best_params = params.clone();
            }
        }

        // Sample from best parameters
        let sample_stats = self.sample_solutions(&circuit, &best_params);
        let (best_bitstring, best_cut) = self.find_best_solution(&sample_stats);

        let approximation_ratio = if self.problem.graph().n_vertices() <= 20 {
            Some(best_cut / self.problem.optimal_cut_value())
        } else {
            None
        };

        QAOAResult {
            best_bitstring,
            best_cut_value: best_cut,
            optimal_params: best_params,
            expectation_value: constant_offset + best_energy,
            approximation_ratio,
            iterations: grid.len(),
            converged: true,
            energy_history,
            elapsed_ms: start.elapsed().as_millis() as u64,
            sample_stats,
        }
    }
}

// =============================================================================
// Convenience Functions
// =============================================================================

/// Solve MaxCut with default QAOA settings
pub fn solve_maxcut(graph: Graph, p: usize) -> QAOAResult {
    let problem = MaxCutProblem::new(graph);
    let mut solver = QAOASolver::new(problem, p);
    solver.solve()
}

/// Solve MaxCut with custom configuration
pub fn solve_maxcut_with_config(graph: Graph, config: QAOASolverConfig) -> QAOAResult {
    let problem = MaxCutProblem::new(graph);
    let mut solver = QAOASolver::with_config(problem, config);
    solver.solve()
}

/// Compare QAOA with different layer counts
pub fn compare_layers(graph: Graph, max_p: usize, max_iter: usize) -> Vec<(usize, QAOAResult)> {
    let problem = MaxCutProblem::new(graph);

    (1..=max_p)
        .map(|p| {
            let config = QAOASolverConfig::default()
                .with_layers(p)
                .with_max_iter(max_iter);
            let mut solver = QAOASolver::with_config(problem.clone(), config);
            let result = solver.solve();
            (p, result)
        })
        .collect()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::maxcut::instances;

    #[test]
    fn test_solver_config_default() {
        let config = QAOASolverConfig::default();
        assert_eq!(config.p, 1);
        assert_eq!(config.optimizer, QAOAOptimizer::Adam);
    }

    #[test]
    fn test_solver_config_builder() {
        let config = QAOASolverConfig::default()
            .with_layers(3)
            .with_max_iter(50)
            .with_learning_rate(0.05);

        assert_eq!(config.p, 3);
        assert_eq!(config.max_iter, 50);
        assert_eq!(config.learning_rate, 0.05);
    }

    #[test]
    fn test_solver_creation() {
        let problem = instances::triangle();
        let solver = QAOASolver::new(problem, 2);

        assert_eq!(solver.ansatz.n_layers(), 2);
    }

    #[test]
    fn test_solver_solve_triangle() {
        let problem = instances::triangle();
        let config = QAOASolverConfig::default()
            .with_layers(1)
            .with_max_iter(20)
            .with_samples(100);

        let mut solver = QAOASolver::with_config(problem, config);
        let result = solver.solve();

        // Should find a valid cut
        assert!(result.best_cut_value >= 0.0);
        assert!(result.best_cut_value <= 3.0); // Max possible for triangle
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_solver_solve_square() {
        let problem = instances::square();
        let config = QAOASolverConfig::default()
            .with_layers(1)
            .with_max_iter(20)
            .with_samples(100);

        let mut solver = QAOASolver::with_config(problem, config);
        let result = solver.solve();

        assert!(result.best_cut_value >= 0.0);
        assert!(result.best_cut_value <= 4.0);
    }

    #[test]
    fn test_solver_approximation_ratio() {
        let problem = instances::triangle();
        let config = QAOASolverConfig::default()
            .with_layers(2)
            .with_max_iter(30)
            .with_samples(200);

        let mut solver = QAOASolver::with_config(problem, config);
        let result = solver.solve();

        // Should have approximation ratio since n <= 20
        assert!(result.approximation_ratio.is_some());
        let ratio = result.approximation_ratio.unwrap();
        assert!(ratio >= 0.0 && ratio <= 1.0);
    }

    #[test]
    fn test_solver_sample_stats() {
        let problem = instances::triangle();
        let config = QAOASolverConfig::default()
            .with_samples(500);

        let mut solver = QAOASolver::with_config(problem, config);
        let result = solver.solve();

        assert_eq!(result.sample_stats.n_samples, 500);
        assert!(result.sample_stats.mean_cut >= 0.0);
        assert!(result.sample_stats.std_cut >= 0.0);
    }

    #[test]
    fn test_grid_search() {
        let problem = instances::triangle();
        let config = QAOASolverConfig::default()
            .with_layers(1)
            .with_samples(100);

        let mut solver = QAOASolver::with_config(problem, config);
        let result = solver.grid_search(3); // Small grid for testing

        assert!(result.best_cut_value >= 0.0);
    }

    #[test]
    fn test_solve_maxcut_convenience() {
        let graph = Graph::triangle();
        let result = solve_maxcut(graph, 1);

        assert!(result.best_cut_value >= 0.0);
    }

    #[test]
    fn test_from_graph() {
        let graph = Graph::complete(4);
        let solver = QAOASolver::from_graph(graph, 2);

        assert_eq!(solver.ansatz.n_qubits(), 4);
    }

    #[test]
    fn test_different_optimizers() {
        let problem = instances::triangle();

        for opt in [QAOAOptimizer::Adam, QAOAOptimizer::SGD, QAOAOptimizer::Momentum] {
            let config = QAOASolverConfig::default()
                .with_optimizer(opt)
                .with_max_iter(10)
                .with_samples(50);

            let mut solver = QAOASolver::with_config(problem.clone(), config);
            let result = solver.solve();

            assert!(result.best_cut_value >= 0.0);
        }
    }

    #[test]
    fn test_energy_decreases() {
        let problem = instances::square();
        let config = QAOASolverConfig::default()
            .with_layers(1)
            .with_max_iter(30);

        let mut solver = QAOASolver::with_config(problem, config);
        let result = solver.solve();

        // Energy should generally decrease (not strictly due to noise)
        if result.energy_history.len() >= 2 {
            let first = result.energy_history.first().unwrap();
            let last = result.energy_history.last().unwrap();
            // Allow some tolerance
            assert!(*last <= *first + 0.5);
        }
    }

    #[test]
    fn test_verbose_mode() {
        let problem = instances::triangle();
        let config = QAOASolverConfig::default()
            .with_verbose()
            .with_max_iter(5);

        let mut solver = QAOASolver::with_config(problem, config);
        let _ = solver.solve(); // Should print output
    }
}

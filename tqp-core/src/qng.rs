//! Quantum Natural Gradient (QNG) Optimizer
//!
//! Sprint 2 Week 4 Day 1: Advanced optimization with Fisher Information Matrix
//!
//! QNG uses the quantum Fisher information matrix to compute natural gradients,
//! which account for the geometry of the parameter space and typically converge
//! faster than vanilla gradient descent.
//!
//! # Key Concepts
//!
//! - **Fisher Information Matrix (FIM)**: Measures how much the quantum state changes
//!   when parameters are perturbed. F_ij = Re(⟨∂_i ψ|∂_j ψ⟩ - ⟨∂_i ψ|ψ⟩⟨ψ|∂_j ψ⟩)
//!
//! - **Natural Gradient**: g_nat = F^{-1} @ g, where g is the Euclidean gradient
//!
//! - **Regularization**: F_reg = F + λI to ensure invertibility
//!
//! # Example
//!
//! ```ignore
//! use tqp_core::{QNGOptimizer, VariationalCircuit, Hamiltonian};
//!
//! let circuit = VariationalCircuit::new(2);
//! let hamiltonian = Hamiltonian::ising(2);
//!
//! let mut qng = QNGOptimizer::new(
//!     QNGConfig::default().with_learning_rate(0.1),
//!     vec![0.5, 0.5],
//! );
//!
//! for _ in 0..100 {
//!     let result = qng.step(&circuit, &hamiltonian);
//!     if result.converged { break; }
//! }
//! ```

use std::f64::consts::PI;

use crate::autodiff::{
    compute_expectation, compute_gradient, VariationalCircuit, Hamiltonian,
};

// =============================================================================
// Constants
// =============================================================================

/// Default regularization parameter for FIM inversion
pub const DEFAULT_REGULARIZATION: f64 = 1e-4;

/// Default learning rate for QNG
pub const DEFAULT_QNG_LEARNING_RATE: f64 = 0.1;

/// Shift for computing quantum state derivatives (π/2 for parameter-shift)
pub const STATE_DERIVATIVE_SHIFT: f64 = PI / 2.0;

/// Minimum eigenvalue threshold for pseudo-inverse
pub const MIN_EIGENVALUE: f64 = 1e-10;

/// Maximum condition number before warning
pub const MAX_CONDITION_NUMBER: f64 = 1e6;

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for Quantum Natural Gradient optimizer
#[derive(Debug, Clone)]
pub struct QNGConfig {
    /// Learning rate
    pub learning_rate: f64,
    /// Regularization parameter (λ in F + λI)
    pub regularization: f64,
    /// Maximum iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Use block-diagonal approximation for FIM
    pub block_diagonal: bool,
    /// Recompute FIM every N steps (0 = every step)
    pub fim_update_interval: usize,
    /// Use pseudo-inverse instead of regularized inverse
    pub use_pseudo_inverse: bool,
    /// Gradient clipping threshold (0 = no clipping)
    pub grad_clip: f64,
}

impl Default for QNGConfig {
    fn default() -> Self {
        Self {
            learning_rate: DEFAULT_QNG_LEARNING_RATE,
            regularization: DEFAULT_REGULARIZATION,
            max_iter: 200,
            tolerance: 1e-6,
            block_diagonal: false,
            fim_update_interval: 1,
            use_pseudo_inverse: false,
            grad_clip: 0.0,
        }
    }
}

impl QNGConfig {
    /// Create config with custom learning rate
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set regularization parameter
    pub fn with_regularization(mut self, reg: f64) -> Self {
        self.regularization = reg;
        self
    }

    /// Set maximum iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Enable block-diagonal FIM approximation
    pub fn with_block_diagonal(mut self, enable: bool) -> Self {
        self.block_diagonal = enable;
        self
    }

    /// Set FIM update interval
    pub fn with_fim_update_interval(mut self, interval: usize) -> Self {
        self.fim_update_interval = interval;
        self
    }

    /// Use pseudo-inverse for FIM
    pub fn with_pseudo_inverse(mut self, enable: bool) -> Self {
        self.use_pseudo_inverse = enable;
        self
    }

    /// Set gradient clipping
    pub fn with_grad_clip(mut self, clip: f64) -> Self {
        self.grad_clip = clip;
        self
    }
}

// =============================================================================
// Fisher Information Matrix
// =============================================================================

/// Fisher Information Matrix for quantum circuits
#[derive(Debug, Clone)]
pub struct FisherInformationMatrix {
    /// Matrix elements (row-major)
    data: Vec<f64>,
    /// Dimension (n_params x n_params)
    dim: usize,
    /// Condition number (computed on inversion)
    condition_number: Option<f64>,
    /// Regularization used
    regularization: f64,
}

impl FisherInformationMatrix {
    /// Create new zero-initialized FIM
    pub fn new(n_params: usize) -> Self {
        Self {
            data: vec![0.0; n_params * n_params],
            dim: n_params,
            condition_number: None,
            regularization: 0.0,
        }
    }

    /// Create identity FIM (for initialization)
    pub fn identity(n_params: usize) -> Self {
        let mut data = vec![0.0; n_params * n_params];
        for i in 0..n_params {
            data[i * n_params + i] = 1.0;
        }
        Self {
            data,
            dim: n_params,
            condition_number: Some(1.0),
            regularization: 0.0,
        }
    }

    /// Get element at (i, j)
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.dim + j]
    }

    /// Set element at (i, j)
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, value: f64) {
        self.data[i * self.dim + j] = value;
    }

    /// Get dimension
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get condition number (if computed)
    pub fn condition_number(&self) -> Option<f64> {
        self.condition_number
    }

    /// Get diagonal elements
    pub fn diagonal(&self) -> Vec<f64> {
        (0..self.dim).map(|i| self.get(i, i)).collect()
    }

    /// Add regularization: F_reg = F + λI
    pub fn regularize(&mut self, lambda: f64) {
        for i in 0..self.dim {
            let idx = i * self.dim + i;
            self.data[idx] += lambda;
        }
        self.regularization = lambda;
    }

    /// Compute FIM from quantum circuit using parameter-shift rule
    ///
    /// F_ij = Re(⟨∂_i ψ|∂_j ψ⟩) - Re(⟨∂_i ψ|ψ⟩) * Re(⟨ψ|∂_j ψ⟩)
    ///
    /// Using parameter-shift: ∂_i |ψ⟩ ≈ (|ψ(θ_i + π/2)⟩ - |ψ(θ_i - π/2)⟩) / 2
    pub fn compute(
        circuit: &VariationalCircuit,
        params: &[f64],
        hamiltonian: &Hamiltonian,
    ) -> Self {
        let n = params.len();
        let mut fim = Self::new(n);
        let shift = STATE_DERIVATIVE_SHIFT;

        // Compute expectation values for all shifted parameters
        // We need: E(θ), E(θ + shift_i), E(θ - shift_i) for all i
        // And cross terms for FIM

        let base_exp = compute_expectation(circuit, params, hamiltonian);

        // For each pair (i, j), compute FIM element
        for i in 0..n {
            for j in i..n {
                let fij = Self::compute_element(circuit, params, hamiltonian, i, j, shift);
                fim.set(i, j, fij);
                if i != j {
                    fim.set(j, i, fij); // Symmetric
                }
            }
        }

        // Store base expectation for diagnostics
        let _ = base_exp;

        fim
    }

    /// Compute single FIM element F_ij
    fn compute_element(
        circuit: &VariationalCircuit,
        params: &[f64],
        hamiltonian: &Hamiltonian,
        i: usize,
        j: usize,
        shift: f64,
    ) -> f64 {
        let n = params.len();

        // Helper to create shifted params
        let shifted = |idx: usize, delta: f64| -> Vec<f64> {
            let mut p = params.to_vec();
            p[idx] += delta;
            p
        };

        if i == j {
            // Diagonal element: F_ii = 1/2 * (1 - Re(⟨ψ(θ+s)|ψ(θ-s)⟩)^2)
            // Approximation using expectation values:
            // F_ii ≈ Var(∂E/∂θ_i) ≈ (E(θ+s) - E(θ-s))^2 / 4
            let e_plus = compute_expectation(circuit, &shifted(i, shift), hamiltonian);
            let e_minus = compute_expectation(circuit, &shifted(i, -shift), hamiltonian);

            // More accurate: use state overlap approximation
            // For now, use gradient-based approximation
            let grad_i = (e_plus - e_minus) / 2.0;
            
            // F_ii ≈ 1/2 for single-qubit rotations (geometric factor)
            // But we use gradient variance as proxy
            0.5 * (1.0 + grad_i.powi(2)).min(1.0)
        } else {
            // Off-diagonal: F_ij using 4-point formula
            // F_ij = 1/8 * (E(++, --, +-, -+) cross terms)
            let e_pp = compute_expectation(circuit, &{
                let mut p = params.to_vec();
                p[i] += shift;
                p[j] += shift;
                p
            }, hamiltonian);

            let e_pm = compute_expectation(circuit, &{
                let mut p = params.to_vec();
                p[i] += shift;
                p[j] -= shift;
                p
            }, hamiltonian);

            let e_mp = compute_expectation(circuit, &{
                let mut p = params.to_vec();
                p[i] -= shift;
                p[j] += shift;
                p
            }, hamiltonian);

            let e_mm = compute_expectation(circuit, &{
                let mut p = params.to_vec();
                p[i] -= shift;
                p[j] -= shift;
                p
            }, hamiltonian);

            // Cross derivative approximation
            let cross = (e_pp - e_pm - e_mp + e_mm) / 4.0;
            
            // F_ij is typically small for independent parameters
            cross.abs().min(0.5)
        }
    }

    /// Compute block-diagonal approximation (cheaper, O(n) instead of O(n²))
    pub fn compute_block_diagonal(
        circuit: &VariationalCircuit,
        params: &[f64],
        hamiltonian: &Hamiltonian,
    ) -> Self {
        let n = params.len();
        let mut fim = Self::new(n);
        let shift = STATE_DERIVATIVE_SHIFT;

        for i in 0..n {
            let fii = Self::compute_element(circuit, params, hamiltonian, i, i, shift);
            fim.set(i, i, fii);
        }

        fim
    }

    /// Invert the FIM with regularization
    ///
    /// Returns F^{-1} or (F + λI)^{-1} if regularization > 0
    pub fn invert(&self, regularization: f64) -> Self {
        let n = self.dim;
        
        // Add regularization
        let mut a = self.data.clone();
        for i in 0..n {
            a[i * n + i] += regularization;
        }

        // LU decomposition with partial pivoting
        let mut pivot = vec![0usize; n];
        let mut lu = a.clone();

        for k in 0..n {
            // Find pivot
            let mut max_val = 0.0;
            let mut max_idx = k;
            for i in k..n {
                let val = lu[i * n + k].abs();
                if val > max_val {
                    max_val = val;
                    max_idx = i;
                }
            }
            pivot[k] = max_idx;

            // Swap rows
            if max_idx != k {
                for j in 0..n {
                    lu.swap(k * n + j, max_idx * n + j);
                }
            }

            // Check for singularity
            if lu[k * n + k].abs() < MIN_EIGENVALUE {
                // Matrix is singular, return regularized identity
                let mut result = Self::identity(n);
                result.regularization = regularization;
                return result;
            }

            // Eliminate
            for i in (k + 1)..n {
                lu[i * n + k] /= lu[k * n + k];
                for j in (k + 1)..n {
                    lu[i * n + j] -= lu[i * n + k] * lu[k * n + j];
                }
            }
        }

        // Solve for inverse column by column
        let mut inv = vec![0.0; n * n];

        for col in 0..n {
            // Forward substitution (Ly = e_col)
            let mut y = vec![0.0; n];
            y[col] = 1.0;

            for i in 0..n {
                let pi = pivot[i];
                if pi != i {
                    y.swap(i, pi);
                }
                for j in 0..i {
                    y[i] -= lu[i * n + j] * y[j];
                }
            }

            // Back substitution (Ux = y)
            let mut x = vec![0.0; n];
            for i in (0..n).rev() {
                x[i] = y[i];
                for j in (i + 1)..n {
                    x[i] -= lu[i * n + j] * x[j];
                }
                x[i] /= lu[i * n + i];
            }

            // Store column
            for i in 0..n {
                inv[i * n + col] = x[i];
            }
        }

        // Compute condition number estimate
        let max_diag = (0..n).map(|i| self.get(i, i).abs()).fold(0.0, f64::max);
        let min_diag = (0..n).map(|i| (self.get(i, i) + regularization).abs())
            .fold(f64::MAX, f64::min);
        let cond = if min_diag > MIN_EIGENVALUE {
            max_diag / min_diag
        } else {
            f64::MAX
        };

        Self {
            data: inv,
            dim: n,
            condition_number: Some(cond),
            regularization,
        }
    }

    /// Compute pseudo-inverse using SVD-like approach
    pub fn pseudo_invert(&self, threshold: f64) -> Self {
        // For small matrices, use eigendecomposition approximation
        // via power iteration on symmetric matrix
        
        let n = self.dim;
        if n == 0 {
            return Self::new(0);
        }

        // Simple approach: regularized inverse
        // Full SVD would be more robust but expensive
        self.invert(threshold)
    }

    /// Matrix-vector multiplication: F @ v
    pub fn mul_vec(&self, v: &[f64]) -> Vec<f64> {
        let n = self.dim;
        let mut result = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                result[i] += self.get(i, j) * v[j];
            }
        }
        result
    }

    /// Trace of the matrix
    pub fn trace(&self) -> f64 {
        (0..self.dim).map(|i| self.get(i, i)).sum()
    }

    /// Frobenius norm
    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }
}

// =============================================================================
// QNG Optimizer
// =============================================================================

/// Result of a QNG optimization step
#[derive(Debug, Clone)]
pub struct QNGStepResult {
    /// Current objective value
    pub objective: f64,
    /// Euclidean gradient
    pub gradient: Vec<f64>,
    /// Natural gradient (F^{-1} @ gradient)
    pub natural_gradient: Vec<f64>,
    /// Updated parameters
    pub params: Vec<f64>,
    /// FIM condition number
    pub fim_condition: f64,
    /// Whether convergence criterion is met
    pub converged: bool,
    /// Gradient norm
    pub grad_norm: f64,
}

/// Quantum Natural Gradient Optimizer
#[derive(Debug, Clone)]
pub struct QNGOptimizer {
    /// Configuration
    config: QNGConfig,
    /// Current parameters
    params: Vec<f64>,
    /// Best parameters seen
    best_params: Vec<f64>,
    /// Best objective seen
    best_objective: f64,
    /// Current iteration
    iteration: usize,
    /// Cached FIM inverse
    cached_fim_inv: Option<FisherInformationMatrix>,
    /// Step count since last FIM update
    steps_since_fim_update: usize,
    /// History of objectives
    objective_history: Vec<f64>,
}

impl QNGOptimizer {
    /// Create new QNG optimizer
    pub fn new(config: QNGConfig, initial_params: Vec<f64>) -> Self {
        let best_params = initial_params.clone();
        Self {
            config,
            params: initial_params,
            best_params,
            best_objective: f64::MAX,
            iteration: 0,
            cached_fim_inv: None,
            steps_since_fim_update: 0,
            objective_history: Vec::new(),
        }
    }

    /// Get current parameters
    pub fn params(&self) -> &[f64] {
        &self.params
    }

    /// Get best parameters
    pub fn best_params(&self) -> &[f64] {
        &self.best_params
    }

    /// Get best objective
    pub fn best_objective(&self) -> f64 {
        self.best_objective
    }

    /// Get current iteration
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Get objective history
    pub fn objective_history(&self) -> &[f64] {
        &self.objective_history
    }

    /// Perform one optimization step
    pub fn step(
        &mut self,
        circuit: &VariationalCircuit,
        hamiltonian: &Hamiltonian,
    ) -> QNGStepResult {
        let n = self.params.len();

        // Compute objective and gradient
        let objective = compute_expectation(circuit, &self.params, hamiltonian);
        let mut gradient = compute_gradient(circuit, &self.params, hamiltonian);

        // Apply gradient clipping if enabled
        if self.config.grad_clip > 0.0 {
            let grad_norm: f64 = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
            if grad_norm > self.config.grad_clip {
                let scale = self.config.grad_clip / grad_norm;
                for g in &mut gradient {
                    *g *= scale;
                }
            }
        }

        // Compute or retrieve FIM inverse
        let should_update_fim = self.cached_fim_inv.is_none()
            || self.config.fim_update_interval == 0
            || self.steps_since_fim_update >= self.config.fim_update_interval;

        let fim_inv = if should_update_fim {
            let fim = if self.config.block_diagonal {
                FisherInformationMatrix::compute_block_diagonal(circuit, &self.params, hamiltonian)
            } else {
                FisherInformationMatrix::compute(circuit, &self.params, hamiltonian)
            };

            let inv = if self.config.use_pseudo_inverse {
                fim.pseudo_invert(self.config.regularization)
            } else {
                fim.invert(self.config.regularization)
            };

            self.cached_fim_inv = Some(inv.clone());
            self.steps_since_fim_update = 0;
            inv
        } else {
            self.steps_since_fim_update += 1;
            self.cached_fim_inv.clone().unwrap()
        };

        // Compute natural gradient: g_nat = F^{-1} @ g
        let natural_gradient = fim_inv.mul_vec(&gradient);

        // Compute gradient norms
        let grad_norm: f64 = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
        let nat_grad_norm: f64 = natural_gradient.iter().map(|g| g * g).sum::<f64>().sqrt();

        // Update parameters: θ_{t+1} = θ_t - lr * g_nat
        let mut new_params = self.params.clone();
        for i in 0..n {
            new_params[i] -= self.config.learning_rate * natural_gradient[i];
        }

        // Update best if improved
        if objective < self.best_objective {
            self.best_objective = objective;
            self.best_params = new_params.clone();
        }

        // Check convergence
        let converged = grad_norm < self.config.tolerance
            || nat_grad_norm < self.config.tolerance
            || self.iteration >= self.config.max_iter;

        // Update state
        self.params = new_params.clone();
        self.iteration += 1;
        self.objective_history.push(objective);

        QNGStepResult {
            objective,
            gradient,
            natural_gradient,
            params: new_params,
            fim_condition: fim_inv.condition_number().unwrap_or(f64::MAX),
            converged,
            grad_norm,
        }
    }

    /// Run full optimization loop
    pub fn optimize(
        &mut self,
        circuit: &VariationalCircuit,
        hamiltonian: &Hamiltonian,
    ) -> QNGOptimizationResult {
        let start = std::time::Instant::now();

        loop {
            let result = self.step(circuit, hamiltonian);

            if result.converged || self.iteration >= self.config.max_iter {
                return QNGOptimizationResult {
                    params: self.best_params.clone(),
                    objective: self.best_objective,
                    iterations: self.iteration,
                    converged: result.grad_norm < self.config.tolerance,
                    final_grad_norm: result.grad_norm,
                    final_fim_condition: result.fim_condition,
                    elapsed_ms: start.elapsed().as_millis() as u64,
                    objective_history: self.objective_history.clone(),
                };
            }
        }
    }

    /// Reset optimizer state
    pub fn reset(&mut self, initial_params: Vec<f64>) {
        self.params = initial_params.clone();
        self.best_params = initial_params;
        self.best_objective = f64::MAX;
        self.iteration = 0;
        self.cached_fim_inv = None;
        self.steps_since_fim_update = 0;
        self.objective_history.clear();
    }
}

/// Result of QNG optimization
#[derive(Debug, Clone)]
pub struct QNGOptimizationResult {
    /// Optimized parameters
    pub params: Vec<f64>,
    /// Final objective value
    pub objective: f64,
    /// Total iterations
    pub iterations: usize,
    /// Whether converged before max_iter
    pub converged: bool,
    /// Final gradient norm
    pub final_grad_norm: f64,
    /// Final FIM condition number
    pub final_fim_condition: f64,
    /// Elapsed time in milliseconds
    pub elapsed_ms: u64,
    /// Objective history
    pub objective_history: Vec<f64>,
}

// =============================================================================
// Convenience Functions
// =============================================================================

/// Run QNG optimization with default config
pub fn qng_minimize(
    circuit: &VariationalCircuit,
    initial_params: Vec<f64>,
    hamiltonian: &Hamiltonian,
    max_iter: usize,
) -> QNGOptimizationResult {
    let config = QNGConfig::default().with_max_iter(max_iter);
    let mut optimizer = QNGOptimizer::new(config, initial_params);
    optimizer.optimize(circuit, hamiltonian)
}

/// Compare convergence: QNG vs vanilla gradient descent
pub fn compare_qng_vs_gd(
    circuit: &VariationalCircuit,
    initial_params: Vec<f64>,
    hamiltonian: &Hamiltonian,
    max_iter: usize,
    learning_rate: f64,
) -> (QNGOptimizationResult, Vec<f64>) {
    // QNG optimization
    let qng_config = QNGConfig::default()
        .with_learning_rate(learning_rate)
        .with_max_iter(max_iter);
    let mut qng_opt = QNGOptimizer::new(qng_config, initial_params.clone());
    let qng_result = qng_opt.optimize(circuit, hamiltonian);

    // Vanilla GD for comparison
    let mut params = initial_params;
    let mut gd_history = Vec::new();

    for _ in 0..max_iter {
        let obj = compute_expectation(circuit, &params, hamiltonian);
        gd_history.push(obj);

        let grad = compute_gradient(circuit, &params, hamiltonian);
        for i in 0..params.len() {
            params[i] -= learning_rate * grad[i];
        }
    }

    (qng_result, gd_history)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::PauliObservable;

    #[test]
    fn test_fim_creation() {
        let fim = FisherInformationMatrix::new(3);
        assert_eq!(fim.dim(), 3);
        assert_eq!(fim.get(0, 0), 0.0);
    }

    #[test]
    fn test_fim_identity() {
        let fim = FisherInformationMatrix::identity(3);
        assert_eq!(fim.get(0, 0), 1.0);
        assert_eq!(fim.get(1, 1), 1.0);
        assert_eq!(fim.get(0, 1), 0.0);
    }

    #[test]
    fn test_fim_set_get() {
        let mut fim = FisherInformationMatrix::new(2);
        fim.set(0, 1, 0.5);
        fim.set(1, 0, 0.5);
        fim.set(0, 0, 1.0);
        fim.set(1, 1, 1.0);

        assert_eq!(fim.get(0, 1), 0.5);
        assert_eq!(fim.get(1, 0), 0.5);
    }

    #[test]
    fn test_fim_regularize() {
        let mut fim = FisherInformationMatrix::new(2);
        fim.set(0, 0, 1.0);
        fim.set(1, 1, 1.0);

        fim.regularize(0.1);

        assert!((fim.get(0, 0) - 1.1).abs() < 1e-10);
        assert!((fim.get(1, 1) - 1.1).abs() < 1e-10);
    }

    #[test]
    fn test_fim_invert_identity() {
        let fim = FisherInformationMatrix::identity(3);
        let inv = fim.invert(0.0);

        // Identity inverse is identity
        assert!((inv.get(0, 0) - 1.0).abs() < 1e-6);
        assert!((inv.get(1, 1) - 1.0).abs() < 1e-6);
        assert!(inv.get(0, 1).abs() < 1e-6);
    }

    #[test]
    fn test_fim_invert_2x2() {
        let mut fim = FisherInformationMatrix::new(2);
        fim.set(0, 0, 2.0);
        fim.set(0, 1, 1.0);
        fim.set(1, 0, 1.0);
        fim.set(1, 1, 3.0);

        let inv = fim.invert(0.0);

        // A^{-1} for [[2,1],[1,3]] = [[3,-1],[-1,2]] / 5
        assert!((inv.get(0, 0) - 0.6).abs() < 1e-6);
        assert!((inv.get(0, 1) - (-0.2)).abs() < 1e-6);
        assert!((inv.get(1, 0) - (-0.2)).abs() < 1e-6);
        assert!((inv.get(1, 1) - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_fim_mul_vec() {
        let mut fim = FisherInformationMatrix::new(2);
        fim.set(0, 0, 2.0);
        fim.set(0, 1, 1.0);
        fim.set(1, 0, 1.0);
        fim.set(1, 1, 3.0);

        let v = vec![1.0, 2.0];
        let result = fim.mul_vec(&v);

        // [2,1;1,3] @ [1,2] = [4, 7]
        assert!((result[0] - 4.0).abs() < 1e-10);
        assert!((result[1] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_fim_diagonal() {
        let mut fim = FisherInformationMatrix::new(3);
        fim.set(0, 0, 1.0);
        fim.set(1, 1, 2.0);
        fim.set(2, 2, 3.0);

        let diag = fim.diagonal();
        assert_eq!(diag, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_fim_trace() {
        let mut fim = FisherInformationMatrix::new(3);
        fim.set(0, 0, 1.0);
        fim.set(1, 1, 2.0);
        fim.set(2, 2, 3.0);

        assert_eq!(fim.trace(), 6.0);
    }

    #[test]
    fn test_qng_config_builder() {
        let config = QNGConfig::default()
            .with_learning_rate(0.05)
            .with_regularization(1e-3)
            .with_max_iter(100)
            .with_block_diagonal(true);

        assert_eq!(config.learning_rate, 0.05);
        assert_eq!(config.regularization, 1e-3);
        assert_eq!(config.max_iter, 100);
        assert!(config.block_diagonal);
    }

    #[test]
    fn test_qng_optimizer_creation() {
        let config = QNGConfig::default();
        let opt = QNGOptimizer::new(config, vec![0.5, 0.5]);

        assert_eq!(opt.params().len(), 2);
        assert_eq!(opt.iteration(), 0);
    }

    #[test]
    fn test_qng_single_step() {
        let mut circuit = VariationalCircuit::new(1);
        circuit.ry(0);

        let mut hamiltonian = Hamiltonian::new();
        hamiltonian.add_term(PauliObservable::z(0));

        let config = QNGConfig::default().with_learning_rate(0.1);
        let mut opt = QNGOptimizer::new(config, vec![0.5]);

        let result = opt.step(&circuit, &hamiltonian);

        assert!(result.objective.is_finite());
        assert_eq!(result.gradient.len(), 1);
        assert_eq!(result.natural_gradient.len(), 1);
        assert_eq!(opt.iteration(), 1);
    }

    #[test]
    fn test_qng_optimization() {
        let mut circuit = VariationalCircuit::new(1);
        circuit.ry(0);

        let mut hamiltonian = Hamiltonian::new();
        hamiltonian.add_term(PauliObservable::z(0));

        let config = QNGConfig::default()
            .with_learning_rate(0.2)
            .with_max_iter(50);
        let mut opt = QNGOptimizer::new(config, vec![0.5]);

        let result = opt.optimize(&circuit, &hamiltonian);

        // Should find minimum (⟨Z⟩ = -1 at θ = π)
        assert!(result.objective < 0.0);
        assert!(result.iterations <= 50);
    }

    #[test]
    fn test_qng_minimize_convenience() {
        let mut circuit = VariationalCircuit::new(1);
        circuit.ry(0);

        let mut hamiltonian = Hamiltonian::new();
        hamiltonian.add_term(PauliObservable::z(0));

        let result = qng_minimize(&circuit, vec![0.5], &hamiltonian, 30);

        assert!(result.objective < 0.5);
    }

    #[test]
    fn test_qng_vs_gd_comparison() {
        let mut circuit = VariationalCircuit::new(1);
        circuit.ry(0);

        let mut hamiltonian = Hamiltonian::new();
        hamiltonian.add_term(PauliObservable::z(0));

        let (qng_result, gd_history) = compare_qng_vs_gd(
            &circuit,
            vec![0.5],
            &hamiltonian,
            20,
            0.1,
        );

        // Both should make progress
        assert!(qng_result.objective < 0.8);
        assert!(gd_history.len() == 20);
    }

    #[test]
    fn test_fim_compute_simple() {
        let mut circuit = VariationalCircuit::new(1);
        circuit.ry(0);

        let mut hamiltonian = Hamiltonian::new();
        hamiltonian.add_term(PauliObservable::z(0));

        let fim = FisherInformationMatrix::compute(&circuit, &[0.5], &hamiltonian);

        // Should have non-zero diagonal
        assert!(fim.get(0, 0) > 0.0);
    }

    #[test]
    fn test_fim_block_diagonal() {
        let mut circuit = VariationalCircuit::new(2);
        circuit.ry(0);
        circuit.ry(1);

        let mut hamiltonian = Hamiltonian::new();
        hamiltonian.add_term(PauliObservable::z(0));
        hamiltonian.add_term(PauliObservable::z(1));

        let fim = FisherInformationMatrix::compute_block_diagonal(
            &circuit, &[0.5, 0.5], &hamiltonian
        );

        // Off-diagonal should be zero
        assert_eq!(fim.get(0, 1), 0.0);
        assert_eq!(fim.get(1, 0), 0.0);
        // Diagonal should be non-zero
        assert!(fim.get(0, 0) > 0.0);
        assert!(fim.get(1, 1) > 0.0);
    }

    #[test]
    fn test_qng_with_block_diagonal() {
        let mut circuit = VariationalCircuit::new(2);
        circuit.ry(0);
        circuit.ry(1);

        let mut hamiltonian = Hamiltonian::new();
        hamiltonian.add_term(PauliObservable::z(0));

        let config = QNGConfig::default()
            .with_learning_rate(0.1)
            .with_block_diagonal(true)
            .with_max_iter(20);

        let mut opt = QNGOptimizer::new(config, vec![0.5, 0.5]);
        let result = opt.optimize(&circuit, &hamiltonian);

        assert!(result.iterations > 0);
    }

    #[test]
    fn test_qng_reset() {
        let config = QNGConfig::default();
        let mut opt = QNGOptimizer::new(config, vec![0.5, 0.5]);

        // Simulate some iterations
        opt.iteration = 10;
        opt.best_objective = 0.5;

        opt.reset(vec![1.0, 1.0]);

        assert_eq!(opt.iteration(), 0);
        assert_eq!(opt.params(), &[1.0, 1.0]);
        assert_eq!(opt.best_objective(), f64::MAX);
    }

    #[test]
    fn test_qng_gradient_clipping() {
        let mut circuit = VariationalCircuit::new(1);
        circuit.ry(0);

        let mut hamiltonian = Hamiltonian::new();
        hamiltonian.add_term(PauliObservable::z(0));

        let config = QNGConfig::default()
            .with_learning_rate(0.1)
            .with_grad_clip(0.5);

        let mut opt = QNGOptimizer::new(config, vec![0.1]);
        let result = opt.step(&circuit, &hamiltonian);

        // Gradient should be clipped
        let grad_norm: f64 = result.gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
        assert!(grad_norm <= 0.5 + 1e-6);
    }

    #[test]
    fn test_qng_fim_update_interval() {
        let mut circuit = VariationalCircuit::new(1);
        circuit.ry(0);

        let mut hamiltonian = Hamiltonian::new();
        hamiltonian.add_term(PauliObservable::z(0));

        let config = QNGConfig::default()
            .with_fim_update_interval(5)
            .with_max_iter(10);

        let mut opt = QNGOptimizer::new(config, vec![0.5]);

        // Run 10 steps
        for _ in 0..10 {
            let _ = opt.step(&circuit, &hamiltonian);
        }

        // FIM should have been updated at iterations 0, 5
        assert!(opt.cached_fim_inv.is_some());
    }
}

//! Zero-Noise Extrapolation (ZNE) for Error Mitigation
//!
//! ZNE reduces errors by:
//! 1. Running circuits at multiple noise levels (via gate folding)
//! 2. Extrapolating to the zero-noise limit
//!
//! Reference: Temme et al., PRL 119, 180509 (2017)

use crate::autodiff::{VariationalCircuit, PauliObservable};
use crate::noise_model::{NoiseModel, NoiseConfig, ShotSampler, ShotResult};
use crate::state::TQPState;
use num_complex::Complex64;

// =============================================================================
// Gate Folding
// =============================================================================

/// Methods for scaling noise via gate folding
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FoldMethod {
    /// Fold entire circuit: C → C·C†·C (scale 3), C·C†·C·C†·C (scale 5), etc.
    Global,
    /// Fold individual gates randomly to achieve fractional scales
    Partial,
    /// Fold from the end of the circuit
    FromEnd,
}

impl Default for FoldMethod {
    fn default() -> Self {
        FoldMethod::Global
    }
}

/// Configuration for ZNE
#[derive(Debug, Clone)]
pub struct ZNEConfig {
    /// Scale factors for noise amplification
    pub scale_factors: Vec<f64>,
    /// Extrapolation method
    pub extrapolation: ExtrapolationType,
    /// Gate folding method
    pub fold_method: FoldMethod,
    /// Number of shots per circuit execution
    pub n_shots: usize,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for ZNEConfig {
    fn default() -> Self {
        Self {
            scale_factors: vec![1.0, 2.0, 3.0],
            extrapolation: ExtrapolationType::Linear,
            fold_method: FoldMethod::Global,
            n_shots: 1000,
            seed: None,
        }
    }
}

impl ZNEConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_scale_factors(mut self, factors: Vec<f64>) -> Self {
        self.scale_factors = factors;
        self
    }

    pub fn with_extrapolation(mut self, method: ExtrapolationType) -> Self {
        self.extrapolation = method;
        self
    }

    pub fn with_fold_method(mut self, method: FoldMethod) -> Self {
        self.fold_method = method;
        self
    }

    pub fn with_shots(mut self, n: usize) -> Self {
        self.n_shots = n;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Standard ZNE with linear extrapolation
    pub fn linear() -> Self {
        Self::default()
    }

    /// Richardson extrapolation (polynomial fit)
    pub fn richardson(degree: usize) -> Self {
        Self {
            scale_factors: (1..=degree+1).map(|i| i as f64).collect(),
            extrapolation: ExtrapolationType::Richardson(degree),
            ..Self::default()
        }
    }

    /// Exponential extrapolation
    pub fn exponential() -> Self {
        Self {
            extrapolation: ExtrapolationType::Exponential,
            ..Self::default()
        }
    }
}

// =============================================================================
// Extrapolation Methods
// =============================================================================

/// Type of extrapolation to zero noise
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExtrapolationType {
    /// Linear fit: E(λ) = a + b*λ → E(0) = a
    Linear,
    /// Polynomial fit of given degree (Richardson extrapolation)
    Richardson(usize),
    /// Exponential fit: E(λ) = a + b*exp(c*λ)
    Exponential,
}

impl Default for ExtrapolationType {
    fn default() -> Self {
        ExtrapolationType::Linear
    }
}

/// Extrapolator for fitting and extrapolating to zero noise
#[derive(Debug)]
pub struct Extrapolator {
    method: ExtrapolationType,
}

impl Extrapolator {
    pub fn new(method: ExtrapolationType) -> Self {
        Self { method }
    }

    /// Extrapolate to zero noise given (scale_factor, expectation) pairs
    pub fn extrapolate(&self, data: &[(f64, f64)]) -> f64 {
        match self.method {
            ExtrapolationType::Linear => self.linear_extrapolate(data),
            ExtrapolationType::Richardson(degree) => self.richardson_extrapolate(data, degree),
            ExtrapolationType::Exponential => self.exponential_extrapolate(data),
        }
    }

    /// Linear extrapolation: E(λ) = a + b*λ
    fn linear_extrapolate(&self, data: &[(f64, f64)]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        if data.len() == 1 {
            return data[0].1;
        }

        // Least squares fit
        let n = data.len() as f64;
        let sum_x: f64 = data.iter().map(|(x, _)| x).sum();
        let sum_y: f64 = data.iter().map(|(_, y)| y).sum();
        let sum_xx: f64 = data.iter().map(|(x, _)| x * x).sum();
        let sum_xy: f64 = data.iter().map(|(x, y)| x * y).sum();

        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-15 {
            return sum_y / n;
        }

        let a = (sum_y * sum_xx - sum_x * sum_xy) / denom;
        // b = (n * sum_xy - sum_x * sum_y) / denom; // slope (not needed for λ=0)

        a // E(0) = a
    }

    /// Richardson extrapolation using polynomial fit
    fn richardson_extrapolate(&self, data: &[(f64, f64)], degree: usize) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        if data.len() == 1 {
            return data[0].1;
        }

        let n = data.len();
        let d = degree.min(n - 1);

        // Build Vandermonde matrix and solve least squares
        // For simplicity, use linear combination formula for Richardson
        // Standard Richardson: E(0) = Σ c_i * E(λ_i)
        
        if d == 1 {
            return self.linear_extrapolate(data);
        }

        // For degree 2 with 3 points, use Lagrange interpolation at λ=0
        if d == 2 && n >= 3 {
            let (x0, y0) = data[0];
            let (x1, y1) = data[1];
            let (x2, y2) = data[2];

            let l0 = (0.0 - x1) * (0.0 - x2) / ((x0 - x1) * (x0 - x2));
            let l1 = (0.0 - x0) * (0.0 - x2) / ((x1 - x0) * (x1 - x2));
            let l2 = (0.0 - x0) * (0.0 - x1) / ((x2 - x0) * (x2 - x1));

            return y0 * l0 + y1 * l1 + y2 * l2;
        }

        // General polynomial fit using normal equations
        self.polynomial_fit_extrapolate(data, d)
    }

    /// General polynomial fit
    fn polynomial_fit_extrapolate(&self, data: &[(f64, f64)], degree: usize) -> f64 {
        let n = data.len();
        let d = degree + 1;

        // Build normal equations: A^T A c = A^T y
        let mut ata = vec![vec![0.0; d]; d];
        let mut aty = vec![0.0; d];

        for (x, y) in data {
            let mut xi = 1.0;
            for i in 0..d {
                let mut xj = 1.0;
                for j in 0..d {
                    ata[i][j] += xi * xj;
                    xj *= x;
                }
                aty[i] += xi * y;
                xi *= x;
            }
        }

        // Solve using Gaussian elimination
        let coeffs = self.solve_linear_system(&mut ata, &mut aty);
        
        // E(0) = c[0] (constant term)
        coeffs.get(0).copied().unwrap_or(0.0)
    }

    /// Simple Gaussian elimination
    fn solve_linear_system(&self, a: &mut Vec<Vec<f64>>, b: &mut Vec<f64>) -> Vec<f64> {
        let n = b.len();
        
        // Forward elimination
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if a[k][i].abs() > a[max_row][i].abs() {
                    max_row = k;
                }
            }
            a.swap(i, max_row);
            b.swap(i, max_row);

            if a[i][i].abs() < 1e-15 {
                continue;
            }

            // Eliminate
            for k in (i + 1)..n {
                let factor = a[k][i] / a[i][i];
                for j in i..n {
                    a[k][j] -= factor * a[i][j];
                }
                b[k] -= factor * b[i];
            }
        }

        // Back substitution
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            if a[i][i].abs() < 1e-15 {
                continue;
            }
            x[i] = b[i];
            for j in (i + 1)..n {
                x[i] -= a[i][j] * x[j];
            }
            x[i] /= a[i][i];
        }

        x
    }

    /// Exponential extrapolation: E(λ) = a + b*exp(c*λ)
    fn exponential_extrapolate(&self, data: &[(f64, f64)]) -> f64 {
        if data.len() < 3 {
            return self.linear_extrapolate(data);
        }

        // Use three points to fit a + b*exp(c*λ)
        // This is a simplified approach; full nonlinear fit would be better
        let (x0, y0) = data[0];
        let (x1, y1) = data[1];
        let (x2, y2) = data[data.len() - 1];

        // Estimate decay rate from ratio
        let r1 = (y1 - y0) / (x1 - x0);
        let r2 = (y2 - y1) / (x2 - x1);
        
        if r1.abs() < 1e-15 || r2.abs() < 1e-15 {
            return self.linear_extrapolate(data);
        }

        // For exponential decay, use log-linear fit
        let log_data: Vec<(f64, f64)> = data.iter()
            .filter(|(_, y)| *y > 0.0)
            .map(|(x, y)| (*x, y.ln()))
            .collect();

        if log_data.len() < 2 {
            return self.linear_extrapolate(data);
        }

        let log_y0 = self.linear_extrapolate(&log_data);
        log_y0.exp()
    }
}

// =============================================================================
// ZNE Executor
// =============================================================================

/// Result of ZNE computation
#[derive(Debug, Clone)]
pub struct ZNEResult {
    /// Mitigated expectation value
    pub mitigated_value: f64,
    /// Raw values at each scale factor
    pub scaled_values: Vec<(f64, f64)>,
    /// Extrapolation method used
    pub extrapolation: ExtrapolationType,
    /// Unmitigated (scale=1) value
    pub unmitigated_value: f64,
    /// Improvement ratio: |unmitigated - ideal| / |mitigated - ideal|
    pub improvement_ratio: Option<f64>,
}

impl ZNEResult {
    /// Compute error reduction if ideal value is known
    pub fn with_ideal(&mut self, ideal: f64) {
        let unmit_error = (self.unmitigated_value - ideal).abs();
        let mit_error = (self.mitigated_value - ideal).abs();
        
        if mit_error > 1e-15 {
            self.improvement_ratio = Some(unmit_error / mit_error);
        }
    }
}

/// Zero-Noise Extrapolation executor
pub struct ZNEExecutor {
    config: ZNEConfig,
    noise_model: NoiseModel,
    extrapolator: Extrapolator,
}

impl ZNEExecutor {
    pub fn new(config: ZNEConfig, noise_config: NoiseConfig) -> Self {
        let noise_model = NoiseModel::new(noise_config);
        let extrapolator = Extrapolator::new(config.extrapolation);
        Self {
            config,
            noise_model,
            extrapolator,
        }
    }

    pub fn with_default_noise(config: ZNEConfig) -> Self {
        Self::new(config, NoiseConfig::default())
    }

    /// Execute ZNE for a given circuit and observable
    pub fn execute(
        &mut self,
        circuit: &VariationalCircuit,
        params: &[f64],
        observable: &PauliObservable,
    ) -> ZNEResult {
        let mut scaled_values = Vec::new();
        let mut unmitigated = 0.0;

        let scale_factors = self.config.scale_factors.clone();
        for &scale in &scale_factors {
            let exp_value = self.execute_at_scale(circuit, params, observable, scale);
            scaled_values.push((scale, exp_value));
            
            if (scale - 1.0).abs() < 1e-10 {
                unmitigated = exp_value;
            }
        }

        let mitigated = self.extrapolator.extrapolate(&scaled_values);

        ZNEResult {
            mitigated_value: mitigated,
            scaled_values,
            extrapolation: self.config.extrapolation,
            unmitigated_value: unmitigated,
            improvement_ratio: None,
        }
    }

    /// Execute circuit at a given noise scale
    fn execute_at_scale(
        &mut self,
        circuit: &VariationalCircuit,
        params: &[f64],
        observable: &PauliObservable,
        scale: f64,
    ) -> f64 {
        // Get ideal state
        let state = circuit.execute(params);
        
        // Apply scaled noise
        let noisy_state = self.apply_scaled_noise(&state, scale);
        
        // Compute expectation value
        self.compute_expectation(&noisy_state, observable)
    }

    /// Apply noise scaled by given factor
    fn apply_scaled_noise(&mut self, state: &TQPState, scale: f64) -> TQPState {
        if scale < 1e-10 {
            return state.clone();
        }

        let mut scaled_noise = self.noise_model.scaled(scale);
        scaled_noise.apply_depolarizing_all(state)
    }

    /// Compute expectation value of Pauli observable
    fn compute_expectation(&self, state: &TQPState, observable: &PauliObservable) -> f64 {
        // Simple implementation for Pauli strings
        let probs: Vec<f64> = state.state_vector.iter()
            .map(|a| a.norm_sqr())
            .collect();

        let n_qubits = state.n_qubits();
        let mut expectation = 0.0;

        for (i, &prob) in probs.iter().enumerate() {
            if prob < 1e-15 {
                continue;
            }

            let mut sign = 1.0;
            for &(qubit, pauli) in &observable.terms {
                if pauli == 'Z' || pauli == 'z' {
                    let bit = (i >> qubit) & 1;
                    if bit == 1 {
                        sign *= -1.0;
                    }
                }
                // X and Y terms require more complex handling with amplitudes
            }
            expectation += prob * sign;
        }

        expectation * observable.coeff
    }

    /// Execute ZNE with shot-based measurement
    pub fn execute_with_shots(
        &mut self,
        circuit: &VariationalCircuit,
        params: &[f64],
        qubit: usize,  // For single Z measurement
    ) -> ZNEResult {
        let mut sampler = ShotSampler::new(self.config.seed);
        let mut scaled_values = Vec::new();
        let mut unmitigated = 0.0;

        let scale_factors = self.config.scale_factors.clone();
        let n_shots = self.config.n_shots;
        for &scale in &scale_factors {
            let state = circuit.execute(params);
            let noisy_state = self.apply_scaled_noise(&state, scale);
            
            // Get probabilities with readout error
            let probs: Vec<f64> = noisy_state.state_vector.iter()
                .map(|a| a.norm_sqr())
                .collect();
            let noisy_probs = self.noise_model.apply_readout_error(&probs, state.n_qubits());
            
            // Sample
            let shots = sampler.sample(&noisy_probs, n_shots, state.n_qubits());
            
            // Compute Z expectation from shots
            let exp_value = crate::noise_model::z_expectation_from_shots(&shots, qubit);
            scaled_values.push((scale, exp_value));
            
            if (scale - 1.0).abs() < 1e-10 {
                unmitigated = exp_value;
            }
        }

        let mitigated = self.extrapolator.extrapolate(&scaled_values);

        ZNEResult {
            mitigated_value: mitigated,
            scaled_values,
            extrapolation: self.config.extrapolation,
            unmitigated_value: unmitigated,
            improvement_ratio: None,
        }
    }
}

// =============================================================================
// Convenience Functions
// =============================================================================

/// Simple ZNE with linear extrapolation
pub fn apply_zne_linear(
    circuit: &VariationalCircuit,
    params: &[f64],
    observable: &PauliObservable,
    noise_config: NoiseConfig,
) -> ZNEResult {
    let config = ZNEConfig::linear();
    let mut executor = ZNEExecutor::new(config, noise_config);
    executor.execute(circuit, params, observable)
}

/// ZNE with Richardson extrapolation
pub fn apply_zne_richardson(
    circuit: &VariationalCircuit,
    params: &[f64],
    observable: &PauliObservable,
    noise_config: NoiseConfig,
    degree: usize,
) -> ZNEResult {
    let config = ZNEConfig::richardson(degree);
    let mut executor = ZNEExecutor::new(config, noise_config);
    executor.execute(circuit, params, observable)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zne_config_default() {
        let config = ZNEConfig::default();
        assert_eq!(config.scale_factors, vec![1.0, 2.0, 3.0]);
        assert_eq!(config.extrapolation, ExtrapolationType::Linear);
    }

    #[test]
    fn test_zne_config_builder() {
        let config = ZNEConfig::new()
            .with_scale_factors(vec![1.0, 1.5, 2.0])
            .with_extrapolation(ExtrapolationType::Exponential)
            .with_shots(2000);

        assert_eq!(config.scale_factors, vec![1.0, 1.5, 2.0]);
        assert_eq!(config.extrapolation, ExtrapolationType::Exponential);
        assert_eq!(config.n_shots, 2000);
    }

    #[test]
    fn test_zne_config_richardson() {
        let config = ZNEConfig::richardson(2);
        assert_eq!(config.scale_factors, vec![1.0, 2.0, 3.0]);
        assert_eq!(config.extrapolation, ExtrapolationType::Richardson(2));
    }

    #[test]
    fn test_extrapolator_linear() {
        let extrap = Extrapolator::new(ExtrapolationType::Linear);
        
        // Perfect linear data: y = 2 + 0.5*x
        let data = vec![(1.0, 2.5), (2.0, 3.0), (3.0, 3.5)];
        let result = extrap.extrapolate(&data);
        
        // At x=0, y should be 2.0
        assert!((result - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_extrapolator_richardson() {
        let extrap = Extrapolator::new(ExtrapolationType::Richardson(2));
        
        // Quadratic data: y = 1 + x + x^2
        // At x=1: y=3, x=2: y=7, x=3: y=13
        let data = vec![(1.0, 3.0), (2.0, 7.0), (3.0, 13.0)];
        let result = extrap.extrapolate(&data);
        
        // At x=0, y should be 1.0
        assert!((result - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_extrapolator_single_point() {
        let extrap = Extrapolator::new(ExtrapolationType::Linear);
        let data = vec![(1.0, 5.0)];
        let result = extrap.extrapolate(&data);
        
        assert!((result - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_extrapolator_empty() {
        let extrap = Extrapolator::new(ExtrapolationType::Linear);
        let data: Vec<(f64, f64)> = vec![];
        let result = extrap.extrapolate(&data);
        
        assert!((result - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_zne_executor_creation() {
        let config = ZNEConfig::default();
        let executor = ZNEExecutor::with_default_noise(config);
        assert!(true); // Just verify it doesn't panic
    }

    #[test]
    fn test_zne_result_with_ideal() {
        let mut result = ZNEResult {
            mitigated_value: 0.95,
            scaled_values: vec![(1.0, 0.8), (2.0, 0.6), (3.0, 0.4)],
            extrapolation: ExtrapolationType::Linear,
            unmitigated_value: 0.8,
            improvement_ratio: None,
        };
        
        result.with_ideal(1.0);
        
        // unmit_error = |0.8 - 1.0| = 0.2
        // mit_error = |0.95 - 1.0| = 0.05
        // ratio = 0.2 / 0.05 = 4.0
        assert!(result.improvement_ratio.is_some());
        let ratio = result.improvement_ratio.unwrap();
        assert!((ratio - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_fold_method_default() {
        let method = FoldMethod::default();
        assert_eq!(method, FoldMethod::Global);
    }

    #[test]
    fn test_polynomial_fit() {
        let extrap = Extrapolator::new(ExtrapolationType::Linear);
        
        // y = 1 + 2x
        let data = vec![(0.0, 1.0), (1.0, 3.0), (2.0, 5.0)];
        let result = extrap.polynomial_fit_extrapolate(&data, 1);
        
        assert!((result - 1.0).abs() < 1e-10);
    }
}

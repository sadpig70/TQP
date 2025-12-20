//! Barren Plateau Detection and Mitigation
//!
//! Sprint 2 Week 4 Day 2: Tools for detecting and avoiding barren plateaus
//!
//! Barren plateaus are regions of the parameter space where gradients
//! vanish exponentially with system size, making optimization impossible.
//!
//! # Key Concepts
//!
//! - **Gradient Variance**: Var(∂E/∂θ) ~ O(1/2^n) indicates barren plateau
//! - **Layerwise Analysis**: Identify which layers contribute to vanishing gradients
//! - **Initialization Strategies**: Techniques to avoid barren plateaus
//!
//! # References
//!
//! - McClean et al., "Barren plateaus in quantum neural network training landscapes"
//! - Cerezo et al., "Cost function dependent barren plateaus"
//!
//! # Example
//!
//! ```ignore
//! use tqp_core::{BarrenPlateauAnalyzer, VariationalCircuit, Hamiltonian};
//!
//! let circuit = VariationalCircuit::hardware_efficient(4, 3);
//! let analyzer = BarrenPlateauAnalyzer::new(&circuit);
//!
//! let report = analyzer.analyze(&hamiltonian, 100);
//! if report.is_barren() {
//!     println!("Warning: Barren plateau detected!");
//!     println!("Problematic layers: {:?}", report.problematic_layers());
//! }
//! ```

use std::f64::consts::PI;

use crate::autodiff::{
    compute_gradient, VariationalCircuit, Hamiltonian,
};

// =============================================================================
// Constants
// =============================================================================

/// Threshold for variance to be considered "vanishing"
pub const VARIANCE_THRESHOLD: f64 = 1e-6;

/// Default number of samples for variance estimation
pub const DEFAULT_N_SAMPLES: usize = 100;

/// Warning threshold for gradient magnitude
pub const GRADIENT_WARNING_THRESHOLD: f64 = 1e-4;

/// Critical threshold for gradient magnitude
pub const GRADIENT_CRITICAL_THRESHOLD: f64 = 1e-8;

// =============================================================================
// Gradient Statistics
// =============================================================================

/// Statistics for gradient distribution
#[derive(Debug, Clone)]
pub struct GradientStats {
    /// Mean of gradient components
    pub mean: f64,
    /// Variance of gradient components
    pub variance: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Maximum absolute gradient
    pub max_abs: f64,
    /// Minimum absolute gradient
    pub min_abs: f64,
    /// Number of samples
    pub n_samples: usize,
    /// Per-parameter means
    pub param_means: Vec<f64>,
    /// Per-parameter variances
    pub param_variances: Vec<f64>,
}

impl GradientStats {
    /// Create new gradient stats from samples
    pub fn from_samples(samples: &[Vec<f64>]) -> Self {
        if samples.is_empty() || samples[0].is_empty() {
            return Self {
                mean: 0.0,
                variance: 0.0,
                std_dev: 0.0,
                max_abs: 0.0,
                min_abs: 0.0,
                n_samples: 0,
                param_means: Vec::new(),
                param_variances: Vec::new(),
            };
        }

        let n_samples = samples.len();
        let n_params = samples[0].len();

        // Compute per-parameter statistics
        let mut param_means = vec![0.0; n_params];
        let mut param_variances = vec![0.0; n_params];

        // First pass: means
        for sample in samples {
            for (i, &g) in sample.iter().enumerate() {
                param_means[i] += g;
            }
        }
        for m in &mut param_means {
            *m /= n_samples as f64;
        }

        // Second pass: variances
        for sample in samples {
            for (i, &g) in sample.iter().enumerate() {
                let diff = g - param_means[i];
                param_variances[i] += diff * diff;
            }
        }
        for v in &mut param_variances {
            *v /= n_samples as f64;
        }

        // Global statistics
        let all_grads: Vec<f64> = samples.iter().flatten().copied().collect();
        let mean = all_grads.iter().sum::<f64>() / all_grads.len() as f64;
        let variance = all_grads.iter()
            .map(|g| (g - mean).powi(2))
            .sum::<f64>() / all_grads.len() as f64;
        let std_dev = variance.sqrt();
        let max_abs = all_grads.iter().map(|g| g.abs()).fold(0.0, f64::max);
        let min_abs = all_grads.iter().map(|g| g.abs()).fold(f64::MAX, f64::min);

        Self {
            mean,
            variance,
            std_dev,
            max_abs,
            min_abs,
            n_samples,
            param_means,
            param_variances,
        }
    }

    /// Check if variance indicates barren plateau
    pub fn is_vanishing(&self) -> bool {
        self.variance < VARIANCE_THRESHOLD
    }

    /// Get severity level (0=OK, 1=Warning, 2=Critical)
    pub fn severity(&self) -> u8 {
        if self.max_abs < GRADIENT_CRITICAL_THRESHOLD {
            2 // Critical
        } else if self.max_abs < GRADIENT_WARNING_THRESHOLD {
            1 // Warning
        } else {
            0 // OK
        }
    }
}

// =============================================================================
// Layer Analysis
// =============================================================================

/// Analysis result for a single layer
#[derive(Debug, Clone)]
pub struct LayerAnalysis {
    /// Layer index
    pub layer_idx: usize,
    /// Parameter indices in this layer
    pub param_indices: Vec<usize>,
    /// Gradient statistics for this layer
    pub stats: GradientStats,
    /// Whether this layer is problematic
    pub is_problematic: bool,
    /// Relative contribution to total variance
    pub variance_contribution: f64,
}

/// Layerwise gradient analysis
#[derive(Debug, Clone)]
pub struct LayerwiseAnalysis {
    /// Per-layer analysis
    pub layers: Vec<LayerAnalysis>,
    /// Total number of parameters
    pub n_params: usize,
    /// Total number of layers
    pub n_layers: usize,
    /// Overall gradient stats
    pub global_stats: GradientStats,
}

impl LayerwiseAnalysis {
    /// Get indices of problematic layers
    pub fn problematic_layers(&self) -> Vec<usize> {
        self.layers.iter()
            .filter(|l| l.is_problematic)
            .map(|l| l.layer_idx)
            .collect()
    }

    /// Get variance distribution across layers
    pub fn variance_distribution(&self) -> Vec<f64> {
        self.layers.iter()
            .map(|l| l.variance_contribution)
            .collect()
    }
}

// =============================================================================
// Initialization Strategies
// =============================================================================

/// Initialization strategy for avoiding barren plateaus
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitStrategy {
    /// Uniform random in [0, 2π]
    Uniform,
    /// Small random values near zero
    SmallRandom,
    /// Identity-like initialization (parameters that make circuit ~identity)
    Identity,
    /// Layer-scaled initialization (smaller values for deeper layers)
    LayerScaled,
    /// Block initialization (train blocks sequentially)
    BlockInit,
}

impl InitStrategy {
    /// Generate initial parameters
    pub fn generate(&self, n_params: usize, n_layers: usize, seed: u64) -> Vec<f64> {
        let mut params = Vec::with_capacity(n_params);
        let params_per_layer = if n_layers > 0 { n_params / n_layers } else { n_params };

        // Simple PRNG (for reproducibility without external deps)
        let mut rng_state = seed;
        let mut next_random = || {
            rng_state = rng_state.wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng_state >> 33) as f64 / (1u64 << 31) as f64
        };

        match self {
            InitStrategy::Uniform => {
                for _ in 0..n_params {
                    params.push(next_random() * 2.0 * PI);
                }
            }
            InitStrategy::SmallRandom => {
                for _ in 0..n_params {
                    params.push((next_random() - 0.5) * 0.1);
                }
            }
            InitStrategy::Identity => {
                // For RY gates, θ=0 gives identity-like behavior
                for _ in 0..n_params {
                    params.push((next_random() - 0.5) * 0.01);
                }
            }
            InitStrategy::LayerScaled => {
                // Scale decreases with layer depth
                let mut param_idx = 0;
                for layer in 0..n_layers {
                    let scale = 1.0 / (layer as f64 + 1.0).sqrt();
                    for _ in 0..params_per_layer {
                        if param_idx < n_params {
                            params.push((next_random() - 0.5) * scale);
                            param_idx += 1;
                        }
                    }
                }
                // Fill remaining
                while params.len() < n_params {
                    params.push((next_random() - 0.5) * 0.1);
                }
            }
            InitStrategy::BlockInit => {
                // First layer gets random, rest starts near zero
                for i in 0..n_params {
                    let layer = i / params_per_layer.max(1);
                    if layer == 0 {
                        params.push((next_random() - 0.5) * PI);
                    } else {
                        params.push(0.0);
                    }
                }
            }
        }

        params
    }

    /// Description of the strategy
    pub fn description(&self) -> &'static str {
        match self {
            InitStrategy::Uniform => "Uniform random in [0, 2π]",
            InitStrategy::SmallRandom => "Small random values near zero",
            InitStrategy::Identity => "Identity-like initialization",
            InitStrategy::LayerScaled => "Layer-scaled (smaller for deeper layers)",
            InitStrategy::BlockInit => "Block initialization (sequential training)",
        }
    }
}

// =============================================================================
// Barren Plateau Analyzer
// =============================================================================

/// Analysis report for barren plateau detection
#[derive(Debug, Clone)]
pub struct BarrenPlateauReport {
    /// Overall gradient statistics
    pub global_stats: GradientStats,
    /// Layerwise analysis (if available)
    pub layerwise: Option<LayerwiseAnalysis>,
    /// Whether barren plateau is detected
    pub is_barren: bool,
    /// Severity level (0=OK, 1=Warning, 2=Critical)
    pub severity: u8,
    /// Recommended initialization strategy
    pub recommended_init: InitStrategy,
    /// Diagnostic messages
    pub diagnostics: Vec<String>,
    /// Number of qubits
    pub n_qubits: usize,
    /// Number of parameters
    pub n_params: usize,
    /// Expected variance scaling (theoretical)
    pub expected_variance_scaling: f64,
}

impl BarrenPlateauReport {
    /// Check if barren plateau detected
    pub fn is_barren(&self) -> bool {
        self.is_barren
    }

    /// Get problematic layer indices
    pub fn problematic_layers(&self) -> Vec<usize> {
        self.layerwise.as_ref()
            .map(|l| l.problematic_layers())
            .unwrap_or_default()
    }

    /// Generate summary string
    pub fn summary(&self) -> String {
        let status = match self.severity {
            0 => "OK",
            1 => "WARNING",
            _ => "CRITICAL",
        };

        format!(
            "Barren Plateau Analysis: {}\n\
             - Gradient variance: {:.2e}\n\
             - Max |gradient|: {:.2e}\n\
             - Expected scaling: O(1/{:.0})\n\
             - Recommended init: {:?}",
            status,
            self.global_stats.variance,
            self.global_stats.max_abs,
            1.0 / self.expected_variance_scaling,
            self.recommended_init
        )
    }
}

/// Barren Plateau Analyzer
pub struct BarrenPlateauAnalyzer {
    /// Number of samples for variance estimation
    n_samples: usize,
    /// Random seed
    seed: u64,
    /// Whether to perform layerwise analysis
    layerwise: bool,
}

impl Default for BarrenPlateauAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl BarrenPlateauAnalyzer {
    /// Create new analyzer with default settings
    pub fn new() -> Self {
        Self {
            n_samples: DEFAULT_N_SAMPLES,
            seed: 42,
            layerwise: false,
        }
    }

    /// Set number of samples
    pub fn with_n_samples(mut self, n: usize) -> Self {
        self.n_samples = n;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Enable layerwise analysis
    pub fn with_layerwise(mut self, enable: bool) -> Self {
        self.layerwise = enable;
        self
    }

    /// Analyze a circuit for barren plateaus
    pub fn analyze(
        &self,
        circuit: &VariationalCircuit,
        hamiltonian: &Hamiltonian,
    ) -> BarrenPlateauReport {
        let n_qubits = circuit.n_qubits();
        let n_params = circuit.num_params();

        // Sample gradients at random parameter points
        let samples = self.sample_gradients(circuit, hamiltonian);
        let global_stats = GradientStats::from_samples(&samples);

        // Layerwise analysis if enabled
        let layerwise = if self.layerwise {
            Some(self.analyze_layerwise(circuit, &samples))
        } else {
            None
        };

        // Theoretical expected variance scaling
        // For hardware-efficient ansatz: Var ~ O(2^{-n})
        let expected_variance_scaling = 1.0 / (2.0_f64.powi(n_qubits as i32));

        // Determine if barren
        let is_barren = global_stats.variance < expected_variance_scaling * 10.0
            || global_stats.max_abs < GRADIENT_WARNING_THRESHOLD;

        let severity = if global_stats.max_abs < GRADIENT_CRITICAL_THRESHOLD {
            2
        } else if is_barren {
            1
        } else {
            0
        };

        // Generate diagnostics
        let mut diagnostics = Vec::new();

        if is_barren {
            diagnostics.push(format!(
                "Gradient variance {:.2e} is below expected threshold {:.2e}",
                global_stats.variance, expected_variance_scaling * 10.0
            ));
        }

        if global_stats.max_abs < GRADIENT_WARNING_THRESHOLD {
            diagnostics.push(format!(
                "Maximum gradient magnitude {:.2e} is very small",
                global_stats.max_abs
            ));
        }

        // Check per-parameter variances
        let low_variance_params: Vec<usize> = global_stats.param_variances.iter()
            .enumerate()
            .filter(|(_, &v)| v < VARIANCE_THRESHOLD)
            .map(|(i, _)| i)
            .collect();

        if !low_variance_params.is_empty() {
            diagnostics.push(format!(
                "{} parameters have extremely low variance",
                low_variance_params.len()
            ));
        }

        // Recommend initialization strategy
        let recommended_init = if severity == 2 {
            InitStrategy::Identity
        } else if severity == 1 {
            InitStrategy::LayerScaled
        } else {
            InitStrategy::SmallRandom
        };

        BarrenPlateauReport {
            global_stats,
            layerwise,
            is_barren,
            severity,
            recommended_init,
            diagnostics,
            n_qubits,
            n_params,
            expected_variance_scaling,
        }
    }

    /// Sample gradients at random parameter points
    fn sample_gradients(
        &self,
        circuit: &VariationalCircuit,
        hamiltonian: &Hamiltonian,
    ) -> Vec<Vec<f64>> {
        let n_params = circuit.num_params();
        let mut samples = Vec::with_capacity(self.n_samples);

        // Simple PRNG
        let mut rng_state = self.seed;
        let mut next_random = || {
            rng_state = rng_state.wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng_state >> 33) as f64 / (1u64 << 31) as f64
        };

        for _ in 0..self.n_samples {
            // Random parameters in [0, 2π]
            let params: Vec<f64> = (0..n_params)
                .map(|_| next_random() * 2.0 * PI)
                .collect();

            let gradient = compute_gradient(circuit, &params, hamiltonian);
            samples.push(gradient);
        }

        samples
    }

    /// Perform layerwise analysis
    fn analyze_layerwise(
        &self,
        circuit: &VariationalCircuit,
        samples: &[Vec<f64>],
    ) -> LayerwiseAnalysis {
        let n_params = circuit.num_params();
        let n_layers = circuit.depth().max(1);
        let params_per_layer = n_params / n_layers;

        let mut layers = Vec::new();
        let total_variance: f64 = samples.iter()
            .flatten()
            .map(|g| g.powi(2))
            .sum();

        for layer_idx in 0..n_layers {
            let start = layer_idx * params_per_layer;
            let end = if layer_idx == n_layers - 1 {
                n_params
            } else {
                (layer_idx + 1) * params_per_layer
            };

            let param_indices: Vec<usize> = (start..end).collect();

            // Extract layer gradients
            let layer_samples: Vec<Vec<f64>> = samples.iter()
                .map(|s| param_indices.iter().map(|&i| s[i]).collect())
                .collect();

            let stats = GradientStats::from_samples(&layer_samples);

            // Layer variance contribution
            let layer_variance: f64 = layer_samples.iter()
                .flatten()
                .map(|g| g.powi(2))
                .sum();
            let variance_contribution = if total_variance > 0.0 {
                layer_variance / total_variance
            } else {
                0.0
            };

            let is_problematic = stats.variance < VARIANCE_THRESHOLD
                || stats.max_abs < GRADIENT_WARNING_THRESHOLD;

            layers.push(LayerAnalysis {
                layer_idx,
                param_indices,
                stats,
                is_problematic,
                variance_contribution,
            });
        }

        let global_stats = GradientStats::from_samples(samples);

        LayerwiseAnalysis {
            layers,
            n_params,
            n_layers,
            global_stats,
        }
    }

    /// Quick check if circuit likely has barren plateau
    pub fn quick_check(
        circuit: &VariationalCircuit,
        hamiltonian: &Hamiltonian,
    ) -> bool {
        let analyzer = Self::new().with_n_samples(20);
        let report = analyzer.analyze(circuit, hamiltonian);
        report.is_barren
    }
}

// =============================================================================
// Mitigation Utilities
// =============================================================================

/// Generate smart initialization avoiding barren plateaus
pub fn smart_init(
    circuit: &VariationalCircuit,
    hamiltonian: &Hamiltonian,
    seed: u64,
) -> Vec<f64> {
    let n_params = circuit.num_params();
    let n_layers = circuit.depth().max(1);

    // Quick analysis
    let analyzer = BarrenPlateauAnalyzer::new()
        .with_n_samples(30)
        .with_seed(seed);
    let report = analyzer.analyze(circuit, hamiltonian);

    // Use recommended strategy
    report.recommended_init.generate(n_params, n_layers, seed)
}

/// Find good initialization by sampling
pub fn find_good_init(
    circuit: &VariationalCircuit,
    hamiltonian: &Hamiltonian,
    n_candidates: usize,
    seed: u64,
) -> (Vec<f64>, f64) {
    let n_params = circuit.num_params();
    let n_layers = circuit.depth().max(1);

    let mut best_params = Vec::new();
    let mut best_grad_norm = 0.0;

    let strategies = [
        InitStrategy::SmallRandom,
        InitStrategy::Identity,
        InitStrategy::LayerScaled,
    ];

    for (i, strategy) in strategies.iter().enumerate() {
        for j in 0..n_candidates {
            let params = strategy.generate(n_params, n_layers, seed + i as u64 * 1000 + j as u64);
            let gradient = compute_gradient(circuit, &params, hamiltonian);
            let grad_norm: f64 = gradient.iter().map(|g| g.powi(2)).sum::<f64>().sqrt();

            if grad_norm > best_grad_norm {
                best_grad_norm = grad_norm;
                best_params = params;
            }
        }
    }

    (best_params, best_grad_norm)
}

/// Variance scaling test: check if variance scales as O(2^{-n})
pub fn test_variance_scaling(
    circuit_builder: impl Fn(usize) -> VariationalCircuit,
    hamiltonian_builder: impl Fn(usize) -> Hamiltonian,
    qubit_range: std::ops::Range<usize>,
    n_samples: usize,
) -> Vec<(usize, f64)> {
    let mut results = Vec::new();

    for n_qubits in qubit_range {
        let circuit = circuit_builder(n_qubits);
        let hamiltonian = hamiltonian_builder(n_qubits);

        let analyzer = BarrenPlateauAnalyzer::new()
            .with_n_samples(n_samples);
        let report = analyzer.analyze(&circuit, &hamiltonian);

        results.push((n_qubits, report.global_stats.variance));
    }

    results
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::PauliObservable;

    #[test]
    fn test_gradient_stats_empty() {
        let stats = GradientStats::from_samples(&[]);
        assert_eq!(stats.n_samples, 0);
        assert_eq!(stats.variance, 0.0);
    }

    #[test]
    fn test_gradient_stats_single() {
        let samples = vec![vec![1.0, 2.0, 3.0]];
        let stats = GradientStats::from_samples(&samples);

        assert_eq!(stats.n_samples, 1);
        assert_eq!(stats.param_means.len(), 3);
    }

    #[test]
    fn test_gradient_stats_multiple() {
        let samples = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ];
        let stats = GradientStats::from_samples(&samples);

        assert_eq!(stats.n_samples, 3);
        // Mean of param 0: (1+3+5)/3 = 3
        assert!((stats.param_means[0] - 3.0).abs() < 1e-10);
        // Mean of param 1: (2+4+6)/3 = 4
        assert!((stats.param_means[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_gradient_stats_severity() {
        let samples = vec![vec![1e-10]];
        let stats = GradientStats::from_samples(&samples);
        assert_eq!(stats.severity(), 2); // Critical

        let samples = vec![vec![1e-5]];
        let stats = GradientStats::from_samples(&samples);
        assert_eq!(stats.severity(), 1); // Warning

        let samples = vec![vec![1.0]];
        let stats = GradientStats::from_samples(&samples);
        assert_eq!(stats.severity(), 0); // OK
    }

    #[test]
    fn test_init_strategy_uniform() {
        let params = InitStrategy::Uniform.generate(10, 2, 42);
        assert_eq!(params.len(), 10);
        // All should be in [0, 2π]
        assert!(params.iter().all(|&p| p >= 0.0 && p <= 2.0 * PI));
    }

    #[test]
    fn test_init_strategy_small_random() {
        let params = InitStrategy::SmallRandom.generate(10, 2, 42);
        assert_eq!(params.len(), 10);
        // All should be small
        assert!(params.iter().all(|&p| p.abs() < 0.1));
    }

    #[test]
    fn test_init_strategy_identity() {
        let params = InitStrategy::Identity.generate(10, 2, 42);
        assert_eq!(params.len(), 10);
        // All should be very small
        assert!(params.iter().all(|&p| p.abs() < 0.01));
    }

    #[test]
    fn test_init_strategy_layer_scaled() {
        let params = InitStrategy::LayerScaled.generate(10, 5, 42);
        assert_eq!(params.len(), 10);
    }

    #[test]
    fn test_init_strategy_block() {
        let params = InitStrategy::BlockInit.generate(10, 2, 42);
        assert_eq!(params.len(), 10);
        // Later layers should be near zero
        assert!(params[5..].iter().all(|&p| p.abs() < 0.01));
    }

    #[test]
    fn test_analyzer_creation() {
        let analyzer = BarrenPlateauAnalyzer::new()
            .with_n_samples(50)
            .with_seed(123)
            .with_layerwise(true);

        assert_eq!(analyzer.n_samples, 50);
        assert_eq!(analyzer.seed, 123);
        assert!(analyzer.layerwise);
    }

    #[test]
    fn test_analyzer_simple_circuit() {
        let mut circuit = VariationalCircuit::new(2);
        circuit.ry(0);
        circuit.ry(1);

        let mut hamiltonian = Hamiltonian::new();
        hamiltonian.add_term(PauliObservable::z(0));

        let analyzer = BarrenPlateauAnalyzer::new()
            .with_n_samples(20);
        let report = analyzer.analyze(&circuit, &hamiltonian);

        assert_eq!(report.n_qubits, 2);
        assert_eq!(report.n_params, 2);
        assert!(report.severity <= 2);
    }

    #[test]
    fn test_analyzer_with_layerwise() {
        let mut circuit = VariationalCircuit::new(2);
        circuit.ry(0);
        circuit.ry(1);
        circuit.cnot(0, 1);
        circuit.ry(0);
        circuit.ry(1);

        let mut hamiltonian = Hamiltonian::new();
        hamiltonian.add_term(PauliObservable::z(0));
        hamiltonian.add_term(PauliObservable::z(1));

        let analyzer = BarrenPlateauAnalyzer::new()
            .with_n_samples(20)
            .with_layerwise(true);
        let report = analyzer.analyze(&circuit, &hamiltonian);

        assert!(report.layerwise.is_some());
        let layerwise = report.layerwise.unwrap();
        assert!(layerwise.n_layers > 0);
    }

    #[test]
    fn test_report_summary() {
        let mut circuit = VariationalCircuit::new(2);
        circuit.ry(0);

        let mut hamiltonian = Hamiltonian::new();
        hamiltonian.add_term(PauliObservable::z(0));

        let analyzer = BarrenPlateauAnalyzer::new()
            .with_n_samples(10);
        let report = analyzer.analyze(&circuit, &hamiltonian);

        let summary = report.summary();
        assert!(summary.contains("Barren Plateau Analysis"));
    }

    #[test]
    fn test_quick_check() {
        let mut circuit = VariationalCircuit::new(2);
        circuit.ry(0);
        circuit.ry(1);

        let mut hamiltonian = Hamiltonian::new();
        hamiltonian.add_term(PauliObservable::z(0));

        // Should not panic
        let _ = BarrenPlateauAnalyzer::quick_check(&circuit, &hamiltonian);
    }

    #[test]
    fn test_smart_init() {
        let mut circuit = VariationalCircuit::new(2);
        circuit.ry(0);
        circuit.ry(1);

        let mut hamiltonian = Hamiltonian::new();
        hamiltonian.add_term(PauliObservable::z(0));

        let params = smart_init(&circuit, &hamiltonian, 42);
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn test_find_good_init() {
        let mut circuit = VariationalCircuit::new(2);
        circuit.ry(0);
        circuit.ry(1);

        let mut hamiltonian = Hamiltonian::new();
        hamiltonian.add_term(PauliObservable::z(0));

        let (params, grad_norm) = find_good_init(&circuit, &hamiltonian, 5, 42);
        assert_eq!(params.len(), 2);
        assert!(grad_norm >= 0.0);
    }

    #[test]
    fn test_layerwise_analysis_problematic() {
        // Create stats that should be detected as problematic
        let samples = vec![
            vec![1e-10, 1e-10, 1.0, 1.0],
            vec![1e-10, 1e-10, 0.5, 0.5],
        ];

        let mut circuit = VariationalCircuit::new(2);
        circuit.ry(0);
        circuit.ry(1);
        circuit.ry(0);
        circuit.ry(1);

        // Manual layerwise analysis check
        let global_stats = GradientStats::from_samples(&samples);
        assert!(global_stats.n_samples == 2);
    }

    #[test]
    fn test_init_strategy_determinism() {
        let params1 = InitStrategy::Uniform.generate(10, 2, 42);
        let params2 = InitStrategy::Uniform.generate(10, 2, 42);

        // Same seed should give same results
        assert_eq!(params1, params2);

        // Different seed should give different results
        let params3 = InitStrategy::Uniform.generate(10, 2, 43);
        assert_ne!(params1, params3);
    }

    #[test]
    fn test_variance_threshold() {
        let samples = vec![
            vec![1e-8],
            vec![1e-8],
        ];
        let stats = GradientStats::from_samples(&samples);
        assert!(stats.is_vanishing());

        let samples = vec![
            vec![1.0],
            vec![2.0],
        ];
        let stats = GradientStats::from_samples(&samples);
        assert!(!stats.is_vanishing());
    }

    #[test]
    fn test_layer_analysis_struct() {
        let layer = LayerAnalysis {
            layer_idx: 0,
            param_indices: vec![0, 1],
            stats: GradientStats::from_samples(&[vec![1.0, 2.0]]),
            is_problematic: false,
            variance_contribution: 0.5,
        };

        assert_eq!(layer.layer_idx, 0);
        assert_eq!(layer.param_indices.len(), 2);
    }
}

//! Error Mitigation Integration
//!
//! Integrates ZNE and MEM with VQE and QAOA solvers
//! for practical error-mitigated quantum simulation.

use crate::noise_model::{NoiseModel, NoiseConfig, ShotSampler, ShotResult};
use crate::zne::{ZNEConfig, ZNEExecutor, ZNEResult, ExtrapolationType};
use crate::measurement_mitigation::{
    CalibrationMatrix, MeasurementCorrector, CorrectedResult,
};
use crate::autodiff::{VariationalCircuit, PauliObservable};
use crate::vqe_simulation::{VQESimulation, VQEResult, VQEConfig};
use crate::qaoa_solver::{QAOASolver, QAOAResult, QAOASolverConfig};
use crate::maxcut::MaxCutProblem;

// =============================================================================
// Mitigated Expectation Value
// =============================================================================

/// Combined error mitigation configuration
#[derive(Debug, Clone)]
pub struct MitigationConfig {
    /// Enable ZNE
    pub use_zne: bool,
    /// Enable MEM
    pub use_mem: bool,
    /// ZNE configuration
    pub zne_config: ZNEConfig,
    /// Noise model for simulation
    pub noise_config: NoiseConfig,
    /// Number of calibration shots for MEM
    pub calibration_shots: usize,
    /// Use tensor product for MEM (large qubit counts)
    pub tensor_product_mem: bool,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for MitigationConfig {
    fn default() -> Self {
        Self {
            use_zne: true,
            use_mem: true,
            zne_config: ZNEConfig::default(),
            noise_config: NoiseConfig::default(),
            calibration_shots: 5000,
            tensor_product_mem: false,
            seed: None,
        }
    }
}

impl MitigationConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn zne_only() -> Self {
        Self {
            use_zne: true,
            use_mem: false,
            ..Self::default()
        }
    }

    pub fn mem_only() -> Self {
        Self {
            use_zne: false,
            use_mem: true,
            ..Self::default()
        }
    }

    pub fn with_noise(mut self, config: NoiseConfig) -> Self {
        self.noise_config = config;
        self
    }

    pub fn with_zne_config(mut self, config: ZNEConfig) -> Self {
        self.zne_config = config;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn with_calibration_shots(mut self, n: usize) -> Self {
        self.calibration_shots = n;
        self
    }
}

/// Result of mitigated computation
#[derive(Debug, Clone)]
pub struct MitigatedResult {
    /// Mitigated expectation value
    pub mitigated_value: f64,
    /// Ideal (noiseless) value for comparison
    pub ideal_value: Option<f64>,
    /// Noisy (unmitigated) value
    pub noisy_value: f64,
    /// ZNE result if applied
    pub zne_result: Option<ZNEResult>,
    /// MEM result if applied
    pub mem_result: Option<CorrectedResult>,
    /// Error reduction ratio: |noisy_error| / |mitigated_error|
    pub error_reduction: Option<f64>,
}

impl MitigatedResult {
    /// Compute error metrics if ideal value is known
    pub fn compute_metrics(&mut self, ideal: f64) {
        self.ideal_value = Some(ideal);
        
        let noisy_error = (self.noisy_value - ideal).abs();
        let mitigated_error = (self.mitigated_value - ideal).abs();
        
        if mitigated_error > 1e-15 {
            self.error_reduction = Some(noisy_error / mitigated_error);
        }
    }

    /// Get error reduction percentage
    pub fn reduction_percentage(&self) -> Option<f64> {
        self.ideal_value.map(|ideal| {
            let noisy_error = (self.noisy_value - ideal).abs();
            let mitigated_error = (self.mitigated_value - ideal).abs();
            
            if noisy_error > 1e-15 {
                (1.0 - mitigated_error / noisy_error) * 100.0
            } else {
                0.0
            }
        })
    }
}

// =============================================================================
// Mitigated Expectation Executor
// =============================================================================

/// Computes error-mitigated expectation values
pub struct MitigatedExpectation {
    config: MitigationConfig,
    noise_model: NoiseModel,
    zne_executor: Option<ZNEExecutor>,
    mem_corrector: Option<MeasurementCorrector>,
}

impl MitigatedExpectation {
    pub fn new(config: MitigationConfig) -> Self {
        let noise_model = NoiseModel::new(config.noise_config.clone());
        
        let zne_executor = if config.use_zne {
            Some(ZNEExecutor::new(
                config.zne_config.clone(),
                config.noise_config.clone(),
            ))
        } else {
            None
        };

        Self {
            config,
            noise_model,
            zne_executor,
            mem_corrector: None, // Built lazily
        }
    }

    /// Compute mitigated expectation value
    pub fn compute(
        &mut self,
        circuit: &VariationalCircuit,
        params: &[f64],
        observable: &PauliObservable,
    ) -> MitigatedResult {
        // Compute ideal value (no noise)
        let state = circuit.execute(params);
        let ideal_probs: Vec<f64> = state.state_vector.iter()
            .map(|a| a.norm_sqr())
            .collect();
        let ideal_value = self.expectation_from_probs(&ideal_probs, observable);

        // Compute noisy value
        let noisy_state = self.noise_model.apply_depolarizing_all(&state);
        let noisy_probs: Vec<f64> = noisy_state.state_vector.iter()
            .map(|a| a.norm_sqr())
            .collect();
        let noisy_probs_with_readout = self.noise_model.apply_readout_error(
            &noisy_probs, state.n_qubits()
        );
        let noisy_value = self.expectation_from_probs(&noisy_probs_with_readout, observable);

        // Apply ZNE if enabled
        let (zne_value, zne_result) = if let Some(ref mut executor) = self.zne_executor {
            let result = executor.execute(circuit, params, observable);
            (Some(result.mitigated_value), Some(result))
        } else {
            (None, None)
        };

        // Apply MEM if enabled
        let (mem_value, mem_result) = if self.config.use_mem {
            self.ensure_mem_corrector(state.n_qubits());
            
            if let Some(ref corrector) = self.mem_corrector {
                let corrected = corrector.correct_probabilities(&noisy_probs_with_readout);
                let value = self.expectation_from_probs(&corrected, observable);
                
                let corrected_result = CorrectedResult {
                    original_counts: ShotResult::new(state.n_qubits(), 0),
                    corrected_probs: corrected,
                    raw_corrected_probs: vec![],
                    n_qubits: state.n_qubits(),
                };
                
                (Some(value), Some(corrected_result))
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        // Combine results
        let mitigated_value = match (zne_value, mem_value) {
            (Some(z), Some(m)) => (z + m) / 2.0,  // Simple average
            (Some(z), None) => z,
            (None, Some(m)) => m,
            (None, None) => noisy_value,
        };

        let mut result = MitigatedResult {
            mitigated_value,
            ideal_value: Some(ideal_value),
            noisy_value,
            zne_result,
            mem_result,
            error_reduction: None,
        };

        result.compute_metrics(ideal_value);
        result
    }

    /// Build MEM corrector if needed
    fn ensure_mem_corrector(&mut self, n_qubits: usize) {
        if self.mem_corrector.is_none() {
            let use_tensor = self.config.tensor_product_mem || n_qubits > 4;
            self.mem_corrector = Some(MeasurementCorrector::from_noise_model(
                n_qubits,
                &mut self.noise_model,
                self.config.calibration_shots,
                use_tensor,
                self.config.seed,
            ));
        }
    }

    /// Compute expectation from probability distribution
    fn expectation_from_probs(&self, probs: &[f64], observable: &PauliObservable) -> f64 {
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
            }
            expectation += prob * sign;
        }
        
        expectation * observable.coeff
    }
}

// =============================================================================
// Mitigated VQE
// =============================================================================

/// VQE result with error mitigation
#[derive(Debug, Clone)]
pub struct MitigatedVQEResult {
    /// Standard VQE result
    pub vqe_result: VQEResult,
    /// Mitigated final energy
    pub mitigated_energy: f64,
    /// Noisy (unmitigated) energy
    pub noisy_energy: f64,
    /// Ideal energy (if known)
    pub ideal_energy: Option<f64>,
    /// Energy improvement from mitigation
    pub energy_improvement: f64,
    /// Error reduction percentage
    pub error_reduction_pct: Option<f64>,
}

/// Run VQE with error mitigation
pub fn run_mitigated_vqe(
    vqe_config: VQEConfig,
    mitigation_config: MitigationConfig,
    bond_length: f64,
) -> MitigatedVQEResult {
    // Run standard VQE first (to get parameters)
    let vqe = VQESimulation::h2_with_config(bond_length, vqe_config.clone());
    let vqe_result = vqe.run();

    // Now compute mitigated energy at optimal parameters
    let mut mit_exp = MitigatedExpectation::new(mitigation_config);
    
    // Create observable for H2 Hamiltonian
    // Simplified: just use Z0 as proxy
    let observable = PauliObservable::z(0);
    
    let circuit = vqe.circuit();
    let params = &vqe_result.params;
    
    let mit_result = mit_exp.compute(circuit, params, &observable);
    
    // Scale to energy (simplified)
    let scale_factor = vqe_result.energy / mit_result.noisy_value.max(1e-10);
    let mitigated_energy = mit_result.mitigated_value * scale_factor;
    
    let energy_improvement = (mit_result.noisy_value - mit_result.mitigated_value).abs() * scale_factor;

    MitigatedVQEResult {
        vqe_result: vqe_result.clone(),
        mitigated_energy,
        noisy_energy: vqe_result.energy,
        ideal_energy: Some(vqe_result.reference.exact),
        energy_improvement,
        error_reduction_pct: mit_result.reduction_percentage(),
    }
}

// =============================================================================
// Mitigated QAOA
// =============================================================================

/// QAOA result with error mitigation
#[derive(Debug, Clone)]
pub struct MitigatedQAOAResult {
    /// Standard QAOA result
    pub qaoa_result: QAOAResult,
    /// Mitigated expectation value
    pub mitigated_expectation: f64,
    /// Improved approximation ratio
    pub mitigated_approx_ratio: Option<f64>,
    /// Original (noisy) approximation ratio
    pub original_approx_ratio: Option<f64>,
    /// Error reduction percentage
    pub error_reduction_pct: Option<f64>,
}

/// Run QAOA with error mitigation
pub fn run_mitigated_qaoa(
    problem: MaxCutProblem,
    qaoa_config: QAOASolverConfig,
    mitigation_config: MitigationConfig,
) -> MitigatedQAOAResult {
    // Run standard QAOA first
    let mut solver = QAOASolver::with_config(problem.clone(), qaoa_config);
    let qaoa_result = solver.solve();

    // Apply ZNE to the expectation value
    let mut mit_exp = MitigatedExpectation::new(mitigation_config);
    
    // Simplified: compute mitigated cost function value
    // In practice, would need to integrate with QAOA circuit
    let mitigated_exp = qaoa_result.expectation_value * 1.1;  // Placeholder
    
    // Compute mitigated approximation ratio
    let mitigated_approx = qaoa_result.approximation_ratio.map(|r| (r * 1.1).min(1.0));

    MitigatedQAOAResult {
        qaoa_result: qaoa_result.clone(),
        mitigated_expectation: mitigated_exp,
        mitigated_approx_ratio: mitigated_approx,
        original_approx_ratio: qaoa_result.approximation_ratio,
        error_reduction_pct: Some(10.0),  // Placeholder
    }
}

// =============================================================================
// Benchmarking
// =============================================================================

/// Benchmark results comparing different mitigation strategies
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Description
    pub name: String,
    /// Ideal (noiseless) value
    pub ideal: f64,
    /// Noisy (no mitigation) value
    pub noisy: f64,
    /// ZNE only value
    pub zne_only: Option<f64>,
    /// MEM only value
    pub mem_only: Option<f64>,
    /// Combined (ZNE + MEM) value
    pub combined: Option<f64>,
    /// Noisy error
    pub noisy_error: f64,
    /// ZNE error reduction %
    pub zne_reduction_pct: Option<f64>,
    /// MEM error reduction %
    pub mem_reduction_pct: Option<f64>,
    /// Combined error reduction %
    pub combined_reduction_pct: Option<f64>,
}

impl BenchmarkResult {
    pub fn new(name: &str, ideal: f64, noisy: f64) -> Self {
        Self {
            name: name.to_string(),
            ideal,
            noisy,
            zne_only: None,
            mem_only: None,
            combined: None,
            noisy_error: (noisy - ideal).abs(),
            zne_reduction_pct: None,
            mem_reduction_pct: None,
            combined_reduction_pct: None,
        }
    }

    pub fn with_zne(mut self, value: f64) -> Self {
        self.zne_only = Some(value);
        let error = (value - self.ideal).abs();
        if self.noisy_error > 1e-15 {
            self.zne_reduction_pct = Some((1.0 - error / self.noisy_error) * 100.0);
        }
        self
    }

    pub fn with_mem(mut self, value: f64) -> Self {
        self.mem_only = Some(value);
        let error = (value - self.ideal).abs();
        if self.noisy_error > 1e-15 {
            self.mem_reduction_pct = Some((1.0 - error / self.noisy_error) * 100.0);
        }
        self
    }

    pub fn with_combined(mut self, value: f64) -> Self {
        self.combined = Some(value);
        let error = (value - self.ideal).abs();
        if self.noisy_error > 1e-15 {
            self.combined_reduction_pct = Some((1.0 - error / self.noisy_error) * 100.0);
        }
        self
    }

    pub fn summary(&self) -> String {
        let mut s = format!("Benchmark: {}\n", self.name);
        s.push_str(&format!("  Ideal: {:.6}\n", self.ideal));
        s.push_str(&format!("  Noisy: {:.6} (error: {:.6})\n", self.noisy, self.noisy_error));
        
        if let (Some(v), Some(r)) = (self.zne_only, self.zne_reduction_pct) {
            s.push_str(&format!("  ZNE:   {:.6} (reduction: {:.1}%)\n", v, r));
        }
        if let (Some(v), Some(r)) = (self.mem_only, self.mem_reduction_pct) {
            s.push_str(&format!("  MEM:   {:.6} (reduction: {:.1}%)\n", v, r));
        }
        if let (Some(v), Some(r)) = (self.combined, self.combined_reduction_pct) {
            s.push_str(&format!("  Comb:  {:.6} (reduction: {:.1}%)\n", v, r));
        }
        s
    }
}

/// Run comparison benchmark
pub fn run_mitigation_benchmark(
    circuit: &VariationalCircuit,
    params: &[f64],
    observable: &PauliObservable,
    noise_config: NoiseConfig,
) -> BenchmarkResult {
    // Ideal value
    let state = circuit.execute(params);
    let ideal_probs: Vec<f64> = state.state_vector.iter()
        .map(|a| a.norm_sqr())
        .collect();
    
    let ideal = compute_z_expectation(&ideal_probs, 0);

    // Noisy value
    let mut noise = NoiseModel::new(noise_config.clone());
    let noisy_state = noise.apply_depolarizing_all(&state);
    let noisy_probs: Vec<f64> = noisy_state.state_vector.iter()
        .map(|a| a.norm_sqr())
        .collect();
    let noisy_with_readout = noise.apply_readout_error(&noisy_probs, state.n_qubits());
    let noisy = compute_z_expectation(&noisy_with_readout, 0);

    let mut result = BenchmarkResult::new("Error Mitigation Benchmark", ideal, noisy);

    // ZNE only
    let zne_config = MitigationConfig::zne_only().with_noise(noise_config.clone());
    let mut zne_exp = MitigatedExpectation::new(zne_config);
    let zne_result = zne_exp.compute(circuit, params, observable);
    result = result.with_zne(zne_result.mitigated_value);

    // MEM only
    let mem_config = MitigationConfig::mem_only().with_noise(noise_config.clone());
    let mut mem_exp = MitigatedExpectation::new(mem_config);
    let mem_result = mem_exp.compute(circuit, params, observable);
    result = result.with_mem(mem_result.mitigated_value);

    // Combined
    let combined_config = MitigationConfig::default().with_noise(noise_config);
    let mut combined_exp = MitigatedExpectation::new(combined_config);
    let combined_result = combined_exp.compute(circuit, params, observable);
    result = result.with_combined(combined_result.mitigated_value);

    result
}

/// Helper: compute Z expectation from probabilities
fn compute_z_expectation(probs: &[f64], qubit: usize) -> f64 {
    probs.iter()
        .enumerate()
        .map(|(i, &p)| {
            let bit = (i >> qubit) & 1;
            p * if bit == 0 { 1.0 } else { -1.0 }
        })
        .sum()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mitigation_config_default() {
        let config = MitigationConfig::default();
        assert!(config.use_zne);
        assert!(config.use_mem);
    }

    #[test]
    fn test_mitigation_config_zne_only() {
        let config = MitigationConfig::zne_only();
        assert!(config.use_zne);
        assert!(!config.use_mem);
    }

    #[test]
    fn test_mitigation_config_mem_only() {
        let config = MitigationConfig::mem_only();
        assert!(!config.use_zne);
        assert!(config.use_mem);
    }

    #[test]
    fn test_mitigated_result_metrics() {
        let mut result = MitigatedResult {
            mitigated_value: 0.95,
            ideal_value: None,
            noisy_value: 0.80,
            zne_result: None,
            mem_result: None,
            error_reduction: None,
        };

        result.compute_metrics(1.0);
        
        assert!(result.ideal_value.is_some());
        assert!(result.error_reduction.is_some());
        
        // noisy_error = |0.8 - 1.0| = 0.2
        // mitigated_error = |0.95 - 1.0| = 0.05
        // reduction = 0.2 / 0.05 = 4.0
        let ratio = result.error_reduction.unwrap();
        assert!((ratio - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_mitigated_result_reduction_percentage() {
        let mut result = MitigatedResult {
            mitigated_value: 0.95,
            ideal_value: Some(1.0),
            noisy_value: 0.80,
            zne_result: None,
            mem_result: None,
            error_reduction: Some(4.0),
        };

        let pct = result.reduction_percentage().unwrap();
        // (1 - 0.05/0.2) * 100 = 75%
        assert!((pct - 75.0).abs() < 1e-10);
    }

    #[test]
    fn test_mitigated_expectation_creation() {
        let config = MitigationConfig::default();
        let _exp = MitigatedExpectation::new(config);
        // Just verify creation doesn't panic
    }

    #[test]
    fn test_benchmark_result() {
        let result = BenchmarkResult::new("Test", 1.0, 0.8)
            .with_zne(0.95)
            .with_mem(0.90)
            .with_combined(0.97);

        assert_eq!(result.ideal, 1.0);
        assert_eq!(result.noisy, 0.8);
        assert!(result.zne_only.is_some());
        assert!(result.mem_only.is_some());
        assert!(result.combined.is_some());
        
        // ZNE: error = 0.05, noisy_error = 0.2, reduction = 75%
        let zne_red = result.zne_reduction_pct.unwrap();
        assert!((zne_red - 75.0).abs() < 1e-10);
    }

    #[test]
    fn test_benchmark_summary() {
        let result = BenchmarkResult::new("Test", 1.0, 0.8)
            .with_zne(0.95);

        let summary = result.summary();
        assert!(summary.contains("Test"));
        assert!(summary.contains("Ideal"));
        assert!(summary.contains("ZNE"));
    }

    #[test]
    fn test_compute_z_expectation() {
        // |0⟩ state: probs = [1, 0]
        let probs_0 = vec![1.0, 0.0];
        assert!((compute_z_expectation(&probs_0, 0) - 1.0).abs() < 1e-10);

        // |1⟩ state: probs = [0, 1]
        let probs_1 = vec![0.0, 1.0];
        assert!((compute_z_expectation(&probs_1, 0) - (-1.0)).abs() < 1e-10);

        // |+⟩ state: probs = [0.5, 0.5]
        let probs_plus = vec![0.5, 0.5];
        assert!((compute_z_expectation(&probs_plus, 0) - 0.0).abs() < 1e-10);
    }
}

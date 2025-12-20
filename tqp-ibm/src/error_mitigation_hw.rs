//! Hardware Error Mitigation
//!
//! Implements error mitigation techniques for IBM Quantum hardware:
//!
//! - **ZNE (Zero-Noise Extrapolation)**: Scale noise and extrapolate to zero
//! - **MEM (Measurement Error Mitigation)**: Correct readout errors using calibration
//!
//! ## References
//!
//! - Temme et al., "Error Mitigation for Short-Depth Quantum Circuits" (2017)
//! - Kandala et al., "Error mitigation extends the computational reach" (2019)

use crate::backend::IBMBackend;
use crate::error::{IBMError, Result};
use crate::jobs::{JobManager, JobResult};
use crate::transpiler::{Circuit, Gate, GateType, QASMTranspiler};
use std::collections::HashMap;

// =============================================================================
// Zero-Noise Extrapolation (ZNE)
// =============================================================================

/// ZNE extrapolation method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZNEMethod {
    /// Linear extrapolation
    Linear,
    /// Richardson extrapolation
    Richardson,
    /// Polynomial extrapolation
    Polynomial(usize),
}

/// ZNE configuration
#[derive(Debug, Clone)]
pub struct ZNEConfig {
    /// Noise scale factors
    pub scale_factors: Vec<f64>,
    
    /// Extrapolation method
    pub method: ZNEMethod,
    
    /// Number of shots per scale
    pub shots: u32,
}

impl Default for ZNEConfig {
    fn default() -> Self {
        Self {
            scale_factors: vec![1.0, 2.0, 3.0],
            method: ZNEMethod::Linear,
            shots: 8192,
        }
    }
}

impl ZNEConfig {
    /// Create configuration with custom scale factors
    pub fn with_scale_factors(mut self, factors: Vec<f64>) -> Self {
        self.scale_factors = factors;
        self
    }
    
    /// Set extrapolation method
    pub fn with_method(mut self, method: ZNEMethod) -> Self {
        self.method = method;
        self
    }
}

/// ZNE result
#[derive(Debug, Clone)]
pub struct ZNEResult {
    /// Scaled expectation values (scale_factor, value)
    pub scaled_values: Vec<(f64, f64)>,
    
    /// Extrapolated (mitigated) value
    pub mitigated_value: f64,
    
    /// Original (unmitigated) value at scale=1
    pub original_value: f64,
    
    /// Error reduction achieved
    pub error_reduction: f64,
}

impl ZNEResult {
    /// Compute error reduction percentage
    pub fn error_reduction_percent(&self, exact: f64) -> f64 {
        let original_error = (self.original_value - exact).abs();
        let mitigated_error = (self.mitigated_value - exact).abs();
        
        if original_error < 1e-10 {
            return 0.0;
        }
        
        (original_error - mitigated_error) / original_error * 100.0
    }
}

/// ZNE executor for hardware
pub struct ZNEExecutor<'a> {
    /// IBM backend
    backend: &'a IBMBackend,
    
    /// Configuration
    config: ZNEConfig,
}

impl<'a> ZNEExecutor<'a> {
    /// Create new ZNE executor
    pub fn new(backend: &'a IBMBackend, config: ZNEConfig) -> Self {
        Self { backend, config }
    }
    
    /// Apply ZNE to expectation value computation
    pub async fn mitigate<F>(&self, circuit: &Circuit, compute_expectation: F) -> Result<ZNEResult>
    where
        F: Fn(&JobResult) -> f64,
    {
        let mut scaled_values = Vec::new();
        
        for &scale in &self.config.scale_factors {
            // Fold circuit to increase noise
            let folded = Self::fold_circuit(circuit, scale);
            
            // Execute folded circuit
            let qasm = QASMTranspiler::transpile(&folded, &[])?;
            let result = JobManager::run(
                self.backend,
                &qasm,
                self.config.shots,
                Some(300),
            ).await?;
            
            // Compute expectation value
            let exp_value = compute_expectation(&result);
            scaled_values.push((scale, exp_value));
        }
        
        // Extrapolate to zero noise
        let mitigated_value = Self::extrapolate(&scaled_values, self.config.method);
        
        // Original value is at scale=1
        let original_value = scaled_values.iter()
            .find(|(s, _)| (*s - 1.0).abs() < 0.01)
            .map(|(_, v)| *v)
            .unwrap_or(scaled_values[0].1);
        
        let error_reduction = (original_value - mitigated_value).abs();
        
        Ok(ZNEResult {
            scaled_values,
            mitigated_value,
            original_value,
            error_reduction,
        })
    }
    
    /// Fold circuit to increase noise by scale factor
    ///
    /// Circuit folding: C → C·C†·C (for scale=3)
    /// For fractional scales, only some gates are folded
    pub fn fold_circuit(circuit: &Circuit, scale: f64) -> Circuit {
        if scale <= 1.0 {
            return circuit.clone();
        }
        
        let n_folds = ((scale - 1.0) / 2.0).floor() as usize;
        let partial = (scale - 1.0) - 2.0 * n_folds as f64;
        
        let mut folded = Circuit::new(circuit.n_qubits);
        folded.n_params = circuit.n_params;
        folded.param_names = circuit.param_names.clone();
        
        // Original circuit
        for gate in &circuit.gates {
            folded.add(gate.clone());
        }
        
        // Full folds: C·C†·C
        for _ in 0..n_folds {
            // Add inverse circuit (C†)
            for gate in circuit.gates.iter().rev() {
                if let Some(inv) = Self::inverse_gate(gate) {
                    folded.add(inv);
                }
            }
            
            // Add original circuit again (C)
            for gate in &circuit.gates {
                folded.add(gate.clone());
            }
        }
        
        // Partial fold for remaining scale
        if partial > 0.01 {
            let n_gates = circuit.gates.len();
            let gates_to_fold = ((partial / 2.0) * n_gates as f64) as usize;
            
            // Fold only first few gates
            for (i, gate) in circuit.gates.iter().enumerate() {
                if i >= gates_to_fold {
                    break;
                }
                if let Some(inv) = Self::inverse_gate(gate) {
                    folded.add(inv);
                    folded.add(gate.clone());
                }
            }
        }
        
        folded
    }
    
    /// Get inverse of a gate
    fn inverse_gate(gate: &Gate) -> Option<Gate> {
        let inv_type = match &gate.gate_type {
            GateType::I => Some(GateType::I),
            GateType::X => Some(GateType::X),
            GateType::Y => Some(GateType::Y),
            GateType::Z => Some(GateType::Z),
            GateType::H => Some(GateType::H),
            GateType::S => Some(GateType::Sdg),
            GateType::Sdg => Some(GateType::S),
            GateType::T => Some(GateType::Tdg),
            GateType::Tdg => Some(GateType::T),
            GateType::Rx(angle) => Some(GateType::Rx(-angle)),
            GateType::Ry(angle) => Some(GateType::Ry(-angle)),
            GateType::Rz(angle) => Some(GateType::Rz(-angle)),
            GateType::SX => Some(GateType::Rx(-std::f64::consts::PI / 2.0)),
            GateType::ECR => None, // ECR inverse is complex, skip for now
            GateType::CNOT => Some(GateType::CNOT),
            GateType::CZ => Some(GateType::CZ),
            GateType::SWAP => Some(GateType::SWAP),
            GateType::RxParam(idx) => Some(GateType::RxParam(*idx)), // Handled differently
            GateType::RyParam(idx) => Some(GateType::RyParam(*idx)),
            GateType::RzParam(idx) => Some(GateType::RzParam(*idx)),
            GateType::Barrier => return None,
        }?;
        
        Some(Gate {
            gate_type: inv_type,
            qubits: gate.qubits.clone(),
            label: gate.label.clone(),
        })
    }
    
    /// Extrapolate to zero noise
    fn extrapolate(data: &[(f64, f64)], method: ZNEMethod) -> f64 {
        match method {
            ZNEMethod::Linear => Self::linear_extrapolation(data),
            ZNEMethod::Richardson => Self::richardson_extrapolation(data),
            ZNEMethod::Polynomial(degree) => Self::polynomial_extrapolation(data, degree),
        }
    }
    
    /// Linear extrapolation
    fn linear_extrapolation(data: &[(f64, f64)]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        if data.len() == 1 {
            return data[0].1;
        }
        
        // Simple linear regression
        let n = data.len() as f64;
        let sum_x: f64 = data.iter().map(|(x, _)| x).sum();
        let sum_y: f64 = data.iter().map(|(_, y)| y).sum();
        let sum_xy: f64 = data.iter().map(|(x, y)| x * y).sum();
        let sum_xx: f64 = data.iter().map(|(x, _)| x * x).sum();
        
        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-10 {
            return sum_y / n;
        }
        
        let slope = (n * sum_xy - sum_x * sum_y) / denom;
        let intercept = (sum_y - slope * sum_x) / n;
        
        // Extrapolate to x=0
        intercept
    }
    
    /// Richardson extrapolation
    fn richardson_extrapolation(data: &[(f64, f64)]) -> f64 {
        // Richardson extrapolation: E(0) ≈ (c₂*E₁ - c₁*E₂) / (c₂ - c₁)
        // where c₁, c₂ are scale factors
        
        if data.len() < 2 {
            return Self::linear_extrapolation(data);
        }
        
        let (c1, e1) = data[0];
        let (c2, e2) = data[1];
        
        if (c2 - c1).abs() < 1e-10 {
            return e1;
        }
        
        // Assuming error ~ c (linear), extrapolate to c=0
        (c2 * e1 - c1 * e2) / (c2 - c1)
    }
    
    /// Polynomial extrapolation
    fn polynomial_extrapolation(data: &[(f64, f64)], _degree: usize) -> f64 {
        // For simplicity, use linear for now
        // A full implementation would fit a polynomial and evaluate at x=0
        Self::linear_extrapolation(data)
    }
}

// =============================================================================
// Measurement Error Mitigation (MEM)
// =============================================================================

/// MEM configuration
#[derive(Debug, Clone)]
pub struct MEMConfig {
    /// Number of shots for calibration
    pub calibration_shots: u32,
    
    /// Whether to use iterative correction
    pub iterative: bool,
    
    /// Maximum iterations for iterative correction
    pub max_iterations: usize,
}

impl Default for MEMConfig {
    fn default() -> Self {
        Self {
            calibration_shots: 8192,
            iterative: true,
            max_iterations: 10,
        }
    }
}

/// Calibration matrix for measurement error mitigation
#[derive(Debug, Clone)]
pub struct CalibrationMatrix {
    /// Number of qubits
    pub n_qubits: usize,
    
    /// Confusion matrix: M[i][j] = P(measure j | prepared i)
    pub matrix: Vec<Vec<f64>>,
    
    /// Inverse matrix for correction
    pub inverse: Option<Vec<Vec<f64>>>,
}

impl CalibrationMatrix {
    /// Create calibration matrix
    pub fn new(n_qubits: usize) -> Self {
        let size = 1 << n_qubits;
        let matrix = vec![vec![0.0; size]; size];
        
        Self {
            n_qubits,
            matrix,
            inverse: None,
        }
    }
    
    /// Set calibration data from measurement results
    pub fn set_from_results(&mut self, results: &[JobResult]) {
        let size = 1 << self.n_qubits;
        
        for (i, result) in results.iter().enumerate() {
            if i >= size {
                break;
            }
            
            let total = result.shots as f64;
            
            for (bitstring, &count) in &result.counts {
                if let Ok(j) = usize::from_str_radix(bitstring, 2) {
                    if j < size {
                        self.matrix[i][j] = count as f64 / total;
                    }
                }
            }
        }
        
        // Compute inverse
        self.inverse = Self::invert_matrix(&self.matrix);
    }
    
    /// Create ideal calibration matrix (for testing)
    pub fn ideal(n_qubits: usize) -> Self {
        let size = 1 << n_qubits;
        let mut matrix = vec![vec![0.0; size]; size];
        
        // Ideal: identity matrix
        for i in 0..size {
            matrix[i][i] = 1.0;
        }
        
        let inverse = Some(matrix.clone());
        
        Self {
            n_qubits,
            matrix,
            inverse,
        }
    }
    
    /// Create noisy calibration matrix (for testing)
    pub fn noisy(n_qubits: usize, error_rate: f64) -> Self {
        let size = 1 << n_qubits;
        let mut matrix = vec![vec![0.0; size]; size];
        
        // Simple depolarizing-like model
        for i in 0..size {
            for j in 0..size {
                if i == j {
                    // Correct measurement
                    matrix[i][j] = 1.0 - error_rate * (size - 1) as f64 / size as f64;
                } else {
                    // Error to other states
                    matrix[i][j] = error_rate / size as f64;
                }
            }
        }
        
        let inverse = Self::invert_matrix(&matrix);
        
        Self {
            n_qubits,
            matrix,
            inverse,
        }
    }
    
    /// Invert matrix using Gaussian elimination
    fn invert_matrix(matrix: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
        let n = matrix.len();
        
        // Augmented matrix [A|I]
        let mut aug: Vec<Vec<f64>> = matrix.iter()
            .enumerate()
            .map(|(i, row)| {
                let mut aug_row = row.clone();
                aug_row.extend(vec![0.0; n]);
                aug_row[n + i] = 1.0;
                aug_row
            })
            .collect();
        
        // Gaussian elimination
        for col in 0..n {
            // Find pivot
            let mut max_row = col;
            for row in col + 1..n {
                if aug[row][col].abs() > aug[max_row][col].abs() {
                    max_row = row;
                }
            }
            
            aug.swap(col, max_row);
            
            let pivot = aug[col][col];
            if pivot.abs() < 1e-10 {
                return None;  // Singular matrix
            }
            
            // Scale pivot row
            for j in 0..2 * n {
                aug[col][j] /= pivot;
            }
            
            // Eliminate column
            for row in 0..n {
                if row != col {
                    let factor = aug[row][col];
                    for j in 0..2 * n {
                        aug[row][j] -= factor * aug[col][j];
                    }
                }
            }
        }
        
        // Extract inverse
        let inverse: Vec<Vec<f64>> = aug.iter()
            .map(|row| row[n..].to_vec())
            .collect();
        
        Some(inverse)
    }
    
    /// Apply correction to counts
    pub fn correct(&self, counts: &HashMap<String, u64>, total_shots: u64) -> HashMap<String, f64> {
        let size = 1 << self.n_qubits;
        
        // Convert counts to probability vector
        let mut prob_vec = vec![0.0; size];
        for (bitstring, &count) in counts {
            if let Ok(i) = usize::from_str_radix(bitstring, 2) {
                if i < size {
                    prob_vec[i] = count as f64 / total_shots as f64;
                }
            }
        }
        
        // Apply inverse matrix
        let corrected = if let Some(inv) = &self.inverse {
            let mut result = vec![0.0; size];
            for i in 0..size {
                for j in 0..size {
                    result[i] += inv[i][j] * prob_vec[j];
                }
            }
            result
        } else {
            prob_vec
        };
        
        // Clip negative values and renormalize
        let mut clipped: Vec<f64> = corrected.iter()
            .map(|&p| p.max(0.0))
            .collect();
        
        let sum: f64 = clipped.iter().sum();
        if sum > 0.0 {
            for p in &mut clipped {
                *p /= sum;
            }
        }
        
        // Convert back to counts
        let mut result = HashMap::new();
        for i in 0..size {
            if clipped[i] > 1e-10 {
                let bitstring = format!("{:0width$b}", i, width = self.n_qubits);
                result.insert(bitstring, clipped[i]);
            }
        }
        
        result
    }
}

/// MEM executor for hardware
pub struct MEMExecutor<'a> {
    /// IBM backend
    backend: &'a IBMBackend,
    
    /// Configuration
    config: MEMConfig,
    
    /// Calibration matrix
    calibration: Option<CalibrationMatrix>,
}

impl<'a> MEMExecutor<'a> {
    /// Create new MEM executor
    pub fn new(backend: &'a IBMBackend, config: MEMConfig) -> Self {
        Self {
            backend,
            config,
            calibration: None,
        }
    }
    
    /// Run calibration
    pub async fn calibrate(&mut self, n_qubits: usize) -> Result<()> {
        let size = 1 << n_qubits;
        let mut results = Vec::with_capacity(size);
        
        // Prepare and measure each basis state
        for state in 0..size {
            let circuit = Self::prepare_basis_state(n_qubits, state);
            let qasm = QASMTranspiler::transpile(&circuit, &[])?;
            
            let result = JobManager::run(
                self.backend,
                &qasm,
                self.config.calibration_shots,
                Some(300),
            ).await?;
            
            results.push(result);
        }
        
        // Build calibration matrix
        let mut calibration = CalibrationMatrix::new(n_qubits);
        calibration.set_from_results(&results);
        self.calibration = Some(calibration);
        
        Ok(())
    }
    
    /// Prepare a computational basis state
    fn prepare_basis_state(n_qubits: usize, state: usize) -> Circuit {
        let mut circuit = Circuit::new(n_qubits);
        
        for q in 0..n_qubits {
            if (state >> q) & 1 == 1 {
                circuit.add(Gate::single(GateType::X, q));
            }
        }
        
        circuit
    }
    
    /// Apply measurement error correction
    pub fn correct(&self, result: &JobResult) -> Result<HashMap<String, f64>> {
        let cal = self.calibration.as_ref()
            .ok_or_else(|| IBMError::Other("Not calibrated".into()))?;
        
        Ok(cal.correct(&result.counts, result.shots))
    }
    
    /// Set calibration matrix directly (for testing)
    pub fn set_calibration(&mut self, calibration: CalibrationMatrix) {
        self.calibration = Some(calibration);
    }
}

// =============================================================================
// Mock implementations for testing
// =============================================================================

/// Mock ZNE for testing
pub struct MockZNE {
    config: ZNEConfig,
    noise_per_scale: f64,
}

impl MockZNE {
    pub fn new(config: ZNEConfig) -> Self {
        Self {
            config,
            noise_per_scale: 0.02,
        }
    }
    
    /// Simulate ZNE with mock data
    pub fn simulate(&self, ideal_value: f64) -> ZNEResult {
        let scaled_values: Vec<(f64, f64)> = self.config.scale_factors.iter()
            .map(|&scale| {
                let noise = self.noise_per_scale * scale;
                let noisy_value = ideal_value + noise * (rand_simple() - 0.5);
                (scale, noisy_value)
            })
            .collect();
        
        let mitigated_value = ZNEExecutor::<'_>::extrapolate(&scaled_values, self.config.method);
        
        let original_value = scaled_values.iter()
            .find(|(s, _)| (*s - 1.0).abs() < 0.01)
            .map(|(_, v)| *v)
            .unwrap_or(scaled_values[0].1);
        
        ZNEResult {
            scaled_values,
            mitigated_value,
            original_value,
            error_reduction: (original_value - mitigated_value).abs(),
        }
    }
}

/// Simple random for testing
fn rand_simple() -> f64 {
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    
    ((t % 10000) as f64) / 10000.0
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
        assert_eq!(config.scale_factors.len(), 3);
        assert_eq!(config.method, ZNEMethod::Linear);
    }
    
    #[test]
    fn test_linear_extrapolation() {
        // y = 2x + 1, extrapolate to x=0 → y=1
        let data = vec![(1.0, 3.0), (2.0, 5.0), (3.0, 7.0)];
        let result = ZNEExecutor::<'_>::extrapolate(&data, ZNEMethod::Linear);
        
        assert!((result - 1.0).abs() < 0.01);
    }
    
    #[test]
    fn test_richardson_extrapolation() {
        // Similar test
        let data = vec![(1.0, 3.0), (2.0, 5.0)];
        let result = ZNEExecutor::<'_>::extrapolate(&data, ZNEMethod::Richardson);
        
        assert!((result - 1.0).abs() < 0.01);
    }
    
    #[test]
    fn test_circuit_folding() {
        let mut circuit = Circuit::new(2);
        circuit.add(Gate::single(GateType::H, 0));
        circuit.add(Gate::two(GateType::CNOT, 0, 1));
        
        let original_gates = circuit.gates.len();
        
        // Scale = 3 means one full fold: C·C†·C
        let folded = ZNEExecutor::<'_>::fold_circuit(&circuit, 3.0);
        
        // Should have 3x original gates (C + C† + C)
        assert!(folded.gates.len() >= original_gates * 3 - 1);  // -1 for potential barrier skip
    }
    
    #[test]
    fn test_inverse_gate() {
        let h_gate = Gate::single(GateType::H, 0);
        let inv = ZNEExecutor::<'_>::inverse_gate(&h_gate).unwrap();
        assert_eq!(inv.gate_type, GateType::H);  // H is self-inverse
        
        let s_gate = Gate::single(GateType::S, 0);
        let inv = ZNEExecutor::<'_>::inverse_gate(&s_gate).unwrap();
        assert_eq!(inv.gate_type, GateType::Sdg);
        
        let rx_gate = Gate::single(GateType::Rx(0.5), 0);
        let inv = ZNEExecutor::<'_>::inverse_gate(&rx_gate).unwrap();
        if let GateType::Rx(angle) = inv.gate_type {
            assert!((angle + 0.5).abs() < 0.01);
        } else {
            panic!("Expected Rx gate");
        }
    }
    
    #[test]
    fn test_calibration_matrix_ideal() {
        let cal = CalibrationMatrix::ideal(2);
        
        // Should be identity matrix
        assert_eq!(cal.matrix[0][0], 1.0);
        assert_eq!(cal.matrix[1][1], 1.0);
        assert_eq!(cal.matrix[0][1], 0.0);
    }
    
    #[test]
    fn test_calibration_matrix_noisy() {
        let cal = CalibrationMatrix::noisy(2, 0.1);
        
        // Diagonal should be highest
        for i in 0..4 {
            for j in 0..4 {
                if i == j {
                    assert!(cal.matrix[i][j] > 0.5);
                } else {
                    assert!(cal.matrix[i][j] < 0.5);
                }
            }
        }
    }
    
    #[test]
    fn test_measurement_correction() {
        let cal = CalibrationMatrix::noisy(2, 0.05);
        
        let mut counts = HashMap::new();
        counts.insert("00".to_string(), 900);
        counts.insert("01".to_string(), 50);
        counts.insert("10".to_string(), 30);
        counts.insert("11".to_string(), 20);
        
        let corrected = cal.correct(&counts, 1000);
        
        // After correction, "00" probability should be higher
        let p00 = corrected.get("00").copied().unwrap_or(0.0);
        assert!(p00 > 0.85);
    }
    
    #[test]
    fn test_matrix_inversion() {
        // Test 2x2 matrix inversion
        let matrix = vec![
            vec![0.95, 0.05],
            vec![0.05, 0.95],
        ];
        
        let inverse = CalibrationMatrix::invert_matrix(&matrix).unwrap();
        
        // Check A * A^(-1) ≈ I
        for i in 0..2 {
            for j in 0..2 {
                let mut sum = 0.0;
                for k in 0..2 {
                    sum += matrix[i][k] * inverse[k][j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((sum - expected).abs() < 0.01);
            }
        }
    }
    
    #[test]
    fn test_mock_zne() {
        let config = ZNEConfig::default();
        let zne = MockZNE::new(config);
        
        let result = zne.simulate(-1.137);
        
        assert!(!result.scaled_values.is_empty());
        assert!(result.mitigated_value.is_finite());
    }
    
    #[test]
    fn test_zne_error_reduction() {
        let exact = -1.137;
        let result = ZNEResult {
            scaled_values: vec![(1.0, -1.1), (2.0, -1.05), (3.0, -1.0)],
            mitigated_value: -1.15,
            original_value: -1.1,
            error_reduction: 0.05,
        };
        
        let reduction = result.error_reduction_percent(exact);
        
        // mitigated_error = |-1.15 - (-1.137)| = 0.013
        // original_error = |-1.1 - (-1.137)| = 0.037
        // reduction = (0.037 - 0.013) / 0.037 ≈ 64.9%
        assert!(reduction > 50.0);
    }
}

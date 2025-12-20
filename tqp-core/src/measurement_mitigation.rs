//! Measurement Error Mitigation (MEM)
//!
//! Corrects readout errors by:
//! 1. Building calibration matrix from basis state measurements
//! 2. Applying inverse calibration to correct noisy counts
//!
//! Reference: Qiskit Ignis readout error mitigation

use crate::noise_model::{NoiseModel, NoiseConfig, ShotSampler, ShotResult};
use std::collections::HashMap;

// =============================================================================
// Calibration Matrix
// =============================================================================

/// Calibration matrix for measurement error mitigation
#[derive(Debug, Clone)]
pub struct CalibrationMatrix {
    /// Number of qubits
    n_qubits: usize,
    /// Response matrix M[i][j] = P(measure i | prepared j)
    response_matrix: Vec<Vec<f64>>,
    /// Inverse (or pseudo-inverse) of response matrix
    inverse_matrix: Option<Vec<Vec<f64>>>,
    /// Whether this is a tensor product approximation
    is_tensor_product: bool,
    /// Single-qubit calibration matrices (for tensor product)
    single_qubit_matrices: Option<Vec<[[f64; 2]; 2]>>,
}

impl CalibrationMatrix {
    /// Create empty calibration matrix
    pub fn new(n_qubits: usize) -> Self {
        let dim = 1 << n_qubits;
        Self {
            n_qubits,
            response_matrix: vec![vec![0.0; dim]; dim],
            inverse_matrix: None,
            is_tensor_product: false,
            single_qubit_matrices: None,
        }
    }

    /// Build full calibration matrix from measurements
    /// 
    /// For each basis state |j⟩, prepare and measure n_shots times
    /// M[i][j] = (count of outcome i when prepared j) / n_shots
    pub fn build_full(
        n_qubits: usize,
        noise_model: &mut NoiseModel,
        n_shots: usize,
        seed: Option<u64>,
    ) -> Self {
        let dim = 1 << n_qubits;
        let mut matrix = vec![vec![0.0; dim]; dim];
        let mut sampler = ShotSampler::new(seed);

        for j in 0..dim {
            // Prepare basis state |j⟩ (ideal probability = 1 at index j)
            let mut probs = vec![0.0; dim];
            probs[j] = 1.0;

            // Apply readout error
            let noisy_probs = noise_model.apply_readout_error(&probs, n_qubits);

            // Sample
            let shots = sampler.sample(&noisy_probs, n_shots, n_qubits);

            // Record probabilities
            for i in 0..dim {
                matrix[i][j] = shots.get_probability(i);
            }
        }

        let mut cal = Self {
            n_qubits,
            response_matrix: matrix,
            inverse_matrix: None,
            is_tensor_product: false,
            single_qubit_matrices: None,
        };

        cal.compute_inverse();
        cal
    }

    /// Build tensor product calibration (for large qubit counts)
    /// 
    /// Assumes independent readout errors per qubit
    /// M_total = M_0 ⊗ M_1 ⊗ ... ⊗ M_{n-1}
    pub fn build_tensor_product(
        n_qubits: usize,
        noise_model: &mut NoiseModel,
        n_shots: usize,
        seed: Option<u64>,
    ) -> Self {
        let mut single_matrices = Vec::new();
        let mut sampler = ShotSampler::new(seed);

        for _q in 0..n_qubits {
            // Calibrate single qubit
            let mut m = [[0.0; 2]; 2];

            // Prepare |0⟩
            let probs_0 = vec![1.0, 0.0];
            let noisy_0 = noise_model.apply_readout_error(&probs_0, 1);
            let shots_0 = sampler.sample(&noisy_0, n_shots, 1);
            m[0][0] = shots_0.get_probability(0);
            m[1][0] = shots_0.get_probability(1);

            // Prepare |1⟩
            let probs_1 = vec![0.0, 1.0];
            let noisy_1 = noise_model.apply_readout_error(&probs_1, 1);
            let shots_1 = sampler.sample(&noisy_1, n_shots, 1);
            m[0][1] = shots_1.get_probability(0);
            m[1][1] = shots_1.get_probability(1);

            single_matrices.push(m);
        }

        // Build full matrix from tensor product
        let dim = 1 << n_qubits;
        let mut matrix = vec![vec![0.0; dim]; dim];

        for i in 0..dim {
            for j in 0..dim {
                let mut prob = 1.0;
                for q in 0..n_qubits {
                    let i_bit = (i >> q) & 1;
                    let j_bit = (j >> q) & 1;
                    prob *= single_matrices[q][i_bit][j_bit];
                }
                matrix[i][j] = prob;
            }
        }

        let mut cal = Self {
            n_qubits,
            response_matrix: matrix,
            inverse_matrix: None,
            is_tensor_product: true,
            single_qubit_matrices: Some(single_matrices),
        };

        cal.compute_inverse();
        cal
    }

    /// Compute inverse (or pseudo-inverse) of response matrix
    fn compute_inverse(&mut self) {
        let dim = 1 << self.n_qubits;
        
        // For small matrices, use direct inversion
        // For larger or ill-conditioned, use pseudo-inverse
        if self.is_tensor_product && self.single_qubit_matrices.is_some() {
            // Invert each single-qubit matrix and tensor product
            self.inverse_matrix = Some(self.compute_tensor_inverse());
        } else {
            self.inverse_matrix = Some(self.compute_direct_inverse());
        }
    }

    /// Direct matrix inversion using Gauss-Jordan
    fn compute_direct_inverse(&self) -> Vec<Vec<f64>> {
        let dim = self.response_matrix.len();
        
        // Augmented matrix [A | I]
        let mut aug: Vec<Vec<f64>> = self.response_matrix.iter()
            .enumerate()
            .map(|(i, row)| {
                let mut new_row = row.clone();
                new_row.extend(vec![0.0; dim]);
                new_row[dim + i] = 1.0;
                new_row
            })
            .collect();

        // Gauss-Jordan elimination
        for i in 0..dim {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..dim {
                if aug[k][i].abs() > aug[max_row][i].abs() {
                    max_row = k;
                }
            }
            aug.swap(i, max_row);

            let pivot = aug[i][i];
            if pivot.abs() < 1e-15 {
                continue; // Singular, skip
            }

            // Scale pivot row
            for j in 0..(2 * dim) {
                aug[i][j] /= pivot;
            }

            // Eliminate column
            for k in 0..dim {
                if k != i {
                    let factor = aug[k][i];
                    for j in 0..(2 * dim) {
                        aug[k][j] -= factor * aug[i][j];
                    }
                }
            }
        }

        // Extract inverse
        aug.iter()
            .map(|row| row[dim..].to_vec())
            .collect()
    }

    /// Compute inverse via tensor product of single-qubit inverses
    fn compute_tensor_inverse(&self) -> Vec<Vec<f64>> {
        let single_matrices = self.single_qubit_matrices.as_ref().unwrap();
        let n_qubits = self.n_qubits;
        let dim = 1 << n_qubits;

        // Invert each 2x2 matrix
        let single_inverses: Vec<[[f64; 2]; 2]> = single_matrices.iter()
            .map(|m| self.invert_2x2(m))
            .collect();

        // Build tensor product
        let mut inverse = vec![vec![0.0; dim]; dim];

        for i in 0..dim {
            for j in 0..dim {
                let mut val = 1.0;
                for q in 0..n_qubits {
                    let i_bit = (i >> q) & 1;
                    let j_bit = (j >> q) & 1;
                    val *= single_inverses[q][i_bit][j_bit];
                }
                inverse[i][j] = val;
            }
        }

        inverse
    }

    /// Invert 2x2 matrix
    fn invert_2x2(&self, m: &[[f64; 2]; 2]) -> [[f64; 2]; 2] {
        let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
        if det.abs() < 1e-15 {
            // Return identity if singular
            return [[1.0, 0.0], [0.0, 1.0]];
        }
        [
            [m[1][1] / det, -m[0][1] / det],
            [-m[1][0] / det, m[0][0] / det],
        ]
    }

    /// Get response matrix
    pub fn response_matrix(&self) -> &Vec<Vec<f64>> {
        &self.response_matrix
    }

    /// Get inverse matrix
    pub fn inverse_matrix(&self) -> Option<&Vec<Vec<f64>>> {
        self.inverse_matrix.as_ref()
    }

    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    pub fn is_tensor_product(&self) -> bool {
        self.is_tensor_product
    }
}

// =============================================================================
// Measurement Corrector
// =============================================================================

/// Corrects measurement results using calibration matrix
#[derive(Debug)]
pub struct MeasurementCorrector {
    calibration: CalibrationMatrix,
}

impl MeasurementCorrector {
    pub fn new(calibration: CalibrationMatrix) -> Self {
        Self { calibration }
    }

    /// Build corrector from noise model
    pub fn from_noise_model(
        n_qubits: usize,
        noise_model: &mut NoiseModel,
        n_shots: usize,
        use_tensor_product: bool,
        seed: Option<u64>,
    ) -> Self {
        let calibration = if use_tensor_product || n_qubits > 4 {
            CalibrationMatrix::build_tensor_product(n_qubits, noise_model, n_shots, seed)
        } else {
            CalibrationMatrix::build_full(n_qubits, noise_model, n_shots, seed)
        };
        Self::new(calibration)
    }

    /// Correct shot counts using inverse calibration matrix
    /// 
    /// p_corrected = M^{-1} * p_noisy
    pub fn correct_counts(&self, shots: &ShotResult) -> CorrectedResult {
        let dim = 1 << self.calibration.n_qubits;
        
        // Convert counts to probability vector
        let noisy_probs = shots.to_probabilities();

        // Apply inverse
        let inverse = self.calibration.inverse_matrix()
            .expect("Calibration matrix not inverted");

        let mut corrected_probs = vec![0.0; dim];
        for i in 0..dim {
            for j in 0..dim {
                corrected_probs[i] += inverse[i][j] * noisy_probs[j];
            }
        }

        // Clip negatives and renormalize
        let clipped = self.clip_and_normalize(&corrected_probs);

        CorrectedResult {
            original_counts: shots.clone(),
            corrected_probs: clipped,
            raw_corrected_probs: corrected_probs,
            n_qubits: self.calibration.n_qubits,
        }
    }

    /// Correct probability vector directly
    pub fn correct_probabilities(&self, probs: &[f64]) -> Vec<f64> {
        let dim = probs.len();
        let inverse = self.calibration.inverse_matrix()
            .expect("Calibration matrix not inverted");

        let mut corrected = vec![0.0; dim];
        for i in 0..dim {
            for j in 0..dim {
                if j < probs.len() {
                    corrected[i] += inverse[i][j] * probs[j];
                }
            }
        }

        self.clip_and_normalize(&corrected)
    }

    /// Clip negative probabilities and renormalize
    fn clip_and_normalize(&self, probs: &[f64]) -> Vec<f64> {
        // Clip to [0, 1]
        let mut clipped: Vec<f64> = probs.iter()
            .map(|&p| p.max(0.0))
            .collect();

        // Normalize
        let sum: f64 = clipped.iter().sum();
        if sum > 1e-15 {
            for p in &mut clipped {
                *p /= sum;
            }
        }

        clipped
    }

    /// Get calibration matrix
    pub fn calibration(&self) -> &CalibrationMatrix {
        &self.calibration
    }
}

// =============================================================================
// Corrected Result
// =============================================================================

/// Result of measurement error correction
#[derive(Debug, Clone)]
pub struct CorrectedResult {
    /// Original shot counts
    pub original_counts: ShotResult,
    /// Corrected probability distribution (clipped and normalized)
    pub corrected_probs: Vec<f64>,
    /// Raw corrected probabilities (may have negatives)
    pub raw_corrected_probs: Vec<f64>,
    /// Number of qubits
    pub n_qubits: usize,
}

impl CorrectedResult {
    /// Get corrected probability for a state
    pub fn get_probability(&self, state: usize) -> f64 {
        self.corrected_probs.get(state).copied().unwrap_or(0.0)
    }

    /// Get most likely state after correction
    pub fn most_likely_state(&self) -> (usize, f64) {
        self.corrected_probs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, &p)| (i, p))
            .unwrap_or((0, 0.0))
    }

    /// Compute expectation value from corrected probabilities
    pub fn expectation(&self, observable_fn: impl Fn(usize) -> f64) -> f64 {
        self.corrected_probs.iter()
            .enumerate()
            .map(|(state, &prob)| observable_fn(state) * prob)
            .sum()
    }

    /// Compute Z expectation for a qubit
    pub fn z_expectation(&self, qubit: usize) -> f64 {
        self.expectation(|state| {
            let bit = (state >> qubit) & 1;
            if bit == 0 { 1.0 } else { -1.0 }
        })
    }

    /// Check if correction introduced significant negatives
    pub fn has_significant_negatives(&self) -> bool {
        self.raw_corrected_probs.iter().any(|&p| p < -0.01)
    }

    /// Get sum of negative probabilities (measure of correction quality)
    pub fn negative_mass(&self) -> f64 {
        self.raw_corrected_probs.iter()
            .filter(|&&p| p < 0.0)
            .map(|&p| p.abs())
            .sum()
    }
}

// =============================================================================
// Convenience Functions
// =============================================================================

/// Apply measurement error mitigation to shot results
pub fn mitigate_measurement_error(
    shots: &ShotResult,
    noise_model: &mut NoiseModel,
    calibration_shots: usize,
    seed: Option<u64>,
) -> CorrectedResult {
    let corrector = MeasurementCorrector::from_noise_model(
        shots.n_qubits,
        noise_model,
        calibration_shots,
        shots.n_qubits > 4,
        seed,
    );
    corrector.correct_counts(shots)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calibration_matrix_new() {
        let cal = CalibrationMatrix::new(2);
        assert_eq!(cal.n_qubits(), 2);
        assert_eq!(cal.response_matrix().len(), 4);
    }

    #[test]
    fn test_calibration_matrix_build_full() {
        let mut noise = NoiseModel::new(
            NoiseConfig::default()
                .with_readout_error(0.1, 0.1)
                .with_seed(42)
        );
        
        let cal = CalibrationMatrix::build_full(2, &mut noise, 10000, Some(42));
        
        // Check diagonal dominance
        for i in 0..4 {
            assert!(cal.response_matrix()[i][i] > 0.5,
                "Diagonal element [{},{}] = {} too low", 
                i, i, cal.response_matrix()[i][i]);
        }
    }

    #[test]
    fn test_calibration_matrix_tensor_product() {
        let mut noise = NoiseModel::new(
            NoiseConfig::default()
                .with_readout_error(0.1, 0.1)
                .with_seed(42)
        );
        
        let cal = CalibrationMatrix::build_tensor_product(2, &mut noise, 10000, Some(42));
        
        assert!(cal.is_tensor_product());
        assert!(cal.inverse_matrix().is_some());
    }

    #[test]
    fn test_calibration_inverse_identity() {
        // With no noise, calibration should be identity
        let mut noise = NoiseModel::ideal();
        let cal = CalibrationMatrix::build_full(2, &mut noise, 1000, Some(42));
        
        let inv = cal.inverse_matrix().unwrap();
        
        // Check approximate identity
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((inv[i][j] - expected).abs() < 0.1,
                    "Inverse[{},{}] = {} != {}", i, j, inv[i][j], expected);
            }
        }
    }

    #[test]
    fn test_measurement_corrector_creation() {
        let mut noise = NoiseModel::new(NoiseConfig::default().with_seed(42));
        let corrector = MeasurementCorrector::from_noise_model(
            2, &mut noise, 1000, false, Some(42)
        );
        
        assert_eq!(corrector.calibration().n_qubits(), 2);
    }

    #[test]
    fn test_correct_counts() {
        let mut noise = NoiseModel::new(
            NoiseConfig::default()
                .with_readout_error(0.1, 0.1)
                .with_seed(42)
        );
        
        let corrector = MeasurementCorrector::from_noise_model(
            2, &mut noise, 5000, false, Some(42)
        );

        // Create noisy shot result
        let mut shots = ShotResult::new(2, 1000);
        shots.counts.insert(0, 800);  // |00⟩
        shots.counts.insert(1, 100);  // |01⟩
        shots.counts.insert(2, 50);   // |10⟩
        shots.counts.insert(3, 50);   // |11⟩

        let result = corrector.correct_counts(&shots);
        
        // Check probabilities sum to 1
        let sum: f64 = result.corrected_probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        
        // Check non-negative
        for &p in &result.corrected_probs {
            assert!(p >= 0.0);
        }
    }

    #[test]
    fn test_corrected_result_z_expectation() {
        let mut result = CorrectedResult {
            original_counts: ShotResult::new(1, 100),
            corrected_probs: vec![0.8, 0.2],  // 80% |0⟩, 20% |1⟩
            raw_corrected_probs: vec![0.8, 0.2],
            n_qubits: 1,
        };

        let z_exp = result.z_expectation(0);
        // 0.8 * 1 + 0.2 * (-1) = 0.6
        assert!((z_exp - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_corrected_result_most_likely() {
        let result = CorrectedResult {
            original_counts: ShotResult::new(2, 100),
            corrected_probs: vec![0.1, 0.6, 0.2, 0.1],
            raw_corrected_probs: vec![0.1, 0.6, 0.2, 0.1],
            n_qubits: 2,
        };

        let (state, prob) = result.most_likely_state();
        assert_eq!(state, 1);  // |01⟩
        assert!((prob - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_negative_handling() {
        let result = CorrectedResult {
            original_counts: ShotResult::new(2, 100),
            corrected_probs: vec![0.5, 0.5, 0.0, 0.0],  // After clipping
            raw_corrected_probs: vec![0.6, 0.5, -0.05, -0.05],  // Before clipping
            n_qubits: 2,
        };

        assert!(result.has_significant_negatives());
        assert!((result.negative_mass() - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_clip_and_normalize() {
        let mut noise = NoiseModel::ideal();
        let corrector = MeasurementCorrector::from_noise_model(
            1, &mut noise, 100, false, Some(42)
        );

        // Test with negatives
        let probs = vec![-0.1, 1.1];
        let clipped = corrector.clip_and_normalize(&probs);
        
        assert!(clipped[0] >= 0.0);
        assert!(clipped[1] >= 0.0);
        let sum: f64 = clipped.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_invert_2x2() {
        let cal = CalibrationMatrix::new(1);
        
        // Test known inverse
        let m = [[0.9, 0.1], [0.1, 0.9]];
        let inv = cal.invert_2x2(&m);
        
        // M * M^-1 should be identity
        let prod = [
            [m[0][0] * inv[0][0] + m[0][1] * inv[1][0],
             m[0][0] * inv[0][1] + m[0][1] * inv[1][1]],
            [m[1][0] * inv[0][0] + m[1][1] * inv[1][0],
             m[1][0] * inv[0][1] + m[1][1] * inv[1][1]],
        ];
        
        assert!((prod[0][0] - 1.0).abs() < 1e-10);
        assert!((prod[1][1] - 1.0).abs() < 1e-10);
        assert!(prod[0][1].abs() < 1e-10);
        assert!(prod[1][0].abs() < 1e-10);
    }

    #[test]
    fn test_mitigate_measurement_error_convenience() {
        let mut noise = NoiseModel::new(
            NoiseConfig::default()
                .with_readout_error(0.05, 0.05)
                .with_seed(42)
        );

        let mut shots = ShotResult::new(2, 1000);
        shots.counts.insert(0, 900);
        shots.counts.insert(3, 100);

        let result = mitigate_measurement_error(&shots, &mut noise, 5000, Some(42));
        
        // Should get valid probability distribution
        let sum: f64 = result.corrected_probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }
}

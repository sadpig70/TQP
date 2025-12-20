//! Noise Model and Shot-based Simulation for Error Mitigation
//!
//! Provides noise channels and shot-based measurement simulation
//! as prerequisites for Zero-Noise Extrapolation (ZNE) and
//! Measurement Error Mitigation (MEM).

use crate::state::TQPState;
use num_complex::Complex64;
use rand::prelude::*;
use rand::SeedableRng;
use std::collections::HashMap;

// =============================================================================
// Noise Model Configuration
// =============================================================================

/// Configuration for noise simulation
#[derive(Debug, Clone)]
pub struct NoiseConfig {
    /// Single-qubit depolarizing error rate
    pub depolarizing_rate: f64,
    /// Two-qubit gate error rate
    pub two_qubit_rate: f64,
    /// Readout error: P(1|0) - probability of measuring 1 when state is 0
    pub readout_p0: f64,
    /// Readout error: P(0|1) - probability of measuring 0 when state is 1
    pub readout_p1: f64,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for NoiseConfig {
    fn default() -> Self {
        Self {
            depolarizing_rate: 0.01,  // 1% single-qubit error
            two_qubit_rate: 0.02,     // 2% two-qubit error
            readout_p0: 0.02,         // 2% readout error |0⟩→|1⟩
            readout_p1: 0.02,         // 2% readout error |1⟩→|0⟩
            seed: None,
        }
    }
}

impl NoiseConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_depolarizing(mut self, rate: f64) -> Self {
        self.depolarizing_rate = rate.clamp(0.0, 1.0);
        self
    }

    pub fn with_two_qubit_rate(mut self, rate: f64) -> Self {
        self.two_qubit_rate = rate.clamp(0.0, 1.0);
        self
    }

    pub fn with_readout_error(mut self, p0: f64, p1: f64) -> Self {
        self.readout_p0 = p0.clamp(0.0, 1.0);
        self.readout_p1 = p1.clamp(0.0, 1.0);
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Create a noise-free configuration
    pub fn ideal() -> Self {
        Self {
            depolarizing_rate: 0.0,
            two_qubit_rate: 0.0,
            readout_p0: 0.0,
            readout_p1: 0.0,
            seed: None,
        }
    }

    /// Create IBM-like noise model
    pub fn ibm_like() -> Self {
        Self {
            depolarizing_rate: 0.001,  // ~0.1% single-qubit
            two_qubit_rate: 0.01,      // ~1% two-qubit
            readout_p0: 0.015,         // ~1.5% readout
            readout_p1: 0.015,
            seed: None,
        }
    }

    /// Create high-noise model for testing
    pub fn high_noise() -> Self {
        Self {
            depolarizing_rate: 0.05,   // 5% single-qubit
            two_qubit_rate: 0.10,      // 10% two-qubit
            readout_p0: 0.05,          // 5% readout
            readout_p1: 0.05,
            seed: None,
        }
    }
}

// =============================================================================
// Noise Model
// =============================================================================

/// Noise model for quantum circuit simulation
#[derive(Debug, Clone)]
pub struct NoiseModel {
    config: NoiseConfig,
    rng: StdRng,
}

impl NoiseModel {
    pub fn new(config: NoiseConfig) -> Self {
        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };
        Self { config, rng }
    }

    pub fn ideal() -> Self {
        Self::new(NoiseConfig::ideal())
    }

    pub fn config(&self) -> &NoiseConfig {
        &self.config
    }

    /// Apply depolarizing channel to a quantum state
    /// 
    /// Depolarizing channel: ρ → (1-p)ρ + p/3 (XρX + YρY + ZρZ)
    /// For pure states, this mixes toward maximally mixed state
    pub fn apply_depolarizing(&mut self, state: &TQPState, qubit: usize) -> TQPState {
        let p = self.config.depolarizing_rate;
        if p < 1e-10 {
            return state.clone();
        }

        let n_qubits = state.n_qubits();
        if qubit >= n_qubits {
            return state.clone();
        }

        // Get probability distribution
        let probs: Vec<f64> = state.state_vector.iter()
            .map(|a| a.norm_sqr())
            .collect();

        // Apply depolarizing: mix with random Pauli
        let roll: f64 = self.rng.gen();
        
        if roll > p {
            // No error
            return state.clone();
        }

        // Apply random Pauli (X, Y, or Z with equal probability)
        let pauli_choice: f64 = self.rng.gen();
        let mut new_amplitudes = state.state_vector.clone();
        let dim = 1 << n_qubits;

        for i in 0..dim {
            let bit = (i >> qubit) & 1;
            let flipped = i ^ (1 << qubit);

            if pauli_choice < 1.0 / 3.0 {
                // X gate: swap amplitudes
                if bit == 0 {
                    let temp = new_amplitudes[i];
                    new_amplitudes[i] = new_amplitudes[flipped];
                    new_amplitudes[flipped] = temp;
                }
            } else if pauli_choice < 2.0 / 3.0 {
                // Y gate: swap with phase
                if bit == 0 {
                    let temp = new_amplitudes[i];
                    new_amplitudes[i] = Complex64::new(0.0, 1.0) * new_amplitudes[flipped];
                    new_amplitudes[flipped] = Complex64::new(0.0, -1.0) * temp;
                }
            } else {
                // Z gate: phase flip
                if bit == 1 {
                    new_amplitudes[i] = -new_amplitudes[i];
                }
            }
        }

        TQPState::from_amplitudes(new_amplitudes.to_vec())
    }

    /// Apply depolarizing noise to all qubits after each gate layer
    pub fn apply_depolarizing_all(&mut self, state: &TQPState) -> TQPState {
        let mut result = state.clone();
        for q in 0..state.n_qubits() {
            result = self.apply_depolarizing(&result, q);
        }
        result
    }

    /// Apply readout error to probability distribution
    /// 
    /// For each qubit independently:
    /// P'(0) = (1-p0)*P(0) + p1*P(1)
    /// P'(1) = p0*P(0) + (1-p1)*P(1)
    pub fn apply_readout_error(&mut self, probs: &[f64], n_qubits: usize) -> Vec<f64> {
        let p0 = self.config.readout_p0;
        let p1 = self.config.readout_p1;

        if p0 < 1e-10 && p1 < 1e-10 {
            return probs.to_vec();
        }

        let dim = 1 << n_qubits;
        let mut noisy_probs = vec![0.0; dim];

        // For each basis state, compute contribution to noisy distribution
        for (i, &prob) in probs.iter().enumerate() {
            if prob < 1e-15 {
                continue;
            }

            // Generate all possible readout outcomes
            for outcome in 0..dim {
                let mut transition_prob = prob;
                
                for q in 0..n_qubits {
                    let true_bit = (i >> q) & 1;
                    let measured_bit = (outcome >> q) & 1;

                    if true_bit == 0 && measured_bit == 0 {
                        transition_prob *= 1.0 - p0;
                    } else if true_bit == 0 && measured_bit == 1 {
                        transition_prob *= p0;
                    } else if true_bit == 1 && measured_bit == 0 {
                        transition_prob *= p1;
                    } else {
                        transition_prob *= 1.0 - p1;
                    }
                }
                
                noisy_probs[outcome] += transition_prob;
            }
        }

        // Normalize
        let sum: f64 = noisy_probs.iter().sum();
        if sum > 1e-10 {
            for p in &mut noisy_probs {
                *p /= sum;
            }
        }

        noisy_probs
    }

    /// Get the readout error matrix for a single qubit
    /// Returns 2x2 matrix [[1-p0, p1], [p0, 1-p1]]
    pub fn single_qubit_readout_matrix(&self) -> [[f64; 2]; 2] {
        let p0 = self.config.readout_p0;
        let p1 = self.config.readout_p1;
        [
            [1.0 - p0, p1],
            [p0, 1.0 - p1],
        ]
    }

    /// Scale noise rates by a factor (for ZNE)
    pub fn scale_noise(&mut self, factor: f64) {
        self.config.depolarizing_rate = (self.config.depolarizing_rate * factor).clamp(0.0, 1.0);
        self.config.two_qubit_rate = (self.config.two_qubit_rate * factor).clamp(0.0, 1.0);
    }

    /// Create a scaled copy of the noise model
    pub fn scaled(&self, factor: f64) -> Self {
        let mut scaled_config = self.config.clone();
        scaled_config.depolarizing_rate = (scaled_config.depolarizing_rate * factor).clamp(0.0, 1.0);
        scaled_config.two_qubit_rate = (scaled_config.two_qubit_rate * factor).clamp(0.0, 1.0);
        Self::new(scaled_config)
    }
}

// =============================================================================
// Shot-based Simulation
// =============================================================================

/// Result of shot-based measurement
#[derive(Debug, Clone)]
pub struct ShotResult {
    /// Number of shots
    pub n_shots: usize,
    /// Counts for each basis state (bitstring -> count)
    pub counts: HashMap<usize, usize>,
    /// Number of qubits
    pub n_qubits: usize,
}

impl ShotResult {
    pub fn new(n_qubits: usize, n_shots: usize) -> Self {
        Self {
            n_shots,
            counts: HashMap::new(),
            n_qubits,
        }
    }

    /// Get count for a specific bitstring
    pub fn get_count(&self, bitstring: usize) -> usize {
        *self.counts.get(&bitstring).unwrap_or(&0)
    }

    /// Get probability for a specific bitstring
    pub fn get_probability(&self, bitstring: usize) -> f64 {
        self.get_count(bitstring) as f64 / self.n_shots as f64
    }

    /// Convert to probability vector
    pub fn to_probabilities(&self) -> Vec<f64> {
        let dim = 1 << self.n_qubits;
        let mut probs = vec![0.0; dim];
        for (&state, &count) in &self.counts {
            if state < dim {
                probs[state] = count as f64 / self.n_shots as f64;
            }
        }
        probs
    }

    /// Get most frequent outcome
    pub fn most_frequent(&self) -> Option<(usize, usize)> {
        self.counts.iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&state, &count)| (state, count))
    }

    /// Get all outcomes sorted by frequency
    pub fn sorted_outcomes(&self) -> Vec<(usize, usize)> {
        let mut outcomes: Vec<_> = self.counts.iter()
            .map(|(&s, &c)| (s, c))
            .collect();
        outcomes.sort_by(|a, b| b.1.cmp(&a.1));
        outcomes
    }

    /// Bitstring to string representation
    pub fn bitstring_to_string(&self, bitstring: usize) -> String {
        (0..self.n_qubits)
            .rev()
            .map(|i| if (bitstring >> i) & 1 == 1 { '1' } else { '0' })
            .collect()
    }

    /// Print summary
    pub fn summary(&self) -> String {
        let sorted = self.sorted_outcomes();
        let top_5: Vec<_> = sorted.iter().take(5).collect();
        
        let mut s = format!("ShotResult: {} shots, {} qubits\n", self.n_shots, self.n_qubits);
        s.push_str("Top outcomes:\n");
        for (state, count) in top_5 {
            let prob = *count as f64 / self.n_shots as f64;
            s.push_str(&format!("  |{}⟩: {} ({:.2}%)\n", 
                self.bitstring_to_string(*state), count, prob * 100.0));
        }
        s
    }
}

/// Shot-based measurement sampler
#[derive(Debug)]
pub struct ShotSampler {
    rng: StdRng,
}

impl ShotSampler {
    pub fn new(seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        Self { rng }
    }

    /// Sample a single measurement from probability distribution
    pub fn sample_once(&mut self, probs: &[f64]) -> usize {
        let r: f64 = self.rng.gen();
        let mut cumulative = 0.0;
        
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if r < cumulative {
                return i;
            }
        }
        
        probs.len() - 1
    }

    /// Sample multiple measurements
    pub fn sample(&mut self, probs: &[f64], n_shots: usize, n_qubits: usize) -> ShotResult {
        let mut result = ShotResult::new(n_qubits, n_shots);
        
        for _ in 0..n_shots {
            let outcome = self.sample_once(probs);
            *result.counts.entry(outcome).or_insert(0) += 1;
        }
        
        result
    }

    /// Sample from quantum state
    pub fn sample_state(&mut self, state: &TQPState, n_shots: usize) -> ShotResult {
        let probs: Vec<f64> = state.state_vector.iter()
            .map(|a| a.norm_sqr())
            .collect();
        self.sample(&probs, n_shots, state.n_qubits())
    }

    /// Sample with readout error
    pub fn sample_with_noise(
        &mut self,
        state: &TQPState,
        n_shots: usize,
        noise: &mut NoiseModel,
    ) -> ShotResult {
        // Get ideal probabilities
        let ideal_probs: Vec<f64> = state.state_vector.iter()
            .map(|a| a.norm_sqr())
            .collect();
        
        // Apply readout error
        let noisy_probs = noise.apply_readout_error(&ideal_probs, state.n_qubits());
        
        // Sample from noisy distribution
        self.sample(&noisy_probs, n_shots, state.n_qubits())
    }
}

// =============================================================================
// Expectation Value Computation
// =============================================================================

/// Compute expectation value from shot results
pub fn expectation_from_shots(
    shots: &ShotResult,
    observable_fn: impl Fn(usize) -> f64,
) -> f64 {
    let mut total = 0.0;
    for (&state, &count) in &shots.counts {
        total += observable_fn(state) * count as f64;
    }
    total / shots.n_shots as f64
}

/// Compute Z expectation for a single qubit from shots
pub fn z_expectation_from_shots(shots: &ShotResult, qubit: usize) -> f64 {
    expectation_from_shots(shots, |state| {
        let bit = (state >> qubit) & 1;
        if bit == 0 { 1.0 } else { -1.0 }
    })
}

/// Compute ZZ expectation for two qubits from shots
pub fn zz_expectation_from_shots(shots: &ShotResult, q1: usize, q2: usize) -> f64 {
    expectation_from_shots(shots, |state| {
        let b1 = (state >> q1) & 1;
        let b2 = (state >> q2) & 1;
        let parity = b1 ^ b2;
        if parity == 0 { 1.0 } else { -1.0 }
    })
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_config_default() {
        let config = NoiseConfig::default();
        assert!((config.depolarizing_rate - 0.01).abs() < 1e-10);
        assert!((config.readout_p0 - 0.02).abs() < 1e-10);
    }

    #[test]
    fn test_noise_config_builder() {
        let config = NoiseConfig::new()
            .with_depolarizing(0.05)
            .with_readout_error(0.03, 0.04)
            .with_seed(42);

        assert!((config.depolarizing_rate - 0.05).abs() < 1e-10);
        assert!((config.readout_p0 - 0.03).abs() < 1e-10);
        assert!((config.readout_p1 - 0.04).abs() < 1e-10);
        assert_eq!(config.seed, Some(42));
    }

    #[test]
    fn test_noise_config_ideal() {
        let config = NoiseConfig::ideal();
        assert!(config.depolarizing_rate < 1e-10);
        assert!(config.readout_p0 < 1e-10);
    }

    #[test]
    fn test_noise_model_creation() {
        let noise = NoiseModel::new(NoiseConfig::default());
        assert!((noise.config().depolarizing_rate - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_noise_model_ideal() {
        let noise = NoiseModel::ideal();
        assert!(noise.config().depolarizing_rate < 1e-10);
    }

    #[test]
    fn test_readout_error_matrix() {
        let noise = NoiseModel::new(NoiseConfig::default().with_readout_error(0.1, 0.2));
        let matrix = noise.single_qubit_readout_matrix();
        
        assert!((matrix[0][0] - 0.9).abs() < 1e-10);  // 1 - p0
        assert!((matrix[0][1] - 0.2).abs() < 1e-10);  // p1
        assert!((matrix[1][0] - 0.1).abs() < 1e-10);  // p0
        assert!((matrix[1][1] - 0.8).abs() < 1e-10);  // 1 - p1
    }

    #[test]
    fn test_apply_readout_error() {
        let mut noise = NoiseModel::new(
            NoiseConfig::default()
                .with_readout_error(0.1, 0.1)
                .with_seed(42)
        );
        
        // Pure |0⟩ state
        let probs = vec![1.0, 0.0];
        let noisy = noise.apply_readout_error(&probs, 1);
        
        // Should be [0.9, 0.1] approximately
        assert!((noisy[0] - 0.9).abs() < 1e-10);
        assert!((noisy[1] - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_noise_scaling() {
        let noise = NoiseModel::new(NoiseConfig::default().with_depolarizing(0.01));
        let scaled = noise.scaled(3.0);
        
        assert!((scaled.config().depolarizing_rate - 0.03).abs() < 1e-10);
    }

    #[test]
    fn test_shot_result_creation() {
        let result = ShotResult::new(2, 1000);
        assert_eq!(result.n_shots, 1000);
        assert_eq!(result.n_qubits, 2);
        assert!(result.counts.is_empty());
    }

    #[test]
    fn test_shot_result_probabilities() {
        let mut result = ShotResult::new(2, 100);
        result.counts.insert(0, 60);  // |00⟩
        result.counts.insert(3, 40);  // |11⟩
        
        let probs = result.to_probabilities();
        assert!((probs[0] - 0.6).abs() < 1e-10);
        assert!((probs[3] - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_shot_result_most_frequent() {
        let mut result = ShotResult::new(2, 100);
        result.counts.insert(0, 60);
        result.counts.insert(1, 40);
        
        let (state, count) = result.most_frequent().unwrap();
        assert_eq!(state, 0);
        assert_eq!(count, 60);
    }

    #[test]
    fn test_shot_sampler_creation() {
        let sampler = ShotSampler::new(Some(42));
        // Just verify it doesn't panic
        assert!(true);
    }

    #[test]
    fn test_shot_sampler_sample() {
        let mut sampler = ShotSampler::new(Some(42));
        let probs = vec![0.5, 0.5];
        
        let result = sampler.sample(&probs, 1000, 1);
        assert_eq!(result.n_shots, 1000);
        
        // Both outcomes should be observed with reasonable frequency
        let count_0 = result.get_count(0);
        let count_1 = result.get_count(1);
        assert!(count_0 > 300 && count_0 < 700);
        assert!(count_1 > 300 && count_1 < 700);
    }

    #[test]
    fn test_shot_sampler_deterministic() {
        let mut s1 = ShotSampler::new(Some(42));
        let mut s2 = ShotSampler::new(Some(42));
        let probs = vec![0.3, 0.7];
        
        let r1 = s1.sample(&probs, 100, 1);
        let r2 = s2.sample(&probs, 100, 1);
        
        // Same seed should give same results
        assert_eq!(r1.counts, r2.counts);
    }

    #[test]
    fn test_z_expectation_from_shots() {
        let mut result = ShotResult::new(1, 100);
        result.counts.insert(0, 80);  // |0⟩ → Z = +1
        result.counts.insert(1, 20);  // |1⟩ → Z = -1
        
        let exp = z_expectation_from_shots(&result, 0);
        // 0.8 * 1 + 0.2 * (-1) = 0.6
        assert!((exp - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_zz_expectation_from_shots() {
        let mut result = ShotResult::new(2, 100);
        result.counts.insert(0, 50);  // |00⟩ → ZZ = +1
        result.counts.insert(3, 50);  // |11⟩ → ZZ = +1
        
        let exp = zz_expectation_from_shots(&result, 0, 1);
        // Both have parity 0, so ZZ = +1
        assert!((exp - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bitstring_to_string() {
        let result = ShotResult::new(3, 100);
        assert_eq!(result.bitstring_to_string(0), "000");
        assert_eq!(result.bitstring_to_string(5), "101");
        assert_eq!(result.bitstring_to_string(7), "111");
    }

    #[test]
    fn test_depolarizing_ideal() {
        let mut noise = NoiseModel::ideal();
        let state = TQPState::zero(2);
        
        let result = noise.apply_depolarizing(&state, 0);
        
        // No noise, state should be unchanged
        for i in 0..4 {
            let diff = (result.state_vector[i] - state.state_vector[i]).norm();
            assert!(diff < 1e-10);
        }
    }

    #[test]
    fn test_sample_state() {
        // Create |+⟩ state
        let mut state = TQPState::zero(1);
        state.state_vector[0] = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
        state.state_vector[1] = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
        
        let mut sampler = ShotSampler::new(Some(42));
        let result = sampler.sample_state(&state, 1000);
        
        // Should be roughly 50/50
        let p0 = result.get_probability(0);
        let p1 = result.get_probability(1);
        assert!((p0 - 0.5).abs() < 0.1);
        assert!((p1 - 0.5).abs() < 0.1);
    }
}

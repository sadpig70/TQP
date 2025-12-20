//! Readout Error Mitigation (MEM) for IBM Quantum hardware
//! 
//! Implements tensor-product noise model for efficient mitigation.

use std::collections::HashMap;

/// Single-qubit readout error rates
#[derive(Debug, Clone)]
pub struct QubitErrorRates {
    /// P(1|0) - probability of measuring 1 when prepared 0
    pub p_flip_01: f64,
    /// P(0|1) - probability of measuring 0 when prepared 1  
    pub p_flip_10: f64,
}

impl Default for QubitErrorRates {
    fn default() -> Self {
        // Typical IBM Quantum error rates
        Self {
            p_flip_01: 0.015,
            p_flip_10: 0.025,
        }
    }
}

/// Readout Error Mitigation engine
#[derive(Debug, Clone)]
pub struct ReadoutMitigation {
    /// Error rates per qubit
    qubit_errors: Vec<QubitErrorRates>,
    /// Number of qubits
    n_qubits: usize,
}

impl ReadoutMitigation {
    /// Create new MEM with default error rates
    pub fn new(n_qubits: usize) -> Self {
        Self {
            qubit_errors: vec![QubitErrorRates::default(); n_qubits],
            n_qubits,
        }
    }
    
    /// Create MEM with custom error rates
    pub fn with_error_rates(error_rates: Vec<QubitErrorRates>) -> Self {
        let n_qubits = error_rates.len();
        Self {
            qubit_errors: error_rates,
            n_qubits,
        }
    }
    
    /// Set error rates from calibration data
    /// 
    /// # Arguments
    /// * `calibration` - HashMap mapping prepared state to measured counts
    ///   e.g., {"00": {"00": 980, "01": 10, "10": 8, "11": 2}, ...}
    pub fn calibrate(&mut self, calibration: &HashMap<String, HashMap<String, u64>>) {
        // Extract single-qubit errors from calibration data
        for q in 0..self.n_qubits {
            // Look at |0...0⟩ and |1...1⟩ preparations
            let all_zeros = "0".repeat(self.n_qubits);
            let all_ones = "1".repeat(self.n_qubits);
            
            if let Some(counts) = calibration.get(&all_zeros) {
                let total: u64 = counts.values().sum();
                let mut flip_count = 0u64;
                for (state, &count) in counts {
                    let bit = state.chars().rev().nth(q).unwrap_or('0');
                    if bit == '1' {
                        flip_count += count;
                    }
                }
                self.qubit_errors[q].p_flip_01 = flip_count as f64 / total as f64;
            }
            
            if let Some(counts) = calibration.get(&all_ones) {
                let total: u64 = counts.values().sum();
                let mut flip_count = 0u64;
                for (state, &count) in counts {
                    let bit = state.chars().rev().nth(q).unwrap_or('1');
                    if bit == '0' {
                        flip_count += count;
                    }
                }
                self.qubit_errors[q].p_flip_10 = flip_count as f64 / total as f64;
            }
        }
    }
    
    /// Apply mitigation to measurement counts
    /// 
    /// Uses tensor-product inverse for efficiency (O(2^n) instead of O(4^n))
    pub fn mitigate(&self, counts: &HashMap<String, u64>) -> HashMap<String, f64> {
        let total: u64 = counts.values().sum();
        let n_states = 1 << self.n_qubits;
        
        // Convert counts to probability vector
        let mut probs = vec![0.0; n_states];
        for (state, &count) in counts {
            let idx = usize::from_str_radix(state, 2).unwrap_or(0);
            probs[idx] = count as f64 / total as f64;
        }
        
        // Apply tensor-product inverse
        // For each qubit, apply 2x2 inverse independently
        for q in 0..self.n_qubits {
            let err = &self.qubit_errors[q];
            
            // 2x2 confusion matrix for qubit q:
            // [[1-p01, p10], [p01, 1-p10]]
            // Inverse: 1/det * [[1-p10, -p10], [-p01, 1-p01]]
            let det = (1.0 - err.p_flip_01) * (1.0 - err.p_flip_10) 
                    - err.p_flip_01 * err.p_flip_10;
            
            if det.abs() < 1e-10 {
                continue; // Singular matrix, skip
            }
            
            let inv00 = (1.0 - err.p_flip_10) / det;
            let inv01 = -err.p_flip_10 / det;
            let inv10 = -err.p_flip_01 / det;
            let inv11 = (1.0 - err.p_flip_01) / det;
            
            // Apply to probability vector
            let mut new_probs = vec![0.0; n_states];
            for state in 0..n_states {
                let bit = (state >> q) & 1;
                let partner = state ^ (1 << q); // Flip bit q
                
                if bit == 0 {
                    new_probs[state] += inv00 * probs[state] + inv01 * probs[partner];
                } else {
                    new_probs[state] += inv10 * probs[partner] + inv11 * probs[state];
                }
            }
            probs = new_probs;
        }
        
        // Clip and normalize
        let sum: f64 = probs.iter().map(|p| p.max(0.0)).sum();
        let mut result = HashMap::new();
        for (idx, &prob) in probs.iter().enumerate() {
            let prob = prob.max(0.0) / sum;
            if prob > 1e-10 {
                let state = format!("{:0width$b}", idx, width = self.n_qubits);
                result.insert(state, prob * total as f64);
            }
        }
        
        result
    }
    
    /// Get current error rates
    pub fn error_rates(&self) -> &[QubitErrorRates] {
        &self.qubit_errors
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mitigation_improves_bell() {
        // Simulated Bell state with readout errors
        let mut counts = HashMap::new();
        counts.insert("00".to_string(), 485u64);
        counts.insert("01".to_string(), 22);
        counts.insert("10".to_string(), 33);
        counts.insert("11".to_string(), 460);
        
        let mem = ReadoutMitigation::with_error_rates(vec![
            QubitErrorRates { p_flip_01: 0.011, p_flip_10: 0.012 },
            QubitErrorRates { p_flip_01: 0.023, p_flip_10: 0.040 },
        ]);
        
        let mitigated = mem.mitigate(&counts);
        
        // Fidelity should improve
        let raw_fid = (counts["00"] + counts["11"]) as f64 / 1000.0;
        let mit_fid = (mitigated.get("00").unwrap_or(&0.0) 
                     + mitigated.get("11").unwrap_or(&0.0)) / 1000.0;
        
        assert!(mit_fid > raw_fid, "MEM should improve Bell fidelity");
    }
    
    #[test]
    fn test_perfect_readout_no_change() {
        let mut counts = HashMap::new();
        counts.insert("00".to_string(), 500u64);
        counts.insert("11".to_string(), 500);
        
        let mem = ReadoutMitigation::with_error_rates(vec![
            QubitErrorRates { p_flip_01: 0.0, p_flip_10: 0.0 },
            QubitErrorRates { p_flip_01: 0.0, p_flip_10: 0.0 },
        ]);
        
        let mitigated = mem.mitigate(&counts);
        
        let p00 = mitigated.get("00").unwrap_or(&0.0);
        let p11 = mitigated.get("11").unwrap_or(&0.0);
        
        assert!((p00 - 500.0).abs() < 1.0);
        assert!((p11 - 500.0).abs() < 1.0);
    }
}

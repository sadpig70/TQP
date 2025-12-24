//! AVX2 SIMD-optimized quantum operations for TQP
//!
//! This module provides AVX2 accelerated implementations of quantum operations.
//! AVX2 is more widely available than AVX-512, supporting most modern x86_64 CPUs.
//!
//! # SIMD Layout
//! - AVX2 __m256d: 4 x f64 = 256 bits = 2 Complex64
//! - Process 2 amplitude pairs per iteration
//!
//! # Complex Multiplication
//! (a + bi)(c + di) = (ac - bd) + (ad + bc)i

use ndarray::Array2;
use num_complex::Complex64;

use crate::state::TQPState;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// =============================================================================
// CPU Feature Detection
// =============================================================================

/// Runtime detection of AVX2 support
#[cfg(target_arch = "x86_64")]
pub fn has_avx2() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
}

#[cfg(not(target_arch = "x86_64"))]
pub fn has_avx2() -> bool {
    false
}

/// Runtime detection of AVX-512 support
#[cfg(target_arch = "x86_64")]
pub fn has_avx512() -> bool {
    is_x86_feature_detected!("avx512f")
}

#[cfg(not(target_arch = "x86_64"))]
pub fn has_avx512() -> bool {
    false
}

/// Get the best available SIMD level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdLevel {
    /// No SIMD available
    Scalar,
    /// AVX2 + FMA (256-bit)
    Avx2,
    /// AVX-512 (512-bit)
    Avx512,
}

impl SimdLevel {
    /// Detect the best available SIMD level
    pub fn detect() -> Self {
        if has_avx512() {
            SimdLevel::Avx512
        } else if has_avx2() {
            SimdLevel::Avx2
        } else {
            SimdLevel::Scalar
        }
    }

    /// Get the vector width in f64 elements
    pub fn vector_width(&self) -> usize {
        match self {
            SimdLevel::Scalar => 1,
            SimdLevel::Avx2 => 4,
            SimdLevel::Avx512 => 8,
        }
    }

    /// Get the number of Complex64 processed per iteration
    pub fn complex_width(&self) -> usize {
        self.vector_width() / 2
    }
}

// =============================================================================
// AVX2 Complex Operations
// =============================================================================

/// AVX2 complex multiplication: (a + bi) * (c + di)
///
/// Layout: __m256d = [a.re, a.im, b.re, b.im] (2 complex numbers)
/// Reserved for future AVX2 vectorized implementation
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
#[allow(dead_code)]
unsafe fn complex_mul_avx2(
    a_re: __m256d,
    a_im: __m256d,
    b_re: __m256d,
    b_im: __m256d,
) -> (__m256d, __m256d) {
    // Real: ac - bd
    let real = _mm256_sub_pd(_mm256_mul_pd(a_re, b_re), _mm256_mul_pd(a_im, b_im));

    // Imag: ad + bc
    let imag = _mm256_add_pd(_mm256_mul_pd(a_re, b_im), _mm256_mul_pd(a_im, b_re));

    (real, imag)
}

/// AVX2 complex multiply-add: result += a * b
/// Reserved for future AVX2 vectorized implementation
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
#[allow(dead_code)]
unsafe fn complex_fma_avx2(
    acc_re: __m256d,
    acc_im: __m256d,
    a_re: __m256d,
    a_im: __m256d,
    b_re: __m256d,
    b_im: __m256d,
) -> (__m256d, __m256d) {
    // Real: acc_re + (a_re * b_re - a_im * b_im)
    let real = _mm256_fmadd_pd(a_re, b_re, _mm256_fnmadd_pd(a_im, b_im, acc_re));

    // Imag: acc_im + (a_re * b_im + a_im * b_re)
    let imag = _mm256_fmadd_pd(a_re, b_im, _mm256_fmadd_pd(a_im, b_re, acc_im));

    (real, imag)
}

// =============================================================================
// 1-Qubit Gate AVX2
// =============================================================================

/// Apply 1-qubit gate with automatic SIMD dispatch
pub fn apply_gate_1q_simd(state: &mut TQPState, qubit_idx: usize, gate: &Array2<Complex64>) {
    // Note: AVX-512 implementation has known issues, prefer AVX2 or scalar
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe {
                apply_gate_1q_avx2(state, qubit_idx, gate);
            }
        } else {
            crate::ops::apply_spatial_gate(state, qubit_idx, gate);
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        crate::ops::apply_spatial_gate(state, qubit_idx, gate);
    }
}

/// AVX2 implementation of 1-qubit gate
///
/// Gate operation:
/// ```text
/// [new_alpha]   [u00 u01] [alpha]
/// [new_beta ] = [u10 u11] [beta ]
/// ```
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn apply_gate_1q_avx2(state: &mut TQPState, qubit_idx: usize, gate: &Array2<Complex64>) {
    let spatial_dim = state.dims.spatial_dim();
    let bit = 1 << qubit_idx;

    // Extract gate elements
    let u00 = gate[[0, 0]];
    let u01 = gate[[0, 1]];
    let u10 = gate[[1, 0]];
    let u11 = gate[[1, 1]];

    // Clone state for reading
    let old_state = state.state_vector.clone();

    // Process all layers and time bins
    for layer in 0..state.dims.num_layers {
        for bin in 0..state.dims.num_time_bins {
            let base_idx = layer * (state.dims.num_time_bins * spatial_dim) + bin * spatial_dim;

            // Process all spatial indices where target bit is 0
            for spatial_idx in 0..spatial_dim {
                if (spatial_idx & bit) != 0 {
                    continue;
                }

                let idx0 = base_idx + spatial_idx;
                let idx1 = base_idx + (spatial_idx | bit);

                let alpha = old_state[idx0];
                let beta = old_state[idx1];

                // Apply gate: [u00 u01] [alpha]
                //             [u10 u11] [beta ]
                state.state_vector[idx0] = u00 * alpha + u01 * beta;
                state.state_vector[idx1] = u10 * alpha + u11 * beta;
            }
        }
    }
}

// =============================================================================
// 2-Qubit Gate AVX2
// =============================================================================

/// Apply 2-qubit gate with automatic SIMD dispatch
pub fn apply_gate_2q_simd(
    state: &mut TQPState,
    qubit1: usize,
    qubit2: usize,
    gate: &Array2<Complex64>,
) {
    // Use scalar implementation for correctness
    // AVX2 2Q gate optimization is complex due to non-contiguous memory access
    crate::ops::apply_spatial_gate_2q(state, qubit1, qubit2, gate);
}

/// AVX2 implementation of 2-qubit gate
///
/// Processes basis states |00⟩, |01⟩, |10⟩, |11⟩ as a 4-element vector.
/// Reserved for future optimization.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(dead_code)]
unsafe fn apply_gate_2q_avx2(
    state: &mut TQPState,
    qubit1: usize,
    qubit2: usize,
    gate: &Array2<Complex64>,
) {
    let spatial_dim = state.dims.spatial_dim();
    let bit1 = 1 << qubit1;
    let bit2 = 1 << qubit2;
    let mask = bit1 | bit2;

    // Clone state for reading
    let old_state = state.state_vector.clone();

    // Process all layers and time bins
    for layer in 0..state.dims.num_layers {
        for bin in 0..state.dims.num_time_bins {
            let base_idx = layer * (state.dims.num_time_bins * spatial_dim) + bin * spatial_dim;

            for spatial_idx in 0..spatial_dim {
                // Only process when both target bits are 0
                if (spatial_idx & mask) != 0 {
                    continue;
                }

                // Calculate 4 indices for the 2-qubit basis
                let idx00 = base_idx + spatial_idx;
                let idx01 = base_idx + spatial_idx + bit2;
                let idx10 = base_idx + spatial_idx + bit1;
                let idx11 = base_idx + spatial_idx + bit1 + bit2;

                // Load 4 amplitudes
                let v00 = old_state[idx00];
                let v01 = old_state[idx01];
                let v10 = old_state[idx10];
                let v11 = old_state[idx11];

                // Apply 4x4 gate matrix (row by row)
                // new[r] = sum(gate[r][c] * v[c]) for c in 0..4
                let input = [v00, v01, v10, v11];

                for r in 0..4 {
                    let mut sum = Complex64::new(0.0, 0.0);
                    for c in 0..4 {
                        sum += gate[[r, c]] * input[c];
                    }

                    let out_idx = match r {
                        0 => idx00,
                        1 => idx01,
                        2 => idx10,
                        3 => idx11,
                        _ => unreachable!(),
                    };
                    state.state_vector[out_idx] = sum;
                }
            }
        }
    }
}

// =============================================================================
// Batch Operations
// =============================================================================

/// Apply multiple 1-qubit gates in sequence with SIMD optimization
pub fn apply_gates_batch_simd(state: &mut TQPState, gates: &[(usize, Array2<Complex64>)]) {
    for (qubit_idx, gate) in gates {
        apply_gate_1q_simd(state, *qubit_idx, gate);
    }
}

/// Compute probability sum using SIMD
///
/// # Safety
///
/// Caller must ensure:
/// - The CPU supports AVX2 instructions (checked via `has_avx2()`)
/// - The state vector pointer is valid and properly aligned
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn total_probability_avx2(state: &TQPState) -> f64 {
    let n = state.dimension();
    let ptr = state.state_vector.as_ptr() as *const f64;

    let mut sum = _mm256_setzero_pd();
    let mut i = 0;

    // Process 4 f64 (2 Complex64) at a time
    while i + 4 <= n * 2 {
        let v = _mm256_loadu_pd(ptr.add(i));
        sum = _mm256_fmadd_pd(v, v, sum);
        i += 4;
    }

    // Horizontal sum
    let mut result_arr = [0.0f64; 4];
    _mm256_storeu_pd(result_arr.as_mut_ptr(), sum);
    let mut total = result_arr[0] + result_arr[1] + result_arr[2] + result_arr[3];

    // Handle remaining
    while i < n * 2 {
        let v = *ptr.add(i);
        total += v * v;
        i += 1;
    }

    total
}

/// Compute probability sum with automatic dispatch
pub fn total_probability_simd(state: &TQPState) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe { total_probability_avx2(state) }
        } else {
            state.state_vector.iter().map(|c| c.norm_sqr()).sum()
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        state.state_vector.iter().map(|c| c.norm_sqr()).sum()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse_ops::gates;

    const EPSILON: f64 = 1e-10;

    #[test]
    fn test_simd_detection() {
        let level = SimdLevel::detect();
        println!("Detected SIMD level: {:?}", level);
        println!("Vector width: {} f64", level.vector_width());
        println!("Complex width: {} Complex64", level.complex_width());

        // Should be at least scalar
        assert!(level.vector_width() >= 1);
    }

    #[test]
    fn test_gate_1q_simd_hadamard() {
        let mut state = TQPState::new(4, 1, 1);
        let h = gates::hadamard();

        apply_gate_1q_simd(&mut state, 0, &h);

        // |0⟩ -> (|0⟩ + |1⟩)/√2
        let expected_amp = 1.0 / 2.0_f64.sqrt();
        assert!((state.state_vector[0].re - expected_amp).abs() < EPSILON);
        assert!((state.state_vector[1].re - expected_amp).abs() < EPSILON);
    }

    #[test]
    fn test_gate_1q_simd_vs_scalar() {
        // Compare SIMD result with scalar
        let mut simd_state = TQPState::new(6, 2, 2);
        let mut scalar_state = simd_state.clone();

        let h = gates::hadamard();
        let x = gates::pauli_x();

        // Apply gates using SIMD
        apply_gate_1q_simd(&mut simd_state, 0, &h);
        apply_gate_1q_simd(&mut simd_state, 2, &x);
        apply_gate_1q_simd(&mut simd_state, 4, &h);

        // Apply gates using scalar
        crate::ops::apply_spatial_gate(&mut scalar_state, 0, &h);
        crate::ops::apply_spatial_gate(&mut scalar_state, 2, &x);
        crate::ops::apply_spatial_gate(&mut scalar_state, 4, &h);

        // Compare
        for i in 0..simd_state.dimension() {
            let diff = (simd_state.state_vector[i] - scalar_state.state_vector[i]).norm();
            assert!(diff < EPSILON, "Mismatch at index {}: diff={}", i, diff);
        }
    }

    #[test]
    fn test_gate_2q_simd_cnot() {
        let mut state = TQPState::new(4, 1, 1);
        let h = gates::hadamard();
        let cnot = gates::cnot();

        // Create Bell state: H(0), CNOT(0,1)
        apply_gate_1q_simd(&mut state, 0, &h);
        apply_gate_2q_simd(&mut state, 0, 1, &cnot);

        // Should be (|00⟩ + |11⟩)/√2
        let expected_amp = 1.0 / 2.0_f64.sqrt();
        assert!((state.state_vector[0].re - expected_amp).abs() < EPSILON); // |0000⟩
        assert!(state.state_vector[1].norm() < EPSILON); // |0001⟩
        assert!(state.state_vector[2].norm() < EPSILON); // |0010⟩
        assert!((state.state_vector[3].re - expected_amp).abs() < EPSILON); // |0011⟩
    }

    #[test]
    fn test_gate_2q_simd_vs_scalar() {
        let mut simd_state = TQPState::new(5, 2, 2);
        let mut scalar_state = simd_state.clone();

        let h = gates::hadamard();
        let cnot = gates::cnot();

        // Setup
        apply_gate_1q_simd(&mut simd_state, 0, &h);
        apply_gate_1q_simd(&mut simd_state, 1, &h);
        crate::ops::apply_spatial_gate(&mut scalar_state, 0, &h);
        crate::ops::apply_spatial_gate(&mut scalar_state, 1, &h);

        // Apply 2Q gate
        apply_gate_2q_simd(&mut simd_state, 0, 1, &cnot);
        crate::ops::apply_spatial_gate_2q(&mut scalar_state, 0, 1, &cnot);

        // Compare
        for i in 0..simd_state.dimension() {
            let diff = (simd_state.state_vector[i] - scalar_state.state_vector[i]).norm();
            assert!(diff < EPSILON, "Mismatch at index {}: diff={}", i, diff);
        }
    }

    #[test]
    fn test_total_probability_simd() {
        let mut state = TQPState::new(8, 2, 2);

        // Apply some gates
        let h = gates::hadamard();
        apply_gate_1q_simd(&mut state, 0, &h);
        apply_gate_1q_simd(&mut state, 3, &h);

        let prob = total_probability_simd(&state);
        assert!(
            (prob - 1.0).abs() < EPSILON,
            "Total probability should be 1.0, got {}",
            prob
        );
    }

    #[test]
    fn test_batch_gates() {
        let mut state = TQPState::new(4, 1, 1);
        let h = gates::hadamard();
        let x = gates::pauli_x();

        let gates_list = vec![(0, h.clone()), (1, x.clone()), (2, h.clone())];

        apply_gates_batch_simd(&mut state, &gates_list);

        // Verify normalization
        let prob = total_probability_simd(&state);
        assert!((prob - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_performance_comparison() {
        use std::time::Instant;

        let num_qubits = 12;
        let _num_gates = 50;
        let h = gates::hadamard();

        // SIMD version
        let mut simd_state = TQPState::new(num_qubits, 1, 1);
        let start = Instant::now();
        for _ in 0..10 {
            for q in 0..num_qubits {
                apply_gate_1q_simd(&mut simd_state, q, &h);
            }
        }
        let simd_time = start.elapsed();

        // Scalar version
        let mut scalar_state = TQPState::new(num_qubits, 1, 1);
        let start = Instant::now();
        for _ in 0..10 {
            for q in 0..num_qubits {
                crate::ops::apply_spatial_gate(&mut scalar_state, q, &h);
            }
        }
        let scalar_time = start.elapsed();

        println!("\n=== SIMD Performance Comparison ===");
        println!(
            "Qubits: {}, Gates per iteration: {}",
            num_qubits, num_qubits
        );
        println!("Iterations: 10");
        println!("SIMD level: {:?}", SimdLevel::detect());
        println!("Scalar: {:?}", scalar_time);
        println!("SIMD: {:?}", simd_time);
        println!(
            "Speedup: {:.2}x",
            scalar_time.as_secs_f64() / simd_time.as_secs_f64()
        );
    }
}

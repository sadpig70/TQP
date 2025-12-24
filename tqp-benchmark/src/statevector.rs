//! 상태벡터 벤치마크 모듈
//!
//! N-qubit 상태벡터 연산의 성능을 측정합니다.

use crate::{BenchmarkConfig, BenchmarkResult, Measurement};
use std::time::Instant;

/// 상태벡터 벤치마크 실행
pub fn run_statevector_benchmark(config: &BenchmarkConfig) -> BenchmarkResult {
    let mut result = BenchmarkResult::new("statevector_operations", config.clone());

    for n_qubits in config.min_qubits..=config.max_qubits {
        // 워밍업
        for _ in 0..config.warmup_runs {
            run_single_statevector_test(n_qubits, 1, 1);
        }

        // 실제 측정
        for _ in 0..config.trials {
            let start = Instant::now();
            run_single_statevector_test(n_qubits, 1, 1);
            let duration = start.elapsed();

            result.add_measurement(Measurement {
                n_qubits,
                time_bins: 1,
                layers: 1,
                duration_ns: duration.as_nanos() as u64,
                peak_memory_bytes: get_memory_usage(),
                metadata: None,
            });
        }
    }

    result.compute_summary();
    result
}

/// 단일 상태벡터 테스트 실행
fn run_single_statevector_test(n_qubits: usize, time_bins: usize, layers: usize) {
    use ndarray::Array2;
    use num_complex::Complex64;
    use tqp_core::state::TQPState;

    // Hadamard 게이트 행렬
    let sqrt2_inv = 1.0 / 2.0_f64.sqrt();
    let h_gate = Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(sqrt2_inv, 0.0),
            Complex64::new(sqrt2_inv, 0.0),
            Complex64::new(sqrt2_inv, 0.0),
            Complex64::new(-sqrt2_inv, 0.0),
        ],
    )
    .unwrap();

    // CNOT 게이트 행렬 (4x4)
    let cnot_gate = Array2::from_shape_vec(
        (4, 4),
        vec![
            // |00> -> |00>
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            // |01> -> |01>
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            // |10> -> |11>
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            // |11> -> |10>
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    )
    .unwrap();

    // 상태 생성
    let mut state = TQPState::new(n_qubits, time_bins, layers);

    // 기본 게이트 연산 (H 체인)
    for i in 0..n_qubits {
        tqp_core::ops::apply_spatial_gate(&mut state, i, &h_gate);
    }

    // CNOT 체인
    for i in 0..(n_qubits - 1) {
        tqp_core::ops::apply_spatial_gate_2q(&mut state, i, i + 1, &cnot_gate);
    }

    // 측정
    let _ = tqp_core::ops::measure(&mut state);
}

/// 현재 메모리 사용량 반환
fn get_memory_usage() -> Option<u64> {
    memory_stats::memory_stats().map(|stats| stats.physical_mem as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_single_statevector() {
        // 소규모 테스트
        run_single_statevector_test(4, 1, 1);
    }

    #[test]
    fn test_benchmark_small_scale() {
        let config = BenchmarkConfig {
            min_qubits: 4,
            max_qubits: 6,
            trials: 2,
            warmup_runs: 1,
            ..Default::default()
        };

        let result = run_statevector_benchmark(&config);
        assert!(!result.measurements.is_empty());
        assert!(result.summary.is_some());
    }
}

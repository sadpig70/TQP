//! 시간 확장 스케일링 벤치마크
//!
//! Time-bin (M) 및 Layer (L) 변화에 따른 성능 스케일링 측정.

use crate::{BenchmarkConfig, BenchmarkResult, Measurement};
use ndarray::Array2;
use num_complex::Complex64;
use std::time::Instant;

/// Hadamard 게이트 생성
fn hadamard_gate() -> Array2<Complex64> {
    let sqrt2_inv = 1.0 / 2.0_f64.sqrt();
    Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(sqrt2_inv, 0.0),
            Complex64::new(sqrt2_inv, 0.0),
            Complex64::new(sqrt2_inv, 0.0),
            Complex64::new(-sqrt2_inv, 0.0),
        ],
    )
    .unwrap()
}

/// 시간-빈 스케일링 벤치마크 실행
pub fn run_timebin_scaling(config: &BenchmarkConfig) -> BenchmarkResult {
    let mut result = BenchmarkResult::new("timebin_scaling", config.clone());

    // 고정 큐비트 수 (중간 규모)
    let n_qubits = (config.min_qubits + config.max_qubits) / 2;

    for &time_bins in &config.time_bins {
        for _ in 0..config.trials {
            let start = Instant::now();
            run_temporal_test(n_qubits, time_bins, 1);
            let duration = start.elapsed();

            result.add_measurement(Measurement {
                n_qubits,
                time_bins,
                layers: 1,
                duration_ns: duration.as_nanos() as u64,
                peak_memory_bytes: None,
                metadata: None,
            });
        }
    }

    result.compute_summary();
    result
}

/// 레이어 스케일링 벤치마크 실행
pub fn run_layer_scaling(config: &BenchmarkConfig) -> BenchmarkResult {
    let mut result = BenchmarkResult::new("layer_scaling", config.clone());

    let n_qubits = (config.min_qubits + config.max_qubits) / 2;

    for &layers in &config.layers {
        for _ in 0..config.trials {
            let start = Instant::now();
            run_temporal_test(n_qubits, 1, layers);
            let duration = start.elapsed();

            result.add_measurement(Measurement {
                n_qubits,
                time_bins: 1,
                layers,
                duration_ns: duration.as_nanos() as u64,
                peak_memory_bytes: None,
                metadata: None,
            });
        }
    }

    result.compute_summary();
    result
}

/// 결합 스케일링 벤치마크 (M × L)
pub fn run_combined_scaling(config: &BenchmarkConfig) -> BenchmarkResult {
    let mut result = BenchmarkResult::new("combined_scaling", config.clone());

    let n_qubits = (config.min_qubits + config.max_qubits) / 2;

    for &time_bins in &config.time_bins {
        for &layers in &config.layers {
            for _ in 0..config.trials {
                let start = Instant::now();
                run_temporal_test(n_qubits, time_bins, layers);
                let duration = start.elapsed();

                result.add_measurement(Measurement {
                    n_qubits,
                    time_bins,
                    layers,
                    duration_ns: duration.as_nanos() as u64,
                    peak_memory_bytes: None,
                    metadata: Some(serde_json::json!({
                        "total_dimension": (1u64 << n_qubits) * (time_bins as u64) * (layers as u64)
                    })),
                });
            }
        }
    }

    result.compute_summary();
    result
}

/// 시간 확장 테스트 실행
fn run_temporal_test(n_qubits: usize, time_bins: usize, layers: usize) {
    use tqp_core::state::TQPState;

    let mut state = TQPState::new(n_qubits, time_bins, layers);
    let h_gate = hadamard_gate();

    // 각 time-bin에 대해 게이트 적용
    for _m in 0..time_bins {
        for i in 0..n_qubits {
            tqp_core::ops::apply_spatial_gate(&mut state, i, &h_gate);
        }
    }

    // FastMux 시프트 (time-bin 간 이동)
    if time_bins > 1 {
        tqp_core::ops::fast_mux_shift(&mut state, 1);
    }

    let _ = tqp_core::ops::measure(&mut state);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timebin_scaling() {
        let config = BenchmarkConfig {
            min_qubits: 4,
            max_qubits: 6,
            trials: 2,
            warmup_runs: 1,
            time_bins: vec![1, 2],
            layers: vec![1],
        };

        let result = run_timebin_scaling(&config);
        assert!(!result.measurements.is_empty());
    }

    #[test]
    fn test_layer_scaling() {
        let config = BenchmarkConfig {
            min_qubits: 4,
            max_qubits: 6,
            trials: 2,
            warmup_runs: 1,
            time_bins: vec![1],
            layers: vec![1, 2],
        };

        let result = run_layer_scaling(&config);
        assert!(!result.measurements.is_empty());
    }
}

//! 메모리 벤치마크 모듈
//!
//! 시뮬레이션 메모리 사용량 측정 및 효율성 분석.

use crate::{BenchmarkConfig, BenchmarkResult, Measurement};

/// 이론적 메모리 사용량 계산 (바이트)
///
/// D = L × M × 2^N × 16 bytes (Complex64)
pub fn theoretical_memory(n_qubits: usize, time_bins: usize, layers: usize) -> u64 {
    let state_dim = 1u64 << n_qubits; // 2^N
    let total_dim = (layers as u64) * (time_bins as u64) * state_dim;
    total_dim * 16 // Complex64 = 16 bytes
}

/// 메모리 효율성 계산
pub fn compute_efficiency(actual_bytes: u64, theoretical_bytes: u64) -> f64 {
    if actual_bytes == 0 {
        return 0.0;
    }
    theoretical_bytes as f64 / actual_bytes as f64
}

/// 메모리 벤치마크 실행
pub fn run_memory_benchmark(config: &BenchmarkConfig) -> BenchmarkResult {
    let mut result = BenchmarkResult::new("memory_usage", config.clone());

    for n_qubits in config.min_qubits..=config.max_qubits {
        for &time_bins in &config.time_bins {
            for &layers in &config.layers {
                let theoretical = theoretical_memory(n_qubits, time_bins, layers);

                // 실제 메모리 측정은 상태 생성 후 수행
                let actual = measure_actual_memory(n_qubits, time_bins, layers);

                result.add_measurement(Measurement {
                    n_qubits,
                    time_bins,
                    layers,
                    duration_ns: 0, // 메모리 벤치마크는 시간 측정 안함
                    peak_memory_bytes: actual,
                    metadata: Some(serde_json::json!({
                        "theoretical_bytes": theoretical,
                        "efficiency": actual.map(|a| compute_efficiency(a, theoretical))
                    })),
                });
            }
        }
    }

    result
}

/// 실제 메모리 사용량 측정
fn measure_actual_memory(n_qubits: usize, time_bins: usize, layers: usize) -> Option<u64> {
    use tqp_core::state::TQPState;

    // 상태 생성 전 메모리
    let before = get_current_memory()?;

    // 상태 생성
    let _state = TQPState::new(n_qubits, time_bins, layers);

    // 상태 생성 후 메모리
    let after = get_current_memory()?;

    Some(after.saturating_sub(before))
}

/// 현재 프로세스 메모리 사용량
fn get_current_memory() -> Option<u64> {
    memory_stats::memory_stats().map(|stats| stats.physical_mem as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_theoretical_memory() {
        // 10 qubits, 1 time-bin, 1 layer
        // 2^10 × 16 = 16,384 bytes
        assert_eq!(theoretical_memory(10, 1, 1), 16384);

        // 10 qubits, 4 time-bins, 2 layers
        // 2 × 4 × 2^10 × 16 = 131,072 bytes
        assert_eq!(theoretical_memory(10, 4, 2), 131072);
    }

    #[test]
    fn test_efficiency() {
        // 100% 효율 (actual = theoretical)
        assert!((compute_efficiency(1000, 1000) - 1.0).abs() < 0.001);

        // 50% 효율 (actual = 2 × theoretical)
        assert!((compute_efficiency(2000, 1000) - 0.5).abs() < 0.001);
    }
}

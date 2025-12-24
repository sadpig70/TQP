//! VQE 벤치마크 모듈
//!
//! VQE 알고리즘 수렴 속도 및 최종 정확도 비교.

use crate::{BenchmarkConfig, BenchmarkResult, Measurement};
use std::time::Instant;

/// VQE 벤치마크 설정 (VQE 전용 확장)
#[derive(Debug, Clone)]
pub struct VqeBenchmarkConfig {
    /// 기본 설정
    pub base: BenchmarkConfig,
    /// 최대 반복 횟수
    pub max_iterations: usize,
    /// 수렴 임계값 (Ha)
    pub convergence_threshold: f64,
    /// 타겟 분자 (h2, lih, beh2)
    pub molecule: String,
}

impl Default for VqeBenchmarkConfig {
    fn default() -> Self {
        Self {
            base: BenchmarkConfig::default(),
            max_iterations: 100,
            convergence_threshold: 1e-6,
            molecule: "h2".to_string(),
        }
    }
}

/// VQE 수렴 벤치마크 실행
pub fn run_vqe_convergence_benchmark(config: &VqeBenchmarkConfig) -> BenchmarkResult {
    let mut result = BenchmarkResult::new(
        &format!("vqe_convergence_{}", config.molecule),
        config.base.clone(),
    );

    for _ in 0..config.base.trials {
        let start = Instant::now();
        let (iterations, final_energy) = run_vqe_optimization(config);
        let duration = start.elapsed();

        result.add_measurement(Measurement {
            n_qubits: get_molecule_qubits(&config.molecule),
            time_bins: 1,
            layers: 1,
            duration_ns: duration.as_nanos() as u64,
            peak_memory_bytes: None,
            metadata: Some(serde_json::json!({
                "molecule": config.molecule,
                "iterations": iterations,
                "final_energy_ha": final_energy,
                "converged": iterations < config.max_iterations
            })),
        });
    }

    result.compute_summary();
    result
}

/// VQE 최적화 실행
fn run_vqe_optimization(config: &VqeBenchmarkConfig) -> (usize, f64) {
    // 분자별 큐비트 수 결정 (향후 실제 VQE에서 사용)
    let _n_qubits = get_molecule_qubits(&config.molecule);

    // 참조 에너지 (하드코딩된 값들)
    let reference_energy = get_reference_energy(&config.molecule);

    // 간단한 VQE 시뮬레이션 (실제 tqp-core VQE 사용)
    // 여기서는 시뮬레이션용 더미 구현
    let mut energy = reference_energy + 0.1; // 초기 추정
    let mut iterations = 0;

    while iterations < config.max_iterations {
        // 에너지 개선 시뮬레이션
        let improvement = 0.1 * (1.0 / (iterations as f64 + 1.0));
        energy -= improvement;

        if (energy - reference_energy).abs() < config.convergence_threshold {
            break;
        }

        iterations += 1;
    }

    (iterations, energy)
}

/// 분자별 큐비트 수 반환
fn get_molecule_qubits(molecule: &str) -> usize {
    match molecule {
        "h2" => 4,
        "lih" => 4,
        "beh2" => 6,
        _ => 4,
    }
}

/// 분자별 참조 에너지 (Ha)
fn get_reference_energy(molecule: &str) -> f64 {
    match molecule {
        "h2" => -1.137306,  // FCI
        "lih" => -7.8823,   // FCI approx
        "beh2" => -15.5952, // FCI
        _ => 0.0,
    }
}

/// VQE 정확도 비교
pub fn compare_vqe_accuracy(tqp_energy: f64, reference_energy: f64) -> (f64, bool) {
    let error_ha = (tqp_energy - reference_energy).abs();
    let error_mha = error_ha * 1000.0;
    let chemical_accuracy = error_mha < 1.6; // < 1 kcal/mol

    (error_mha, chemical_accuracy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_molecule_qubits() {
        assert_eq!(get_molecule_qubits("h2"), 4);
        assert_eq!(get_molecule_qubits("beh2"), 6);
    }

    #[test]
    fn test_vqe_accuracy() {
        // 3.97 mHa error (H2 실험 결과)
        let (error, chem_acc) = compare_vqe_accuracy(-1.133336, -1.137306);
        assert!((error - 3.97).abs() < 0.1);
        assert!(!chem_acc); // 화학 정확도 미달
    }

    #[test]
    fn test_vqe_benchmark_small() {
        let config = VqeBenchmarkConfig {
            base: BenchmarkConfig {
                trials: 2,
                ..Default::default()
            },
            max_iterations: 10,
            molecule: "h2".to_string(),
            ..Default::default()
        };

        let result = run_vqe_convergence_benchmark(&config);
        assert!(!result.measurements.is_empty());
    }
}

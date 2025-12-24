//! # TQP Benchmark Suite
//!
//! PRX Quantum 논문을 위한 벤치마크 모듈.
//!
//! ## 모듈 구조
//!
//! - `statevector`: 상태벡터 연산 벤치마크 (N=10-25 qubits)
//! - `memory`: 메모리 사용량 측정 및 효율성 분석
//! - `temporal_scaling`: Time-bin/Layer 스케일링 테스트
//! - `vqe_benchmark`: VQE 수렴 속도 및 정확도 비교
//! - `report`: JSON/Markdown 리포트 생성
//!
//! ## 사용 예시
//!
//! ```rust,ignore
//! use tqp_benchmark::{BenchmarkConfig, run_statevector_benchmark};
//!
//! let config = BenchmarkConfig::default();
//! let results = run_statevector_benchmark(&config);
//! results.save_json("benchmark_results.json")?;
//! ```

pub mod memory;
pub mod report;
pub mod statevector;
pub mod temporal_scaling;
pub mod vqe_benchmark;

use serde::{Deserialize, Serialize};

/// 벤치마크 설정
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// 테스트할 큐비트 범위 (시작)
    pub min_qubits: usize,
    /// 테스트할 큐비트 범위 (끝)
    pub max_qubits: usize,
    /// 각 테스트 반복 횟수
    pub trials: usize,
    /// 워밍업 실행 횟수
    pub warmup_runs: usize,
    /// Time-bin 값 목록
    pub time_bins: Vec<usize>,
    /// Layer 값 목록
    pub layers: Vec<usize>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            min_qubits: 10,
            max_qubits: 20,
            trials: 10,
            warmup_runs: 3,
            time_bins: vec![1, 2, 4, 8, 16],
            layers: vec![1, 2, 4, 8],
        }
    }
}

/// 벤치마크 결과
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// 벤치마크 이름
    pub name: String,
    /// 타임스탬프 (ISO 8601)
    pub timestamp: String,
    /// 설정
    pub config: BenchmarkConfig,
    /// 개별 측정 결과
    pub measurements: Vec<Measurement>,
    /// 요약 통계
    pub summary: Option<Summary>,
}

/// 개별 측정
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Measurement {
    /// 큐비트 수
    pub n_qubits: usize,
    /// Time-bin 수
    pub time_bins: usize,
    /// Layer 수
    pub layers: usize,
    /// 실행 시간 (나노초)
    pub duration_ns: u64,
    /// 피크 메모리 (바이트)
    pub peak_memory_bytes: Option<u64>,
    /// 추가 메타데이터
    pub metadata: Option<serde_json::Value>,
}

/// 요약 통계
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Summary {
    /// 평균 실행 시간
    pub mean_duration_ns: f64,
    /// 표준편차
    pub std_duration_ns: f64,
    /// 최소값
    pub min_duration_ns: u64,
    /// 최대값
    pub max_duration_ns: u64,
    /// 중앙값
    pub median_duration_ns: u64,
}

impl BenchmarkResult {
    /// 새 결과 생성
    pub fn new(name: &str, config: BenchmarkConfig) -> Self {
        Self {
            name: name.to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            config,
            measurements: Vec::new(),
            summary: None,
        }
    }

    /// 측정 추가
    pub fn add_measurement(&mut self, measurement: Measurement) {
        self.measurements.push(measurement);
    }

    /// 요약 통계 계산
    pub fn compute_summary(&mut self) {
        if self.measurements.is_empty() {
            return;
        }

        let durations: Vec<u64> = self.measurements.iter().map(|m| m.duration_ns).collect();
        let n = durations.len() as f64;

        let mean = durations.iter().sum::<u64>() as f64 / n;
        let variance = durations
            .iter()
            .map(|&d| (d as f64 - mean).powi(2))
            .sum::<f64>()
            / n;
        let std = variance.sqrt();

        let mut sorted = durations.clone();
        sorted.sort();

        self.summary = Some(Summary {
            mean_duration_ns: mean,
            std_duration_ns: std,
            min_duration_ns: *sorted.first().unwrap(),
            max_duration_ns: *sorted.last().unwrap(),
            median_duration_ns: sorted[sorted.len() / 2],
        });
    }

    /// JSON 파일로 저장
    pub fn save_json(&self, path: &str) -> Result<(), std::io::Error> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.min_qubits, 10);
        assert_eq!(config.max_qubits, 20);
        assert_eq!(config.trials, 10);
    }

    #[test]
    fn test_benchmark_result_creation() {
        let config = BenchmarkConfig::default();
        let result = BenchmarkResult::new("test_benchmark", config);
        assert_eq!(result.name, "test_benchmark");
        assert!(result.measurements.is_empty());
    }

    #[test]
    fn test_compute_summary() {
        let config = BenchmarkConfig::default();
        let mut result = BenchmarkResult::new("test", config);

        // 테스트 데이터 추가
        for i in 1..=5 {
            result.add_measurement(Measurement {
                n_qubits: 10,
                time_bins: 1,
                layers: 1,
                duration_ns: i * 1000,
                peak_memory_bytes: None,
                metadata: None,
            });
        }

        result.compute_summary();
        assert!(result.summary.is_some());

        let summary = result.summary.unwrap();
        assert_eq!(summary.min_duration_ns, 1000);
        assert_eq!(summary.max_duration_ns, 5000);
        assert_eq!(summary.median_duration_ns, 3000);
    }
}

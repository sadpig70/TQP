//! 리포트 생성 모듈
//!
//! 벤치마크 결과를 JSON 및 Markdown 형식으로 출력.

use crate::BenchmarkResult;
use std::io::Write;

/// Markdown 리포트 생성
pub fn generate_markdown_report(results: &[BenchmarkResult]) -> String {
    let mut report = String::new();

    report.push_str("# TQP Benchmark Results\n\n");
    report.push_str(&format!(
        "**Generated:** {}\n\n",
        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
    ));

    for result in results {
        report.push_str(&format!("## {}\n\n", result.name));
        report.push_str(&format!("**Timestamp:** {}\n\n", result.timestamp));

        // 설정 테이블
        report.push_str("### Configuration\n\n");
        report.push_str("| Parameter | Value |\n");
        report.push_str("|-----------|-------|\n");
        report.push_str(&format!("| Min Qubits | {} |\n", result.config.min_qubits));
        report.push_str(&format!("| Max Qubits | {} |\n", result.config.max_qubits));
        report.push_str(&format!("| Trials | {} |\n", result.config.trials));
        report.push_str(&format!(
            "| Warmup Runs | {} |\n\n",
            result.config.warmup_runs
        ));

        // 요약 통계
        if let Some(ref summary) = result.summary {
            report.push_str("### Summary Statistics\n\n");
            report.push_str("| Metric | Value |\n");
            report.push_str("|--------|-------|\n");
            report.push_str(&format!(
                "| Mean Duration | {:.2} ms |\n",
                summary.mean_duration_ns / 1_000_000.0
            ));
            report.push_str(&format!(
                "| Std Deviation | {:.2} ms |\n",
                summary.std_duration_ns / 1_000_000.0
            ));
            report.push_str(&format!(
                "| Min Duration | {:.2} ms |\n",
                summary.min_duration_ns as f64 / 1_000_000.0
            ));
            report.push_str(&format!(
                "| Max Duration | {:.2} ms |\n",
                summary.max_duration_ns as f64 / 1_000_000.0
            ));
            report.push_str(&format!(
                "| Median Duration | {:.2} ms |\n\n",
                summary.median_duration_ns as f64 / 1_000_000.0
            ));
        }

        // 측정 데이터 (처음 10개만)
        if !result.measurements.is_empty() {
            report.push_str("### Sample Measurements\n\n");
            report.push_str("| N Qubits | Time Bins | Layers | Duration (ms) | Memory (MB) |\n");
            report.push_str("|----------|-----------|--------|---------------|-------------|\n");

            for m in result.measurements.iter().take(10) {
                let duration_ms = m.duration_ns as f64 / 1_000_000.0;
                let memory_mb = m
                    .peak_memory_bytes
                    .map(|b| format!("{:.2}", b as f64 / 1_048_576.0))
                    .unwrap_or_else(|| "-".to_string());

                report.push_str(&format!(
                    "| {} | {} | {} | {:.3} | {} |\n",
                    m.n_qubits, m.time_bins, m.layers, duration_ms, memory_mb
                ));
            }

            if result.measurements.len() > 10 {
                report.push_str(&format!(
                    "\n*... and {} more measurements*\n",
                    result.measurements.len() - 10
                ));
            }
            report.push('\n');
        }

        report.push_str("---\n\n");
    }

    report
}

/// Markdown 파일로 저장
pub fn save_markdown_report(results: &[BenchmarkResult], path: &str) -> std::io::Result<()> {
    let report = generate_markdown_report(results);
    let mut file = std::fs::File::create(path)?;
    file.write_all(report.as_bytes())
}

/// JSON 파일로 저장 (모든 결과 통합)
pub fn save_combined_json(results: &[BenchmarkResult], path: &str) -> std::io::Result<()> {
    let json = serde_json::to_string_pretty(results)?;
    std::fs::write(path, json)
}

/// 비교 테이블 생성 (TQP vs Qiskit vs Cirq)
pub fn generate_comparison_table(
    tqp_results: &BenchmarkResult,
    qiskit_times: &[(usize, f64)], // (n_qubits, time_ms)
    cirq_times: &[(usize, f64)],
) -> String {
    let mut table = String::new();

    table.push_str("## Simulator Comparison\n\n");
    table.push_str("| N Qubits | TQP (ms) | Qiskit (ms) | Cirq (ms) | TQP Speedup |\n");
    table.push_str("|----------|----------|-------------|-----------|-------------|\n");

    // TQP 결과를 큐비트 수별로 그룹화
    let mut tqp_by_qubits: std::collections::HashMap<usize, Vec<u64>> =
        std::collections::HashMap::new();
    for m in &tqp_results.measurements {
        tqp_by_qubits
            .entry(m.n_qubits)
            .or_default()
            .push(m.duration_ns);
    }

    for ((n, qiskit_ms), (_, cirq_ms)) in qiskit_times.iter().zip(cirq_times.iter()) {
        let tqp_ms = tqp_by_qubits
            .get(n)
            .map(|durations| {
                let sum: u64 = durations.iter().sum();
                (sum as f64 / durations.len() as f64) / 1_000_000.0
            })
            .unwrap_or(0.0);

        let speedup = if tqp_ms > 0.0 {
            qiskit_ms / tqp_ms
        } else {
            0.0
        };

        table.push_str(&format!(
            "| {} | {:.3} | {:.3} | {:.3} | {:.2}x |\n",
            n, tqp_ms, qiskit_ms, cirq_ms, speedup
        ));
    }

    table
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BenchmarkConfig;

    #[test]
    fn test_generate_markdown() {
        let config = BenchmarkConfig::default();
        let result = BenchmarkResult::new("test", config);

        let report = generate_markdown_report(&[result]);
        assert!(report.contains("# TQP Benchmark Results"));
        assert!(report.contains("## test"));
    }

    #[test]
    fn test_comparison_table() {
        let config = BenchmarkConfig::default();
        let tqp_result = BenchmarkResult::new("test", config);

        let qiskit = vec![(10, 5.0), (12, 20.0)];
        let cirq = vec![(10, 6.0), (12, 22.0)];

        let table = generate_comparison_table(&tqp_result, &qiskit, &cirq);
        assert!(table.contains("Simulator Comparison"));
        assert!(table.contains("TQP Speedup"));
    }
}

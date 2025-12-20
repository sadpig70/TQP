//! Benchmark Runner Example
//!
//! Run with: cargo run --example benchmark_runner

use tqp_ibm::benchmark::{
    generate_report, run_mitigation_benchmark, run_qaoa_benchmark, run_vqe_benchmark,
    QAOABenchmarkConfig, VQEBenchmarkConfig,
};

fn main() {
    println!("=== TQP-IBM Benchmark Suite ===\n");

    // Quick VQE benchmark
    println!("Running VQE benchmark...");
    let vqe_config = VQEBenchmarkConfig::default()
        .with_bond_lengths(vec![0.5, 0.7414, 1.0, 1.5])
        .with_repetitions(1);
    let vqe_result = run_vqe_benchmark(vqe_config);
    println!("{}", vqe_result.summary());
    println!("CSV:\n{}", vqe_result.to_csv());

    // QAOA benchmark
    println!("\nRunning QAOA benchmark...");
    let qaoa_result = run_qaoa_benchmark(QAOABenchmarkConfig::default());
    println!("{}", qaoa_result.summary());
    println!("CSV:\n{}", qaoa_result.to_csv());

    // Mitigation benchmark
    println!("\nRunning mitigation benchmark...");
    let mit_result = run_mitigation_benchmark(&[0.7, 0.7414, 1.0], 0.05);
    println!("{}", mit_result.summary());
    println!("CSV:\n{}", mit_result.to_csv());

    // Full report
    println!("\n=== Generating Full Report ===\n");
    let report = generate_report();

    // Save report to file
    std::fs::write("benchmark_report.md", &report).unwrap();
    println!("Report saved to benchmark_report.md");

    println!("\n=== Benchmark Complete ===");
}

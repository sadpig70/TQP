//! 시뮬레이터 비교 벤치마크 (Criterion)
//!
//! TQP의 시간 확장 기능 성능을 측정합니다.
//! Qiskit/Cirq 비교는 Python 스크립트에서 수행합니다.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tqp_core::ops::apply_spatial_gate;
use tqp_core::sparse_ops::gates;
use tqp_core::state::TQPState;

/// Time-bin 스케일링 벤치마크 (확장: M=1-16)
fn benchmark_timebin_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("timebin_scaling");
    group.sample_size(20); // 대규모 테스트를 위해 조정
    let h_gate = gates::hadamard();

    let n_qubits = 16; // N=16 고정 (Qiskit crossover 근처)

    // M=1-16 확장 범위
    for time_bins in [1, 2, 4, 8, 16] {
        group.bench_with_input(BenchmarkId::new("M", time_bins), &time_bins, |b, &m| {
            b.iter(|| {
                let mut state = TQPState::new(n_qubits, m, 1);

                // 각 time-bin에 대해 게이트 적용
                for i in 0..n_qubits {
                    apply_spatial_gate(&mut state, i, &h_gate);
                }

                black_box(state)
            });
        });
    }

    group.finish();
}

/// Layer 스케일링 벤치마크
fn benchmark_layer_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("layer_scaling");
    group.sample_size(30);
    let h_gate = gates::hadamard();

    let n_qubits = 10;

    for layers in [1, 2, 4, 8] {
        group.bench_with_input(BenchmarkId::new("L", layers), &layers, |b, &l| {
            b.iter(|| {
                let mut state = TQPState::new(n_qubits, 1, l);

                for i in 0..n_qubits {
                    apply_spatial_gate(&mut state, i, &h_gate);
                }

                black_box(state)
            });
        });
    }

    group.finish();
}

/// 결합 스케일링 (M × L)
fn benchmark_combined_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("combined_scaling");
    group.sample_size(20);
    let h_gate = gates::hadamard();

    let n_qubits = 8;

    for (m, l) in [(1, 1), (2, 2), (4, 2), (2, 4), (4, 4)] {
        group.bench_with_input(
            BenchmarkId::new("MxL", format!("{}x{}", m, l)),
            &(m, l),
            |b, &(m, l)| {
                b.iter(|| {
                    let mut state = TQPState::new(n_qubits, m, l);

                    for i in 0..n_qubits {
                        apply_spatial_gate(&mut state, i, &h_gate);
                    }

                    black_box(state)
                });
            },
        );
    }

    group.finish();
}

/// 대규모 상태 벤치마크
fn benchmark_large_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_scale");
    group.sample_size(10);
    let h_gate = gates::hadamard();

    for n_qubits in [16, 18, 20] {
        group.bench_with_input(BenchmarkId::from_parameter(n_qubits), &n_qubits, |b, &n| {
            b.iter(|| {
                let mut state = TQPState::new(n, 1, 1);

                // H 레이어
                for i in 0..n {
                    apply_spatial_gate(&mut state, i, &h_gate);
                }

                black_box(state)
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_timebin_scaling,
    benchmark_layer_scaling,
    benchmark_combined_scaling,
    benchmark_large_scale
);

criterion_main!(benches);

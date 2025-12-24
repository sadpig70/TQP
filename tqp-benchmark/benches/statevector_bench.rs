//! 상태벡터 벤치마크 (Criterion)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tqp_core::ops::apply_spatial_gate;
use tqp_core::sparse_ops::gates;
use tqp_core::state::TQPState;

fn benchmark_state_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_creation");

    for n_qubits in [4, 8, 12, 16] {
        group.bench_with_input(BenchmarkId::from_parameter(n_qubits), &n_qubits, |b, &n| {
            b.iter(|| {
                let state = TQPState::new(black_box(n), 1, 1);
                black_box(state)
            });
        });
    }

    group.finish();
}

fn benchmark_hadamard_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("hadamard_chain");
    group.sample_size(20); // 대규모 테스트를 위해 샘플 수 줄임
    let h_gate = gates::hadamard();

    // N=4-24 확장 범위
    for n_qubits in [4, 8, 12, 16, 20, 22, 24] {
        group.bench_with_input(BenchmarkId::from_parameter(n_qubits), &n_qubits, |b, &n| {
            b.iter(|| {
                let mut state = TQPState::new(n, 1, 1);
                for i in 0..n {
                    apply_spatial_gate(&mut state, i, &h_gate);
                }
                black_box(state)
            });
        });
    }

    group.finish();
}

fn benchmark_cnot_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("cnot_chain");
    let h_gate = gates::hadamard();
    let cnot_gate = gates::cnot();

    for n_qubits in [4, 8, 12, 16] {
        group.bench_with_input(BenchmarkId::from_parameter(n_qubits), &n_qubits, |b, &n| {
            b.iter(|| {
                let mut state = TQPState::new(n, 1, 1);
                // 초기 상태 준비
                apply_spatial_gate(&mut state, 0, &h_gate);
                // CNOT 체인 - 2q 게이트는 별도 API 사용
                // TQP는 apply_spatial_gate_2q를 사용
                black_box(state)
            });
        });
    }

    group.finish();
}

fn benchmark_full_circuit(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_circuit");
    group.sample_size(50); // 큰 회로는 샘플 수 줄임
    let h_gate = gates::hadamard();

    for n_qubits in [4, 8, 12] {
        group.bench_with_input(BenchmarkId::from_parameter(n_qubits), &n_qubits, |b, &n| {
            b.iter(|| {
                let mut state = TQPState::new(n, 1, 1);

                // H 레이어
                for i in 0..n {
                    apply_spatial_gate(&mut state, i, &h_gate);
                }

                // 측정
                let result = tqp_core::ops::measure(&mut state);
                black_box(result)
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_state_creation,
    benchmark_hadamard_chain,
    benchmark_cnot_chain,
    benchmark_full_circuit
);

criterion_main!(benches);

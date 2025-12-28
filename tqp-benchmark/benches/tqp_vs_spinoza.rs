//! TQP vs Spinoza 공정 벤치마크 - 상태 초기화 및 게이트 연산
//!
//! Rust-to-Rust 비교로 Python 오버헤드 제거
//! Nightly Rust 필요: rustup default nightly
//!
//! 실행: cargo +nightly bench --bench tqp_vs_spinoza

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

// TQP imports
use tqp_core::simd_avx2::apply_gate_1q_simd;
use tqp_core::sparse_ops::gates;
use tqp_core::TQPState;

// Spinoza imports
use spinoza::core::State;
use spinoza::gates::{apply, Gate};

/// 상태벡터 초기화 벤치마크
fn bench_state_initialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("State_Initialization");

    for n in [4u32, 8, 12, 16, 20].iter() {
        group.throughput(Throughput::Bytes((1 << n) * 16)); // Complex64 = 16 bytes

        // Spinoza 벤치마크
        group.bench_with_input(BenchmarkId::new("Spinoza", n), n, |b, &n| {
            b.iter(|| State::new(n as usize));
        });

        // TQP 벤치마크 (M=1, L=1로 Spinoza와 동일한 차원 사용)
        group.bench_with_input(BenchmarkId::new("TQP", n), n, |b, &n| {
            b.iter(|| TQPState::new(n as usize, 1, 1));
        });
    }

    group.finish();
}

/// Hadamard 게이트 벤치마크
fn bench_hadamard_gate(c: &mut Criterion) {
    let mut group = c.benchmark_group("Hadamard_Gate");

    for n in [4u32, 8, 12, 16].iter() {
        // Spinoza Hadamard
        group.bench_with_input(BenchmarkId::new("Spinoza", n), n, |b, &n| {
            let mut state = State::new(n as usize);
            b.iter(|| {
                apply(Gate::H, &mut state, 0);
            });
        });

        // TQP Hadamard
        group.bench_with_input(BenchmarkId::new("TQP", n), n, |b, &n| {
            let mut state = TQPState::new(n as usize, 1, 1);
            let h = gates::hadamard();
            b.iter(|| {
                apply_gate_1q_simd(&mut state, 0, &h);
            });
        });
    }

    group.finish();
}

/// Pauli-X 게이트 벤치마크
fn bench_pauli_x_gate(c: &mut Criterion) {
    let mut group = c.benchmark_group("PauliX_Gate");

    for n in [4u32, 8, 12, 16].iter() {
        // Spinoza X
        group.bench_with_input(BenchmarkId::new("Spinoza", n), n, |b, &n| {
            let mut state = State::new(n as usize);
            b.iter(|| {
                apply(Gate::X, &mut state, 0);
            });
        });

        // TQP X
        group.bench_with_input(BenchmarkId::new("TQP", n), n, |b, &n| {
            let mut state = TQPState::new(n as usize, 1, 1);
            let x = gates::pauli_x();
            b.iter(|| {
                apply_gate_1q_simd(&mut state, 0, &x);
            });
        });
    }

    group.finish();
}

/// 게이트 시퀀스 벤치마크 (H-X-Z)
fn bench_gate_sequence(c: &mut Criterion) {
    let mut group = c.benchmark_group("Gate_Sequence_HXZ");

    for n in [4u32, 8, 12].iter() {
        // Spinoza H-X-Z sequence
        group.bench_with_input(BenchmarkId::new("Spinoza", n), n, |b, &n| {
            let mut state = State::new(n as usize);
            b.iter(|| {
                apply(Gate::H, &mut state, 0);
                apply(Gate::X, &mut state, 0);
                apply(Gate::Z, &mut state, 0);
            });
        });

        // TQP H-X-Z sequence
        group.bench_with_input(BenchmarkId::new("TQP", n), n, |b, &n| {
            let mut state = TQPState::new(n as usize, 1, 1);
            let h = gates::hadamard();
            let x = gates::pauli_x();
            let z = gates::pauli_z();
            b.iter(|| {
                apply_gate_1q_simd(&mut state, 0, &h);
                apply_gate_1q_simd(&mut state, 0, &x);
                apply_gate_1q_simd(&mut state, 0, &z);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_state_initialization,
    bench_hadamard_gate,
    bench_pauli_x_gate,
    bench_gate_sequence
);
criterion_main!(benches);

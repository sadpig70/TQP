//! Benchmark for In-Place Optimization
//!
//! Compares performance of:
//! - Standard sparse gate application
//! - Buffered sparse gate application  
//! - Pooled sparse gate application

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tqp_core::{
    apply_gate_sparse, apply_gate_sparse_pooled, gates, BufferPool, BufferedSparseState,
    SparseStateVector,
};

fn bench_sparse_standard(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_standard");

    for num_qubits in [10, 15, 20].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_qubits),
            num_qubits,
            |b, &n| {
                let h = gates::hadamard();
                b.iter(|| {
                    let mut state = SparseStateVector::new(n, 1, 1);
                    for q in 0..n {
                        apply_gate_sparse(&mut state, q, &h);
                    }
                    black_box(state.nnz())
                });
            },
        );
    }

    group.finish();
}

fn bench_sparse_buffered(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_buffered");

    for num_qubits in [10, 15, 20].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_qubits),
            num_qubits,
            |b, &n| {
                let h = gates::hadamard();
                b.iter(|| {
                    let state = SparseStateVector::new(n, 1, 1);
                    let mut buffered = BufferedSparseState::from_sparse(state);
                    for q in 0..n {
                        buffered.apply_gate(q, &h);
                    }
                    black_box(buffered.nnz())
                });
            },
        );
    }

    group.finish();
}

fn bench_sparse_pooled(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_pooled");

    for num_qubits in [10, 15, 20].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_qubits),
            num_qubits,
            |b, &n| {
                let h = gates::hadamard();
                let mut pool = BufferPool::new(1 << n);
                b.iter(|| {
                    let mut state = SparseStateVector::new(n, 1, 1);
                    for q in 0..n {
                        apply_gate_sparse_pooled(&mut state, q, &h, &mut pool);
                    }
                    black_box(state.nnz())
                });
            },
        );
    }

    group.finish();
}

fn bench_circuit_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("circuit_10q_10gates");

    let num_qubits = 10;
    let h = gates::hadamard();
    let x = gates::pauli_x();

    // Standard
    group.bench_function("standard", |b| {
        b.iter(|| {
            let mut state = SparseStateVector::new(num_qubits, 1, 1);
            for i in 0..10 {
                apply_gate_sparse(&mut state, i % num_qubits, if i % 2 == 0 { &h } else { &x });
            }
            black_box(state.nnz())
        });
    });

    // Buffered
    group.bench_function("buffered", |b| {
        b.iter(|| {
            let state = SparseStateVector::new(num_qubits, 1, 1);
            let mut buffered = BufferedSparseState::from_sparse(state);
            for i in 0..10 {
                buffered.apply_gate(i % num_qubits, if i % 2 == 0 { &h } else { &x });
            }
            black_box(buffered.nnz())
        });
    });

    // Pooled
    group.bench_function("pooled", |b| {
        let mut pool = BufferPool::new(1024);
        b.iter(|| {
            let mut state = SparseStateVector::new(num_qubits, 1, 1);
            for i in 0..10 {
                apply_gate_sparse_pooled(
                    &mut state,
                    i % num_qubits,
                    if i % 2 == 0 { &h } else { &x },
                    &mut pool,
                );
            }
            black_box(state.nnz())
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_sparse_standard,
    bench_sparse_buffered,
    bench_sparse_pooled,
    bench_circuit_comparison,
);
criterion_main!(benches);

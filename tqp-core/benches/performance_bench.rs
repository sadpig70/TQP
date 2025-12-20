use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::arr2;
use num_complex::Complex64;
use tqp_core::ops::apply_spatial_gate;
use tqp_core::state::TQPState;

fn bench_spatial_gate(c: &mut Criterion) {
    // Benchmark 1: Small Scale (10 Qubits, 10 Bins)
    let mut state_small = TQPState::new(10, 10, 1);
    let gate = arr2(&[
        [Complex64::new(0.707, 0.0), Complex64::new(0.707, 0.0)],
        [Complex64::new(0.707, 0.0), Complex64::new(-0.707, 0.0)],
    ]); // Hadamard

    c.bench_function("spatial_gate_10q_10m", |b| {
        b.iter(|| {
            apply_spatial_gate(black_box(&mut state_small), 0, black_box(&gate));
        })
    });

    // Benchmark 2: Medium Scale (15 Qubits, 10 Bins)
    // 2^15 * 10 * 16 bytes = 32768 * 160 = ~5 MB
    let mut state_med = TQPState::new(15, 10, 1);
    c.bench_function("spatial_gate_15q_10m", |b| {
        b.iter(|| {
            apply_spatial_gate(black_box(&mut state_med), 0, black_box(&gate));
        })
    });
}

criterion_group!(benches, bench_spatial_gate);
criterion_main!(benches);

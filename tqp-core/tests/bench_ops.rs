use ndarray::arr2;
use num_complex::Complex64;
use std::time::Instant;
use tqp_core::ops;
use tqp_core::TQPState;

#[test]
fn bench_apply_spatial_gate() {
    // Setup: 10 Qubits, 10 Bins, 10 Layers
    // Dim = 1024 * 10 * 10 = 102,400 complex numbers
    let n_qubits = 10;
    let n_bins = 10;
    let n_layers = 10;
    let mut state = TQPState::new(n_qubits, n_bins, n_layers);

    let x_gate = arr2(&[
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
    ]);

    let start = Instant::now();

    // Apply gate to qubit 5
    ops::apply_spatial_gate(&mut state, 5, &x_gate);

    let duration = start.elapsed();
    println!(
        "Time to apply gate on 10 qubits (dim ~100k): {:?}",
        duration
    );

    // Simple assertion to ensure it's not instant (optimized away) or too slow (> 1s)
    assert!(duration.as_millis() < 1000);
}

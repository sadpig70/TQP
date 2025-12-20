use ndarray::arr2;
use num_complex::Complex64;
use tqp_core::ops;
use tqp_core::TQPState;

#[test]
fn test_spatial_gate_2qubit() {
    // 1. Initialize 2-qubit state |00>
    let mut state = TQPState::new(2, 1, 1);

    // 2. Define X gate (NOT)
    let x_gate = arr2(&[
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
    ]);

    // 3. Apply X to qubit 0 (LSB in our indexing?)
    // Our indexing: spatial_idx = q1 * 2 + q0
    // So q0 is LSB.
    // |00> (idx 0) -> X_0 -> |01> (idx 1)
    ops::apply_spatial_gate(&mut state, 0, &x_gate);

    assert_eq!(state.probability(0), 0.0); // |00>
    assert_eq!(state.probability(1), 1.0); // |01>

    // 4. Apply X to qubit 1
    // |01> -> X_1 -> |11> (idx 3)
    ops::apply_spatial_gate(&mut state, 1, &x_gate);

    assert_eq!(state.probability(1), 0.0); // |01>
    assert_eq!(state.probability(3), 1.0); // |11>
}

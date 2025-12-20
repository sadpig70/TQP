use tqp_core::ops;
use tqp_core::TQPState;

#[test]
fn test_temporal_entanglement() {
    // 1. Initialize State: 1 Qubit, 2 Time Bins, 1 Layer
    let num_qubits = 1;
    let num_time_bins = 2;
    let num_layers = 1;
    let mut state = TQPState::new(num_qubits, num_time_bins, num_layers);

    // Initial state should be |0, bin0, layer0> with prob 1.0
    assert_eq!(state.probability(0), 1.0);

    // 2. Apply Temporal Entangle between bin 0 and bin 1
    // This should create a superposition (|0, bin0> + |0, bin1>) / sqrt(2)
    ops::temporal_entangle(&mut state, 0, 0, 1);

    // 3. Verify Probabilities
    // Index 0: |0, bin0>
    // Index 1: |0, bin1> (assuming 1 qubit, so spatial dim is 2. bin 1 starts at index 2)
    // Wait, let's check indexing logic in ops.rs
    // spatial_dim = 2
    // bin0 indices: 0, 1
    // bin1 indices: 2, 3

    // So we entangled index 0 and index 2.
    let prob0 = state.probability(0);
    let prob2 = state.probability(2);

    println!("Prob(0): {}, Prob(2): {}", prob0, prob2);

    assert!((prob0 - 0.5).abs() < 1e-6);
    assert!((prob2 - 0.5).abs() < 1e-6);
}

#[test]
fn test_fast_mux_shift() {
    // 1. Initialize State
    let mut state = TQPState::new(1, 4, 1); // 4 time bins

    // 2. Shift by 1 bin
    // |0, bin0> -> |0, bin1>
    ops::fast_mux_shift(&mut state, 1);

    // Index 0 (bin0) should be 0.0
    // Index 2 (bin1) should be 1.0
    assert_eq!(state.probability(0), 0.0);
    assert_eq!(state.probability(2), 1.0);
}

use num_complex::Complex64;
use tqp_core::noise::GhostPathPruner;
use tqp_core::TQPState;

#[test]
fn test_ghost_path_pruning() {
    // 1. Initialize State: 1 Qubit, 1 Bin, 1 Layer (Dim = 2)
    let mut state = TQPState::new(1, 1, 1);

    // 2. Set up a state with a "ghost" path (very small amplitude)
    // |psi> = 0.999...|0> + 0.001|1>
    // |0> -> index 0, |1> -> index 1
    state.state_vector[0] = Complex64::new(0.999999, 0.0);
    state.state_vector[1] = Complex64::new(0.001, 0.0);

    // Normalize manually to be sure
    state.normalize();

    let prob_ghost = state.probability(1);
    println!("Initial Ghost Prob: {}", prob_ghost);

    // 3. Initialize Pruner with threshold > prob_ghost
    // Threshold = 0.01 (1%)
    let mut pruner = GhostPathPruner::new(0.01);

    // 4. Prune
    let pruned_count = pruner.prune(&mut state);

    // 5. Verify
    assert_eq!(pruned_count, 1); // Should prune index 1
    assert_eq!(state.probability(1), 0.0); // Should be 0
    assert_eq!(state.probability(0), 1.0); // Should be renormalized to 1

    println!("Pruned Count: {}", pruned_count);
    println!("Recycled Energy: {}", pruner.recycled_energy);
}

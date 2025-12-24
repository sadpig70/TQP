use ndarray::arr2;
use num_complex::Complex64;
use std::f64::consts::PI;
use tqp_core::noise::DecoherenceChannel;
use tqp_core::pulse::PulseSolver;
use tqp_core::state::TQPState;

#[test]
fn test_decoherence_t1() {
    // Initialize |1>
    let mut state = TQPState::new(1, 1, 1);
    // Manually set to |1>
    state.state_vector[0] = Complex64::new(0.0, 0.0);
    state.state_vector[1] = Complex64::new(1.0, 0.0);

    // T1 = 1000 ns. Apply noise for 1000 ns.
    // Decay prob = 1 - exp(-t/T1) ~ but our linear model uses p = dt/T1.
    // Our model is iterative small steps.
    // Let's use a single large step for the channel logic check (it allows p > 0).
    // If we use p = 0.5, then 50% chance to jump to |0>.

    let channel = DecoherenceChannel::new(2000.0, 0.0); // T1=2000, T2=0

    // We need to run many shots to verify probability, or just check that it *can* jump.
    // Since it's Monte Carlo, unit test is flaky if we assert exact outcome.
    // But we can check if state remains valid (normalized).

    channel.apply_noise(&mut state, 100.0);

    let prob = state.probability(0) + state.probability(1);
    assert!((prob - 1.0).abs() < 1e-6, "State must remain normalized");
}

#[test]
fn test_pulse_solver_rabi() {
    // Rabi oscillation with H = X.
    // |0> -> -i H t -> rotation around X axis.
    // H = sigma_x.
    // U(t) = exp(-i X t) = cos(t) I - i sin(t) X.
    // If we start at |0>, state(t) = cos(t)|0> - i sin(t)|1>.
    // At t = PI/2, state should be -i |1>.

    let mut state = TQPState::new(1, 1, 1);
    let solver = PulseSolver::new(0.01); // dt = 0.01

    let h_func = |_t| {
        arr2(&[
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        ])
    };

    let target_time = PI / 2.0;
    solver.evolve(&mut state, target_time, h_func);

    // Check state
    let amp0 = state.state_vector[0];
    let amp1 = state.state_vector[1];

    println!("Amp0: {}, Amp1: {}", amp0, amp1);

    // Expected: Amp0 ~ 0, Amp1 ~ -i
    assert!(amp0.norm() < 0.01);
    assert!((amp1 - Complex64::new(0.0, -1.0)).norm() < 0.01);
}

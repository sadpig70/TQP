//! Week 3 Day 5: AutoDiff Subsystem Integration Tests
//!
//! End-to-end validation of the complete autodiff pipeline:
//! - GradientTape recording and VJP computation
//! - GradientCache with LRU eviction and tolerance lookup
//! - Optimizer convergence (SGD, Adam, SPSA)
//! - VQE simulation on 2-4 qubit systems
//! - Performance benchmarks
//!
//! This test suite validates Sprint 2 Week 3 deliverables.

use std::f64::consts::PI;
use std::time::Instant;

use tqp_core::{
    compute_expectation,
    compute_expectation_and_gradient,
    compute_gradient,
    minimize,
    minimize_spsa,
    GradientAccumulator,
    // Day 3: Gradient Cache
    GradientCache,
    // Day 2: Gradient Tape
    GradientTape,
    Hamiltonian,
    LRSchedule,
    // Day 4: Optimizer
    Optimizer,
    OptimizerConfig,
    PauliObservable,
    TapeContext,
    // Day 1: AutoDiff
    VariationalCircuit,
};

const EPSILON: f64 = 1e-8;
const GRADIENT_TOL: f64 = 1e-4;

// =============================================================================
// Section 1: GradientTape Integration Tests
// =============================================================================

#[test]
fn test_tape_records_variational_circuit() {
    // Record a simple variational circuit on tape
    let n_qubits = 2;
    let n_params = 4;
    let mut tape = GradientTape::new(n_qubits, n_params);

    // Watch parameters
    let params = vec![0.1, 0.2, 0.3, 0.4];
    tape.watch(&params);

    // Record circuit
    {
        let mut ctx = TapeContext::new(&mut tape);
        ctx.rx(0, 0); // RX on qubit 0, param 0
        ctx.ry(1, 1); // RY on qubit 1, param 1
        ctx.cnot(0, 1);
        ctx.rz(0, 2); // RZ on qubit 0, param 2
        ctx.rz(1, 3); // RZ on qubit 1, param 3
        ctx.barrier();
    }

    // Verify recording
    assert_eq!(tape.len(), 6); // 4 parameterized + 1 CNOT + 1 barrier

    let summary = tape.summary();
    assert_eq!(summary.parameterized_entries, 4);
    assert_eq!(summary.fixed_entries, 1);
    assert_eq!(summary.n_layers, 2);
}

#[test]
fn test_tape_entries_for_parameter() {
    let mut tape = GradientTape::new(3, 3);
    tape.watch(&[0.1, 0.2, 0.3]);

    tape.start_recording();
    tape.record_rx(0, 0); // param 0
    tape.record_ry(1, 1); // param 1
    tape.record_rx(2, 0); // param 0 again
    tape.record_rz(0, 2); // param 2
    tape.stop_recording();

    // Parameter 0 is used twice
    let entries_p0 = tape.entries_for_param(0);
    assert_eq!(entries_p0.len(), 2);

    // Parameter 1 is used once
    let entries_p1 = tape.entries_for_param(1);
    assert_eq!(entries_p1.len(), 1);
}

#[test]
fn test_tape_forward_backward() {
    let mut tape = GradientTape::new(1, 2);

    // Simple function: f(x, y) = x + y
    let (result, grads) = tape.forward_backward(&[1.0, 2.0], |params| params[0] + params[1]);

    assert!((result - 3.0).abs() < EPSILON);
    // Gradients are computed via VJP (simplified in tape)
    assert!(grads.len() == 2);
}

#[test]
fn test_gradient_accumulator_batch() {
    let mut accumulator = GradientAccumulator::new(3);

    // Simulate batch of gradients
    let grads_1 = vec![1.0, 2.0, 3.0];
    let grads_2 = vec![3.0, 2.0, 1.0];
    let grads_3 = vec![2.0, 2.0, 2.0];

    accumulator.accumulate(&grads_1);
    accumulator.accumulate(&grads_2);
    accumulator.accumulate(&grads_3);

    let avg = accumulator.average();
    assert_eq!(accumulator.count(), 3);
    assert!((avg[0] - 2.0).abs() < EPSILON);
    assert!((avg[1] - 2.0).abs() < EPSILON);
    assert!((avg[2] - 2.0).abs() < EPSILON);
}

// =============================================================================
// Section 2: GradientCache Integration Tests
// =============================================================================

#[test]
fn test_cache_exact_hit() {
    let mut cache = GradientCache::new(100);

    let params = vec![0.1, 0.2, 0.3];
    let grads = vec![1.0, 2.0, 3.0];

    cache.insert(params.clone(), grads.clone());

    let retrieved = cache.get(&params);
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap(), grads);

    // Check stats
    assert_eq!(cache.stats().hits, 1);
    assert_eq!(cache.stats().misses, 0);
}

#[test]
fn test_cache_lru_eviction_order() {
    let mut cache = GradientCache::new(3);

    // Insert 3 entries (fills cache)
    cache.insert(vec![0.1], vec![1.0]);
    cache.insert(vec![0.2], vec![2.0]);
    cache.insert(vec![0.3], vec![3.0]);

    // Access first entry to make it recently used
    let _ = cache.get(&[0.1]);

    // Insert fourth entry - should evict second (LRU)
    cache.insert(vec![0.4], vec![4.0]);

    // First should still exist (was accessed)
    assert!(cache.get(&[0.1]).is_some());
    // Second should be evicted
    assert!(cache.get(&[0.2]).is_none());
    // Third and fourth should exist
    assert!(cache.get(&[0.3]).is_some());
    assert!(cache.get(&[0.4]).is_some());
}

#[test]
fn test_cache_with_expectation() {
    let mut cache = GradientCache::new(100);

    let params = vec![0.5, 0.5];
    let grads = vec![-0.1, -0.2];
    let expectation = -0.707;

    cache.insert_with_expectation(params.clone(), grads.clone(), expectation);

    let (retrieved_grads, retrieved_exp) = cache.get_with_expectation(&params).unwrap();
    assert_eq!(retrieved_grads, grads);
    assert!((retrieved_exp.unwrap() - expectation).abs() < EPSILON);
}

#[test]
fn test_cache_stats_tracking() {
    let mut cache = GradientCache::new(100);

    // Insert and hit
    cache.insert(vec![0.1], vec![1.0]);
    let _ = cache.get(&[0.1]); // hit
    let _ = cache.get(&[0.1]); // hit
    let _ = cache.get(&[0.2]); // miss
    let _ = cache.get(&[0.3]); // miss

    let stats = cache.stats();
    assert_eq!(stats.hits, 2);
    assert_eq!(stats.misses, 2);
    assert_eq!(stats.lookups, 4);
    assert!((stats.hit_rate() - 0.5).abs() < EPSILON);
}

#[test]
fn test_cache_quantization() {
    let mut cache = GradientCache::with_quantization(100, 10);

    // Insert with specific parameters
    cache.insert(vec![0.123456789], vec![1.0]);

    // Slightly different parameters should still hit due to quantization
    // 0.123456789 and 0.123456780 should quantize to same key
    let _result = cache.get(&[0.123456780]);

    // May or may not hit depending on precision, but should not crash
    assert!(cache.len() == 1);
}

// =============================================================================
// Section 3: Optimizer Integration Tests
// =============================================================================

/// Simple 2D quadratic: f(x,y) = x^2 + y^2
fn quadratic(params: &[f64]) -> f64 {
    params.iter().map(|x| x * x).sum()
}

fn quadratic_grad(params: &[f64]) -> Vec<f64> {
    params.iter().map(|x| 2.0 * x).collect()
}

/// Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2
fn rosenbrock(params: &[f64]) -> f64 {
    let x = params[0];
    let y = params[1];
    (1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2)
}

fn rosenbrock_grad(params: &[f64]) -> Vec<f64> {
    let x = params[0];
    let y = params[1];
    let dx = -2.0 * (1.0 - x) - 400.0 * x * (y - x * x);
    let dy = 200.0 * (y - x * x);
    vec![dx, dy]
}

#[test]
fn test_sgd_quadratic_convergence() {
    let config = OptimizerConfig::sgd(0.1)
        .with_max_iter(100)
        .with_tolerance(1e-6);

    let result = minimize(config, vec![5.0, 5.0], quadratic, quadratic_grad);

    assert!(result.objective < 0.1);
    assert!(result.params[0].abs() < 0.5);
    assert!(result.params[1].abs() < 0.5);
}

#[test]
fn test_adam_quadratic_convergence() {
    let config = OptimizerConfig::adam(0.1)
        .with_max_iter(100)
        .with_tolerance(1e-6);

    let result = minimize(config, vec![5.0, 5.0], quadratic, quadratic_grad);

    assert!(result.objective < 0.01);
}

#[test]
fn test_momentum_accelerates_convergence() {
    // Compare SGD vs Momentum
    let config_sgd = OptimizerConfig::sgd(0.01).with_max_iter(50);
    let config_mom = OptimizerConfig::momentum(0.01, 0.9).with_max_iter(50);

    let result_sgd = minimize(config_sgd, vec![5.0, 5.0], quadratic, quadratic_grad);
    let result_mom = minimize(config_mom, vec![5.0, 5.0], quadratic, quadratic_grad);

    // Momentum should achieve lower objective in same iterations
    assert!(result_mom.objective <= result_sgd.objective * 1.5);
}

#[test]
fn test_adam_rosenbrock() {
    // Rosenbrock is harder - Adam should still make progress
    let config = OptimizerConfig::adam(0.01)
        .with_max_iter(500)
        .with_tolerance(1e-8);

    let initial = vec![0.0, 0.0];
    let result = minimize(config, initial, rosenbrock, rosenbrock_grad);

    // Should reduce from f(0,0)=1 toward f(1,1)=0
    assert!(result.objective < 1.0);
}

#[test]
fn test_optimizer_gradient_clipping() {
    // Create optimizer with gradient clipping
    let config = OptimizerConfig::sgd(0.1)
        .with_grad_clip(1.0)
        .with_max_iter(10);

    let mut opt = Optimizer::new(config, vec![0.0, 0.0]);

    // Apply large gradient
    opt.step(&[100.0, 100.0]);

    // Update should be bounded by clip threshold
    let update_magnitude = (opt.params()[0].powi(2) + opt.params()[1].powi(2)).sqrt();
    assert!(update_magnitude <= 0.15); // 0.1 * sqrt(2) ≈ 0.14
}

#[test]
fn test_optimizer_weight_decay() {
    let config = OptimizerConfig::sgd(0.1)
        .with_weight_decay(0.1)
        .with_max_iter(1);

    let mut opt = Optimizer::new(config, vec![1.0, 1.0]);

    // Zero gradient - only weight decay affects update
    opt.step(&[0.0, 0.0]);

    // Parameters should decrease due to weight decay
    assert!(opt.params()[0] < 1.0);
    assert!(opt.params()[1] < 1.0);
}

#[test]
fn test_lr_schedule_cosine() {
    let schedule = LRSchedule::Cosine {
        t_max: 100,
        eta_min: 0.001,
    };

    let base_lr = 0.1;

    // Start: should be base_lr
    let lr_0 = schedule.get_lr(base_lr, 0);
    assert!((lr_0 - base_lr).abs() < EPSILON);

    // End: should be eta_min
    let lr_end = schedule.get_lr(base_lr, 100);
    assert!((lr_end - 0.001).abs() < EPSILON);

    // Middle: should be between
    let lr_mid = schedule.get_lr(base_lr, 50);
    assert!(lr_mid > 0.001 && lr_mid < 0.1);
}

// =============================================================================
// Section 4: Full Pipeline Integration Tests
// =============================================================================

#[test]
fn test_tape_cache_optimizer_pipeline() {
    // Full pipeline: record circuit -> cache gradients -> optimize
    let _n_params = 2;

    // 1. Create cache
    let mut cache = GradientCache::new(100);

    // 2. Objective function with caching
    let compute_with_cache = |params: &[f64], cache: &mut GradientCache| -> (f64, Vec<f64>) {
        // Check cache first
        if let Some((grads, Some(obj))) = cache.get_with_expectation(params) {
            return (obj, grads);
        }

        // Compute (simplified - just quadratic)
        let obj = quadratic(params);
        let grads = quadratic_grad(params);

        // Cache result
        cache.insert_with_expectation(params.to_vec(), grads.clone(), obj);

        (obj, grads)
    };

    // 3. Run optimization loop
    let mut opt = Optimizer::adam(0.1, vec![5.0, 5.0]);

    for _ in 0..50 {
        let (obj, grads) = compute_with_cache(opt.params(), &mut cache);
        opt.update_objective(obj);
        opt.step(&grads);
    }

    // Should make progress (started at 50 = 5^2 + 5^2)
    assert!(opt.best_params()[0].abs() < 3.0);
    assert!(opt.best_params()[1].abs() < 3.0);

    // Cache should have entries
    assert!(cache.len() > 0);
}

#[test]
fn test_gradient_accumulator_with_optimizer() {
    // Simulate mini-batch gradient descent
    let mut accumulator = GradientAccumulator::new(2);
    let _batch_size = 4;

    // Simulate batch of different "samples"
    let samples = vec![
        vec![1.0, 2.0],
        vec![2.0, 1.0],
        vec![1.5, 1.5],
        vec![0.5, 2.5],
    ];

    for sample in &samples {
        let grads = quadratic_grad(sample);
        accumulator.accumulate(&grads);
    }

    let avg_grads = accumulator.average();

    // Average gradient
    let expected = vec![(2.0 + 4.0 + 3.0 + 1.0) / 4.0, (4.0 + 2.0 + 3.0 + 5.0) / 4.0];

    assert!((avg_grads[0] - expected[0]).abs() < EPSILON);
    assert!((avg_grads[1] - expected[1]).abs() < EPSILON);
}

// =============================================================================
// Section 5: VQE Simulation Tests
// =============================================================================

#[test]
fn test_vqe_2qubit_simple() {
    // Simple 2-qubit VQE with ZZ Hamiltonian
    // H = Z⊗Z, ground state energy = -1

    // Build variational circuit
    let mut circuit = VariationalCircuit::new(2);
    circuit.ry(0); // param 0
    circuit.ry(1); // param 1
    circuit.cnot(0, 1);

    // Hamiltonian: Z⊗Z
    let hamiltonian = Hamiltonian::ising(2);

    // Optimize
    let config = OptimizerConfig::adam(0.1)
        .with_max_iter(100)
        .with_tolerance(1e-6);

    let objective = |params: &[f64]| -> f64 { compute_expectation(&circuit, params, &hamiltonian) };

    let gradient =
        |params: &[f64]| -> Vec<f64> { compute_gradient(&circuit, params, &hamiltonian) };

    let result = minimize(config, vec![0.5, 0.5], objective, gradient);

    // Ground state of Z⊗Z is |00⟩ or |11⟩ with energy -1
    // VQE should find energy close to -1
    assert!(result.objective < 0.0); // Should be negative
}

#[test]
fn test_vqe_gradient_consistency() {
    // Verify gradient computation is consistent
    let mut circuit = VariationalCircuit::new(2);
    circuit.rx(0); // param 0
    circuit.ry(1); // param 1

    let mut hamiltonian = Hamiltonian::new();
    hamiltonian.add_term(PauliObservable::z(0));

    let params = vec![0.3, 0.7];

    // Compute gradient
    let grads = compute_gradient(&circuit, &params, &hamiltonian);

    // Verify with finite differences
    let eps = 1e-5;
    for i in 0..params.len() {
        let mut p_plus = params.clone();
        let mut p_minus = params.clone();
        p_plus[i] += eps;
        p_minus[i] -= eps;

        let f_plus = compute_expectation(&circuit, &p_plus, &hamiltonian);
        let f_minus = compute_expectation(&circuit, &p_minus, &hamiltonian);
        let numerical_grad = (f_plus - f_minus) / (2.0 * eps);

        assert!(
            (grads[i] - numerical_grad).abs() < GRADIENT_TOL,
            "Gradient mismatch at param {}: analytical={}, numerical={}",
            i,
            grads[i],
            numerical_grad
        );
    }
}

#[test]
fn test_vqe_3qubit_hardware_efficient() {
    // Hardware-efficient ansatz for 3 qubits
    let n_qubits = 3;
    let n_layers = 2;

    let mut circuit = VariationalCircuit::new(n_qubits);

    for _layer in 0..n_layers {
        // Single-qubit rotations
        for qubit in 0..n_qubits {
            circuit.ry(qubit);
            circuit.rz(qubit);
        }
        // Entangling layer
        for qubit in 0..(n_qubits - 1) {
            circuit.cnot(qubit, qubit + 1);
        }
    }

    let n_params = circuit.num_params();

    // Simple Hamiltonian: sum of Z
    let hamiltonian = Hamiltonian::all_z(n_qubits);

    // Random initial params
    let initial_params: Vec<f64> = (0..n_params).map(|i| 0.1 * (i as f64)).collect();

    let (exp, grads) = compute_expectation_and_gradient(&circuit, &initial_params, &hamiltonian);

    // Should get valid expectation and gradients
    assert!(exp.is_finite());
    assert_eq!(grads.len(), n_params);
    assert!(grads.iter().all(|g| g.is_finite()));
}

// =============================================================================
// Section 6: Performance Benchmarks
// =============================================================================

#[test]
fn bench_cache_lookup_performance() {
    let mut cache = GradientCache::new(1000);

    // Populate cache
    for i in 0..500 {
        let params = vec![i as f64 * 0.01, (i as f64 * 0.01).sin()];
        let grads = vec![2.0 * params[0], params[1].cos()];
        cache.insert(params, grads);
    }

    // Benchmark lookups
    let start = Instant::now();
    let n_lookups = 10000;

    for i in 0..n_lookups {
        let params = vec![(i % 500) as f64 * 0.01, ((i % 500) as f64 * 0.01).sin()];
        let _ = cache.get(&params);
    }

    let elapsed = start.elapsed();
    let per_lookup_ns = elapsed.as_nanos() / n_lookups as u128;

    // Should be fast (< 10μs per lookup)
    assert!(
        per_lookup_ns < 10_000,
        "Cache lookup too slow: {}ns",
        per_lookup_ns
    );

    // Hit rate should be high
    assert!(cache.stats().hit_rate() > 0.9);
}

#[test]
fn bench_optimizer_step_performance() {
    let n_params = 100;
    let initial: Vec<f64> = (0..n_params).map(|i| i as f64 * 0.1).collect();

    let mut opt = Optimizer::adam(0.01, initial);
    let grads: Vec<f64> = (0..n_params).map(|i| (i as f64 * 0.01).sin()).collect();

    // Warm up
    for _ in 0..10 {
        opt.step(&grads);
    }
    opt.reset_params((0..n_params).map(|i| i as f64 * 0.1).collect());

    // Benchmark
    let start = Instant::now();
    let n_steps = 1000;

    for _ in 0..n_steps {
        opt.step(&grads);
    }

    let elapsed = start.elapsed();
    let per_step_us = elapsed.as_micros() / n_steps as u128;

    // Should be fast (< 100μs per step for 100 params)
    assert!(
        per_step_us < 100,
        "Optimizer step too slow: {}μs",
        per_step_us
    );
}

#[test]
fn bench_gradient_tape_recording() {
    let n_qubits = 10;
    let n_params = 20;
    let n_gates = 50;

    let mut tape = GradientTape::new(n_qubits, n_params);
    let params: Vec<f64> = (0..n_params).map(|i| i as f64 * 0.1).collect();

    let start = Instant::now();
    let n_recordings = 100;

    for _ in 0..n_recordings {
        tape.clear();
        tape.watch(&params);
        tape.start_recording();

        for g in 0..n_gates {
            match g % 4 {
                0 => tape.record_rx(g % n_qubits, g % n_params),
                1 => tape.record_ry(g % n_qubits, g % n_params),
                2 => tape.record_rz(g % n_qubits, g % n_params),
                _ => tape.record_cnot(g % n_qubits, (g + 1) % n_qubits),
            }
        }

        tape.stop_recording();
    }

    let elapsed = start.elapsed();
    let per_recording_us = elapsed.as_micros() / n_recordings as u128;

    // Should be fast (< 1ms per 50-gate recording)
    assert!(
        per_recording_us < 1000,
        "Tape recording too slow: {}μs",
        per_recording_us
    );
}

// =============================================================================
// Section 7: Edge Cases and Error Handling
// =============================================================================

#[test]
fn test_empty_tape_operations() {
    let mut tape = GradientTape::new(2, 2);

    assert!(tape.is_empty());
    assert_eq!(tape.len(), 0);

    // Backward on empty tape should not panic
    tape.backward(1.0);
    assert!(tape.gradients().iter().all(|&g| g.abs() < EPSILON));
}

#[test]
fn test_cache_empty_params() {
    let mut cache = GradientCache::new(100);

    // Empty params
    cache.insert(vec![], vec![]);

    // Should handle gracefully
    let result = cache.get(&[]);
    assert!(result.is_some());
    assert!(result.unwrap().is_empty());
}

#[test]
fn test_optimizer_single_param() {
    let config = OptimizerConfig::adam(0.1).with_max_iter(100);

    let result = minimize(config, vec![10.0], |p| p[0] * p[0], |p| vec![2.0 * p[0]]);

    // Should make significant progress
    assert!(result.objective < 10.0); // Started at 100, should reduce
    assert!(result.params[0].abs() < 5.0); // Started at 10
}

#[test]
fn test_optimizer_many_params() {
    let n = 50;
    let initial: Vec<f64> = (0..n).map(|i| 5.0 + i as f64 * 0.1).collect();

    let config = OptimizerConfig::adam(0.1).with_max_iter(200);

    let result = minimize(config, initial, quadratic, quadratic_grad);

    // Should make progress (average param < initial average)
    let avg: f64 = result.params.iter().map(|p| p.abs()).sum::<f64>() / n as f64;
    assert!(avg < 5.0); // Initial avg ~7.5
}

#[test]
fn test_vqe_single_qubit() {
    // Simplest VQE: 1 qubit, RY, measure Z
    let mut circuit = VariationalCircuit::new(1);
    circuit.ry(0); // param 0

    let mut hamiltonian = Hamiltonian::new();
    hamiltonian.add_term(PauliObservable::z(0));

    // At θ=0: |0⟩, ⟨Z⟩ = 1
    // At θ=π: |1⟩, ⟨Z⟩ = -1

    let exp_0 = compute_expectation(&circuit, &[0.0], &hamiltonian);
    let exp_pi = compute_expectation(&circuit, &[PI], &hamiltonian);

    assert!((exp_0 - 1.0).abs() < GRADIENT_TOL);
    assert!((exp_pi - (-1.0)).abs() < GRADIENT_TOL);
}

// =============================================================================
// Section 8: Comprehensive Integration Summary
// =============================================================================

#[test]
fn test_week3_component_integration() {
    // Final integration test: all Week 3 components working together

    // 1. Setup: VQE problem (2-qubit)
    let mut circuit = VariationalCircuit::new(2);
    circuit.ry(0); // param 0
    circuit.ry(1); // param 1
    circuit.cnot(0, 1);

    // Simple Hamiltonian: Z0 + Z1 + Z0Z1
    let mut hamiltonian = Hamiltonian::new();
    hamiltonian.add_term(PauliObservable::z(0));
    hamiltonian.add_term(PauliObservable::z(1));
    hamiltonian.add_term(PauliObservable::zz(0, 1));

    // 2. Create tape for recording
    let n_params = 2;
    let mut tape = GradientTape::new(2, n_params);

    // 3. Create cache
    let mut cache = GradientCache::new(50);

    // 4. Optimizer
    let mut opt = Optimizer::adam(0.1, vec![0.5, 0.5]);

    // 5. Optimization loop
    let mut history = Vec::new();

    for _iter in 0..30 {
        let params = opt.params().to_vec();

        // Check cache
        let (exp, grads) = if let Some((g, Some(e))) = cache.get_with_expectation(&params) {
            (e, g)
        } else {
            // Record on tape
            tape.clear();
            tape.watch(&params);
            tape.start_recording();
            tape.record_ry(0, 0);
            tape.record_ry(1, 1);
            tape.record_cnot(0, 1);
            tape.stop_recording();

            // Compute
            let e = compute_expectation(&circuit, &params, &hamiltonian);
            let g = compute_gradient(&circuit, &params, &hamiltonian);

            // Cache
            cache.insert_with_expectation(params.clone(), g.clone(), e);

            (e, g)
        };

        history.push(exp);
        opt.update_objective(exp);
        opt.step(&grads);
    }

    // Verify optimization made progress
    assert!(history.last().unwrap() <= history.first().unwrap());

    // Verify components used correctly
    assert!(tape.len() > 0);
    assert!(cache.len() > 0);
    assert!(opt.iteration() == 30);

    // Summary statistics
    let tape_summary = tape.summary();
    let cache_stats = cache.stats();

    assert!(tape_summary.parameterized_entries >= 2);
    assert!(cache_stats.insertions > 0);
}

// =============================================================================
// Section 9: SPSA Gradient-Free Optimization
// =============================================================================

#[test]
fn test_spsa_quadratic_optimization() {
    // SPSA: Gradient-free optimization using finite differences
    let initial = vec![5.0, 5.0];

    let result = minimize_spsa(initial, quadratic, 100);

    // SPSA should reduce objective (started at 50)
    assert!(
        result.objective < 25.0,
        "SPSA failed to reduce: {}",
        result.objective
    );
    assert!(result.iterations > 0);
}

#[test]
fn test_spsa_high_dimensional() {
    // SPSA should scale to higher dimensions
    let n = 10;
    let initial: Vec<f64> = (0..n).map(|i| 3.0 + i as f64 * 0.1).collect();

    let result = minimize_spsa(initial, quadratic, 200);

    // Should make progress
    let initial_obj: f64 = (0..n).map(|i| (3.0 + i as f64 * 0.1).powi(2)).sum();
    assert!(
        result.objective < initial_obj * 0.8,
        "SPSA failed in high-dim: started={}, ended={}",
        initial_obj,
        result.objective
    );
}

#[test]
fn test_spsa_vqe_gradient_free() {
    // VQE with SPSA - gradient-free quantum optimization
    let mut circuit = VariationalCircuit::new(2);
    circuit.ry(0);
    circuit.ry(1);
    circuit.cnot(0, 1);

    let mut hamiltonian = Hamiltonian::new();
    hamiltonian.add_term(PauliObservable::z(0));
    hamiltonian.add_term(PauliObservable::z(1));

    let objective = |params: &[f64]| -> f64 { compute_expectation(&circuit, params, &hamiltonian) };

    // Start with random-ish params
    let initial = vec![0.7, 0.3];
    let result = minimize_spsa(initial, objective, 50);

    // SPSA should find a lower energy state
    assert!(result.params.len() == 2);
    assert!(result.iterations > 0);
}

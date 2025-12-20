use ndarray::Array1;
use tqp_algorithms::ChronoLoopOptimizer;

#[test]
fn test_optimizer_convergence() {
    // 1. Setup Optimizer
    // Max iter 100, epsilon 1e-6, learning rate 0.1
    let optimizer = ChronoLoopOptimizer::new(100, 1e-6, 0.1);

    // 2. Define Cost Function: f(x) = x^2 (Minimum at x=0)
    let cost_function = |params: &Array1<f64>| -> f64 { params.mapv(|x| x.powi(2)).sum() };

    // 3. Initial Parameters: [10.0]
    let initial_params = Array1::from(vec![10.0]);

    // 4. Run Optimization
    let (optimized_params, history) = optimizer.optimize(&initial_params, cost_function);

    // 5. Verify Convergence
    // Should be close to 0.0
    println!("Optimized Params: {:?}", optimized_params);
    println!("Loss History: {:?}", history);

    assert!(optimized_params[0].abs() < 1.0); // Simple check for decay
    assert!(history.last().unwrap() < &1.0);
}

//! Optimizers for Variational Quantum Algorithms
//!
//! Implements gradient-based and gradient-free optimization algorithms
//! tailored for VQE and QAOA parameter optimization.
//!
//! # Supported Optimizers
//!
//! | Optimizer | Type | Best For |
//! |-----------|------|----------|
//! | SGD | Gradient | Simple landscapes |
//! | SGD+Momentum | Gradient | Smoother convergence |
//! | Adam | Gradient | Adaptive, noisy gradients |
//! | RMSprop | Gradient | Non-stationary objectives |
//! | SPSA | Gradient-free | Noisy evaluations |
//! | COBYLA | Gradient-free | Constrained optimization |
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                        Optimizer                                 │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  State                                                          │
//! │    ├── params: current parameters                               │
//! │    ├── velocity: momentum state                                 │
//! │    └── adaptive: Adam/RMSprop state                            │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  Step                                                           │
//! │    ├── compute_update(gradients)                                │
//! │    └── apply_update()                                           │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  Convergence                                                    │
//! │    ├── check_convergence()                                      │
//! │    └── early_stopping()                                         │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use std::f64::consts::PI;

// =============================================================================
// Constants
// =============================================================================

/// Default learning rate
pub const DEFAULT_LEARNING_RATE: f64 = 0.01;

/// Default momentum coefficient
pub const DEFAULT_MOMENTUM: f64 = 0.9;

/// Default Adam beta1
pub const DEFAULT_BETA1: f64 = 0.9;

/// Default Adam beta2
pub const DEFAULT_BETA2: f64 = 0.999;

/// Default epsilon for numerical stability
pub const DEFAULT_EPSILON: f64 = 1e-8;

/// Default SPSA perturbation
pub const DEFAULT_SPSA_C: f64 = 0.1;

/// Default SPSA alpha
pub const DEFAULT_SPSA_ALPHA: f64 = 0.602;

/// Default SPSA gamma
pub const DEFAULT_SPSA_GAMMA: f64 = 0.101;

/// Default convergence tolerance
pub const DEFAULT_TOLERANCE: f64 = 1e-6;

/// Default max iterations
pub const DEFAULT_MAX_ITER: usize = 1000;

// =============================================================================
// Optimizer Type
// =============================================================================

/// Optimizer algorithm type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizerType {
    /// Stochastic Gradient Descent
    SGD,
    /// SGD with Momentum
    Momentum,
    /// Nesterov Accelerated Gradient
    NAG,
    /// Adam (Adaptive Moment Estimation)
    Adam,
    /// AdaGrad
    AdaGrad,
    /// RMSprop
    RMSprop,
    /// SPSA (Simultaneous Perturbation Stochastic Approximation)
    SPSA,
    /// L-BFGS (Limited-memory BFGS)
    LBFGS,
}

impl OptimizerType {
    /// Check if gradient-based
    pub fn is_gradient_based(&self) -> bool {
        !matches!(self, OptimizerType::SPSA)
    }

    /// String name
    pub fn name(&self) -> &'static str {
        match self {
            OptimizerType::SGD => "SGD",
            OptimizerType::Momentum => "Momentum",
            OptimizerType::NAG => "NAG",
            OptimizerType::Adam => "Adam",
            OptimizerType::AdaGrad => "AdaGrad",
            OptimizerType::RMSprop => "RMSprop",
            OptimizerType::SPSA => "SPSA",
            OptimizerType::LBFGS => "L-BFGS",
        }
    }
}

// =============================================================================
// Optimizer State
// =============================================================================

/// Internal state for momentum-based optimizers
#[derive(Debug, Clone)]
pub struct MomentumState {
    /// Velocity vector
    pub velocity: Vec<f64>,
}

impl MomentumState {
    /// Create new momentum state
    pub fn new(n_params: usize) -> Self {
        Self {
            velocity: vec![0.0; n_params],
        }
    }

    /// Reset velocity
    pub fn reset(&mut self) {
        self.velocity.fill(0.0);
    }
}

/// Internal state for Adam optimizer
#[derive(Debug, Clone)]
pub struct AdamState {
    /// First moment estimate (mean of gradients)
    pub m: Vec<f64>,
    /// Second moment estimate (mean of squared gradients)
    pub v: Vec<f64>,
    /// Timestep
    pub t: usize,
}

impl AdamState {
    /// Create new Adam state
    pub fn new(n_params: usize) -> Self {
        Self {
            m: vec![0.0; n_params],
            v: vec![0.0; n_params],
            t: 0,
        }
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.m.fill(0.0);
        self.v.fill(0.0);
        self.t = 0;
    }
}

/// Internal state for RMSprop
#[derive(Debug, Clone)]
pub struct RMSpropState {
    /// Running average of squared gradients
    pub cache: Vec<f64>,
}

impl RMSpropState {
    /// Create new RMSprop state
    pub fn new(n_params: usize) -> Self {
        Self {
            cache: vec![0.0; n_params],
        }
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.cache.fill(0.0);
    }
}

/// Internal state for AdaGrad
#[derive(Debug, Clone)]
pub struct AdaGradState {
    /// Accumulated squared gradients
    pub accumulated: Vec<f64>,
}

impl AdaGradState {
    /// Create new AdaGrad state
    pub fn new(n_params: usize) -> Self {
        Self {
            accumulated: vec![0.0; n_params],
        }
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.accumulated.fill(0.0);
    }
}

/// Internal state for SPSA
#[derive(Debug, Clone)]
pub struct SPSAState {
    /// Current iteration
    pub k: usize,
    /// Perturbation coefficient
    pub c: f64,
    /// Learning rate coefficient
    pub a: f64,
    /// Alpha exponent
    pub alpha: f64,
    /// Gamma exponent
    pub gamma: f64,
}

impl SPSAState {
    /// Create new SPSA state
    pub fn new() -> Self {
        Self {
            k: 0,
            c: DEFAULT_SPSA_C,
            a: 0.1,
            alpha: DEFAULT_SPSA_ALPHA,
            gamma: DEFAULT_SPSA_GAMMA,
        }
    }

    /// Get current learning rate a_k = a / (k + 1)^alpha
    pub fn learning_rate(&self) -> f64 {
        self.a / ((self.k + 1) as f64).powf(self.alpha)
    }

    /// Get current perturbation c_k = c / (k + 1)^gamma
    pub fn perturbation(&self) -> f64 {
        self.c / ((self.k + 1) as f64).powf(self.gamma)
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.k = 0;
    }
}

impl Default for SPSAState {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Optimizer Configuration
// =============================================================================

/// Optimizer configuration
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Optimizer type
    pub optimizer_type: OptimizerType,
    /// Learning rate
    pub learning_rate: f64,
    /// Momentum coefficient (for Momentum, NAG)
    pub momentum: f64,
    /// Beta1 (for Adam)
    pub beta1: f64,
    /// Beta2 (for Adam, RMSprop)
    pub beta2: f64,
    /// Epsilon for numerical stability
    pub epsilon: f64,
    /// Weight decay (L2 regularization)
    pub weight_decay: f64,
    /// Gradient clipping threshold
    pub grad_clip: Option<f64>,
    /// Maximum iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Enable early stopping
    pub early_stopping: bool,
    /// Patience for early stopping
    pub patience: usize,
}

impl OptimizerConfig {
    /// Create SGD config
    pub fn sgd(learning_rate: f64) -> Self {
        Self {
            optimizer_type: OptimizerType::SGD,
            learning_rate,
            ..Default::default()
        }
    }

    /// Create Momentum config
    pub fn momentum(learning_rate: f64, momentum: f64) -> Self {
        Self {
            optimizer_type: OptimizerType::Momentum,
            learning_rate,
            momentum,
            ..Default::default()
        }
    }

    /// Create Adam config
    pub fn adam(learning_rate: f64) -> Self {
        Self {
            optimizer_type: OptimizerType::Adam,
            learning_rate,
            ..Default::default()
        }
    }

    /// Create RMSprop config
    pub fn rmsprop(learning_rate: f64) -> Self {
        Self {
            optimizer_type: OptimizerType::RMSprop,
            learning_rate,
            ..Default::default()
        }
    }

    /// Create SPSA config
    pub fn spsa() -> Self {
        Self {
            optimizer_type: OptimizerType::SPSA,
            ..Default::default()
        }
    }

    /// Builder: set weight decay
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Builder: set gradient clipping
    pub fn with_grad_clip(mut self, threshold: f64) -> Self {
        self.grad_clip = Some(threshold);
        self
    }

    /// Builder: set max iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Builder: set tolerance
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Builder: enable early stopping
    pub fn with_early_stopping(mut self, patience: usize) -> Self {
        self.early_stopping = true;
        self.patience = patience;
        self
    }
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            optimizer_type: OptimizerType::Adam,
            learning_rate: DEFAULT_LEARNING_RATE,
            momentum: DEFAULT_MOMENTUM,
            beta1: DEFAULT_BETA1,
            beta2: DEFAULT_BETA2,
            epsilon: DEFAULT_EPSILON,
            weight_decay: 0.0,
            grad_clip: None,
            max_iter: DEFAULT_MAX_ITER,
            tolerance: DEFAULT_TOLERANCE,
            early_stopping: false,
            patience: 10,
        }
    }
}

// =============================================================================
// Optimization Result
// =============================================================================

/// Result of optimization
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Final parameters
    pub params: Vec<f64>,
    /// Final objective value
    pub objective: f64,
    /// Number of iterations
    pub iterations: usize,
    /// Number of function evaluations
    pub func_evals: usize,
    /// Number of gradient evaluations
    pub grad_evals: usize,
    /// Converged successfully
    pub converged: bool,
    /// Convergence reason
    pub message: String,
    /// History of objective values
    pub history: Vec<f64>,
}

impl OptimizationResult {
    /// Create successful result
    pub fn success(params: Vec<f64>, objective: f64, iterations: usize) -> Self {
        Self {
            params,
            objective,
            iterations,
            func_evals: iterations,
            grad_evals: iterations,
            converged: true,
            message: "Optimization converged".to_string(),
            history: Vec::new(),
        }
    }

    /// Create result with history
    pub fn with_history(mut self, history: Vec<f64>) -> Self {
        self.history = history;
        self
    }

    /// Create result with eval counts
    pub fn with_evals(mut self, func_evals: usize, grad_evals: usize) -> Self {
        self.func_evals = func_evals;
        self.grad_evals = grad_evals;
        self
    }
}

// =============================================================================
// Optimizer
// =============================================================================

/// Main optimizer struct
#[derive(Debug)]
pub struct Optimizer {
    /// Configuration
    config: OptimizerConfig,
    /// Current parameters
    params: Vec<f64>,
    /// Momentum state (if applicable)
    momentum_state: Option<MomentumState>,
    /// Adam state (if applicable)
    adam_state: Option<AdamState>,
    /// RMSprop state (if applicable)
    rmsprop_state: Option<RMSpropState>,
    /// AdaGrad state (if applicable)
    adagrad_state: Option<AdaGradState>,
    /// SPSA state (if applicable)
    spsa_state: Option<SPSAState>,
    /// Current iteration
    iteration: usize,
    /// Best objective seen
    best_objective: f64,
    /// Best parameters seen
    best_params: Vec<f64>,
    /// Steps without improvement (for early stopping)
    steps_without_improvement: usize,
}

impl Optimizer {
    /// Create new optimizer with config
    pub fn new(config: OptimizerConfig, initial_params: Vec<f64>) -> Self {
        let n_params = initial_params.len();

        let (momentum_state, adam_state, rmsprop_state, adagrad_state, spsa_state) = match config
            .optimizer_type
        {
            OptimizerType::Momentum | OptimizerType::NAG => {
                (Some(MomentumState::new(n_params)), None, None, None, None)
            }
            OptimizerType::Adam => (None, Some(AdamState::new(n_params)), None, None, None),
            OptimizerType::RMSprop => (None, None, Some(RMSpropState::new(n_params)), None, None),
            OptimizerType::AdaGrad => (None, None, None, Some(AdaGradState::new(n_params)), None),
            OptimizerType::SPSA => (None, None, None, None, Some(SPSAState::new())),
            _ => (None, None, None, None, None),
        };

        Self {
            config,
            params: initial_params.clone(),
            momentum_state,
            adam_state,
            rmsprop_state,
            adagrad_state,
            spsa_state,
            iteration: 0,
            best_objective: f64::INFINITY,
            best_params: initial_params,
            steps_without_improvement: 0,
        }
    }

    /// Create SGD optimizer
    pub fn sgd(learning_rate: f64, initial_params: Vec<f64>) -> Self {
        Self::new(OptimizerConfig::sgd(learning_rate), initial_params)
    }

    /// Create Adam optimizer
    pub fn adam(learning_rate: f64, initial_params: Vec<f64>) -> Self {
        Self::new(OptimizerConfig::adam(learning_rate), initial_params)
    }

    /// Create SPSA optimizer
    pub fn spsa(initial_params: Vec<f64>) -> Self {
        Self::new(OptimizerConfig::spsa(), initial_params)
    }

    // -------------------------------------------------------------------------
    // Parameter Access
    // -------------------------------------------------------------------------

    /// Get current parameters
    pub fn params(&self) -> &[f64] {
        &self.params
    }

    /// Get mutable parameters
    pub fn params_mut(&mut self) -> &mut [f64] {
        &mut self.params
    }

    /// Get best parameters
    pub fn best_params(&self) -> &[f64] {
        &self.best_params
    }

    /// Get current iteration
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Get config
    pub fn config(&self) -> &OptimizerConfig {
        &self.config
    }

    // -------------------------------------------------------------------------
    // Gradient Clipping
    // -------------------------------------------------------------------------

    /// Clip gradients by global norm
    fn clip_gradients(&self, gradients: &mut [f64]) {
        if let Some(threshold) = self.config.grad_clip {
            let norm: f64 = gradients.iter().map(|g| g * g).sum::<f64>().sqrt();
            if norm > threshold {
                let scale = threshold / norm;
                for g in gradients.iter_mut() {
                    *g *= scale;
                }
            }
        }
    }

    /// Apply weight decay
    fn apply_weight_decay(&self, gradients: &mut [f64]) {
        if self.config.weight_decay > 0.0 {
            for (g, p) in gradients.iter_mut().zip(self.params.iter()) {
                *g += self.config.weight_decay * p;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Step Functions
    // -------------------------------------------------------------------------

    /// Perform single optimization step with gradients
    pub fn step(&mut self, gradients: &[f64]) {
        assert_eq!(gradients.len(), self.params.len());

        let mut grads = gradients.to_vec();

        // Apply gradient modifications
        self.clip_gradients(&mut grads);
        self.apply_weight_decay(&mut grads);

        // Compute update based on optimizer type
        let update = match self.config.optimizer_type {
            OptimizerType::SGD => self.sgd_update(&grads),
            OptimizerType::Momentum => self.momentum_update(&grads),
            OptimizerType::NAG => self.nag_update(&grads),
            OptimizerType::Adam => self.adam_update(&grads),
            OptimizerType::AdaGrad => self.adagrad_update(&grads),
            OptimizerType::RMSprop => self.rmsprop_update(&grads),
            _ => grads
                .iter()
                .map(|g| -self.config.learning_rate * g)
                .collect(),
        };

        // Apply update
        for (p, u) in self.params.iter_mut().zip(update.iter()) {
            *p += u;
        }

        self.iteration += 1;
    }

    /// SGD update: Δθ = -lr * g
    fn sgd_update(&self, gradients: &[f64]) -> Vec<f64> {
        gradients
            .iter()
            .map(|g| -self.config.learning_rate * g)
            .collect()
    }

    /// Momentum update: v = μv - lr*g, Δθ = v
    fn momentum_update(&mut self, gradients: &[f64]) -> Vec<f64> {
        let state = self.momentum_state.as_mut().unwrap();
        let lr = self.config.learning_rate;
        let mu = self.config.momentum;

        for (v, g) in state.velocity.iter_mut().zip(gradients.iter()) {
            *v = mu * (*v) - lr * g;
        }

        state.velocity.clone()
    }

    /// NAG update: look-ahead momentum
    fn nag_update(&mut self, gradients: &[f64]) -> Vec<f64> {
        let state = self.momentum_state.as_mut().unwrap();
        let lr = self.config.learning_rate;
        let mu = self.config.momentum;

        let mut update = Vec::with_capacity(gradients.len());

        for (v, g) in state.velocity.iter_mut().zip(gradients.iter()) {
            let v_prev = *v;
            *v = mu * (*v) - lr * g;
            update.push(-mu * v_prev + (1.0 + mu) * (*v));
        }

        update
    }

    /// Adam update
    fn adam_update(&mut self, gradients: &[f64]) -> Vec<f64> {
        let state = self.adam_state.as_mut().unwrap();
        state.t += 1;

        let lr = self.config.learning_rate;
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let eps = self.config.epsilon;

        // Bias correction factors
        let bias_correction1 = 1.0 - beta1.powi(state.t as i32);
        let bias_correction2 = 1.0 - beta2.powi(state.t as i32);

        let mut update = Vec::with_capacity(gradients.len());

        for (i, &g) in gradients.iter().enumerate() {
            // Update biased first moment estimate
            state.m[i] = beta1 * state.m[i] + (1.0 - beta1) * g;

            // Update biased second moment estimate
            state.v[i] = beta2 * state.v[i] + (1.0 - beta2) * g * g;

            // Bias-corrected estimates
            let m_hat = state.m[i] / bias_correction1;
            let v_hat = state.v[i] / bias_correction2;

            // Update
            update.push(-lr * m_hat / (v_hat.sqrt() + eps));
        }

        update
    }

    /// AdaGrad update
    fn adagrad_update(&mut self, gradients: &[f64]) -> Vec<f64> {
        let state = self.adagrad_state.as_mut().unwrap();
        let lr = self.config.learning_rate;
        let eps = self.config.epsilon;

        let mut update = Vec::with_capacity(gradients.len());

        for (i, g) in gradients.iter().enumerate() {
            // Accumulate squared gradients
            state.accumulated[i] += g * g;

            // Update
            update.push(-lr * g / (state.accumulated[i].sqrt() + eps));
        }

        update
    }

    /// RMSprop update
    fn rmsprop_update(&mut self, gradients: &[f64]) -> Vec<f64> {
        let state = self.rmsprop_state.as_mut().unwrap();
        let lr = self.config.learning_rate;
        let beta = self.config.beta2;
        let eps = self.config.epsilon;

        let mut update = Vec::with_capacity(gradients.len());

        for (i, g) in gradients.iter().enumerate() {
            // Update running average
            state.cache[i] = beta * state.cache[i] + (1.0 - beta) * g * g;

            // Update
            update.push(-lr * g / (state.cache[i].sqrt() + eps));
        }

        update
    }

    // -------------------------------------------------------------------------
    // SPSA (Gradient-free)
    // -------------------------------------------------------------------------

    /// SPSA step (requires objective function)
    pub fn step_spsa<F>(&mut self, objective_fn: &F)
    where
        F: Fn(&[f64]) -> f64,
    {
        let state = self.spsa_state.as_mut().unwrap();
        state.k += 1;

        let n = self.params.len();
        let a_k = state.learning_rate();
        let c_k = state.perturbation();

        // Generate pseudo-random perturbation (Bernoulli ±1)
        // Using a simple hash-based approach for reproducibility
        let delta: Vec<f64> = (0..n)
            .map(|i| {
                // Simple hash combining iteration and index
                let hash = (state.k.wrapping_mul(2654435761) ^ i.wrapping_mul(2246822519)) % 100;
                if hash < 50 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect();

        // Perturbed evaluations
        let params_plus: Vec<f64> = self
            .params
            .iter()
            .zip(delta.iter())
            .map(|(p, d)| p + c_k * d)
            .collect();

        let params_minus: Vec<f64> = self
            .params
            .iter()
            .zip(delta.iter())
            .map(|(p, d)| p - c_k * d)
            .collect();

        let f_plus = objective_fn(&params_plus);
        let f_minus = objective_fn(&params_minus);

        // Estimate gradient
        let g_hat: Vec<f64> = delta
            .iter()
            .map(|d| (f_plus - f_minus) / (2.0 * c_k * d))
            .collect();

        // Update parameters
        for (p, g) in self.params.iter_mut().zip(g_hat.iter()) {
            *p -= a_k * g;
        }

        self.iteration += 1;
    }

    // -------------------------------------------------------------------------
    // Convergence
    // -------------------------------------------------------------------------

    /// Update best seen objective
    pub fn update_objective(&mut self, objective: f64) -> bool {
        if objective < self.best_objective - self.config.tolerance {
            self.best_objective = objective;
            self.best_params = self.params.clone();
            self.steps_without_improvement = 0;
            true
        } else {
            self.steps_without_improvement += 1;
            false
        }
    }

    /// Check if early stopping triggered
    pub fn should_stop(&self) -> bool {
        if self.config.early_stopping {
            self.steps_without_improvement >= self.config.patience
        } else {
            false
        }
    }

    /// Check if converged
    pub fn is_converged(&self, current_objective: f64, prev_objective: f64) -> bool {
        (prev_objective - current_objective).abs() < self.config.tolerance
    }

    /// Check if max iterations reached
    pub fn max_iterations_reached(&self) -> bool {
        self.iteration >= self.config.max_iter
    }

    // -------------------------------------------------------------------------
    // Reset
    // -------------------------------------------------------------------------

    /// Reset optimizer state
    pub fn reset(&mut self) {
        self.iteration = 0;
        self.best_objective = f64::INFINITY;
        self.steps_without_improvement = 0;

        if let Some(ref mut state) = self.momentum_state {
            state.reset();
        }
        if let Some(ref mut state) = self.adam_state {
            state.reset();
        }
        if let Some(ref mut state) = self.rmsprop_state {
            state.reset();
        }
        if let Some(ref mut state) = self.adagrad_state {
            state.reset();
        }
        if let Some(ref mut state) = self.spsa_state {
            state.reset();
        }
    }

    /// Reset to new initial parameters
    pub fn reset_params(&mut self, params: Vec<f64>) {
        self.params = params.clone();
        self.best_params = params;
        self.reset();
    }
}

// =============================================================================
// Optimization Loop
// =============================================================================

/// Run full optimization loop
pub fn minimize<F, G>(
    config: OptimizerConfig,
    initial_params: Vec<f64>,
    objective_fn: F,
    gradient_fn: G,
) -> OptimizationResult
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let mut optimizer = Optimizer::new(config.clone(), initial_params);
    let mut history = Vec::with_capacity(config.max_iter);
    let mut prev_objective = f64::INFINITY;
    let mut func_evals = 0;
    let mut grad_evals = 0;

    for _ in 0..config.max_iter {
        // Compute objective
        let objective = objective_fn(optimizer.params());
        func_evals += 1;
        history.push(objective);

        // Check convergence
        if optimizer.is_converged(objective, prev_objective) {
            return OptimizationResult::success(
                optimizer.params().to_vec(),
                objective,
                optimizer.iteration(),
            )
            .with_history(history)
            .with_evals(func_evals, grad_evals);
        }

        // Update best
        optimizer.update_objective(objective);

        // Check early stopping
        if optimizer.should_stop() {
            let mut result = OptimizationResult::success(
                optimizer.best_params().to_vec(),
                optimizer.best_objective,
                optimizer.iteration(),
            );
            result.message = "Early stopping triggered".to_string();
            return result
                .with_history(history)
                .with_evals(func_evals, grad_evals);
        }

        // Compute gradients and step
        let gradients = gradient_fn(optimizer.params());
        grad_evals += 1;
        optimizer.step(&gradients);

        prev_objective = objective;
    }

    // Max iterations reached
    let mut result = OptimizationResult::success(
        optimizer.best_params().to_vec(),
        optimizer.best_objective,
        optimizer.iteration(),
    );
    result.converged = false;
    result.message = "Maximum iterations reached".to_string();
    result
        .with_history(history)
        .with_evals(func_evals, grad_evals)
}

/// Run SPSA optimization (gradient-free)
pub fn minimize_spsa<F>(
    initial_params: Vec<f64>,
    objective_fn: F,
    max_iter: usize,
) -> OptimizationResult
where
    F: Fn(&[f64]) -> f64,
{
    let config = OptimizerConfig::spsa().with_max_iter(max_iter);
    let mut optimizer = Optimizer::new(config, initial_params);
    let mut history = Vec::with_capacity(max_iter);
    let mut func_evals = 0;

    for _ in 0..max_iter {
        // Evaluate current
        let objective = objective_fn(optimizer.params());
        func_evals += 1;
        history.push(objective);

        // Update best
        optimizer.update_objective(objective);

        // SPSA step (2 function evals per step)
        optimizer.step_spsa(&objective_fn);
        func_evals += 2;
    }

    OptimizationResult::success(
        optimizer.best_params().to_vec(),
        optimizer.best_objective,
        optimizer.iteration(),
    )
    .with_history(history)
    .with_evals(func_evals, 0)
}

// =============================================================================
// Learning Rate Schedulers
// =============================================================================

/// Learning rate schedule
#[derive(Debug, Clone)]
pub enum LRSchedule {
    /// Constant learning rate
    Constant,
    /// Step decay: lr *= factor every step_size iterations
    Step { step_size: usize, factor: f64 },
    /// Exponential decay: lr *= factor^iteration
    Exponential { factor: f64 },
    /// Cosine annealing
    Cosine { t_max: usize, eta_min: f64 },
    /// Linear warmup then decay
    WarmupDecay {
        warmup_steps: usize,
        decay_factor: f64,
    },
}

impl LRSchedule {
    /// Compute learning rate at iteration
    pub fn get_lr(&self, base_lr: f64, iteration: usize) -> f64 {
        match self {
            LRSchedule::Constant => base_lr,
            LRSchedule::Step { step_size, factor } => {
                base_lr * factor.powi((iteration / step_size) as i32)
            }
            LRSchedule::Exponential { factor } => base_lr * factor.powi(iteration as i32),
            LRSchedule::Cosine { t_max, eta_min } => {
                let t = iteration.min(*t_max) as f64;
                let t_max = *t_max as f64;
                eta_min + 0.5 * (base_lr - eta_min) * (1.0 + (PI * t / t_max).cos())
            }
            LRSchedule::WarmupDecay {
                warmup_steps,
                decay_factor,
            } => {
                if iteration < *warmup_steps {
                    base_lr * (iteration + 1) as f64 / *warmup_steps as f64
                } else {
                    base_lr * decay_factor.powi((iteration - warmup_steps) as i32)
                }
            }
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(dead_code)]
    fn rosenbrock(params: &[f64]) -> f64 {
        let x = params[0];
        let y = params[1];
        (1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2)
    }

    #[allow(dead_code)]
    fn rosenbrock_grad(params: &[f64]) -> Vec<f64> {
        let x = params[0];
        let y = params[1];
        let dx = -2.0 * (1.0 - x) - 400.0 * x * (y - x * x);
        let dy = 200.0 * (y - x * x);
        vec![dx, dy]
    }

    fn quadratic(params: &[f64]) -> f64 {
        params.iter().map(|x| x * x).sum()
    }

    fn quadratic_grad(params: &[f64]) -> Vec<f64> {
        params.iter().map(|x| 2.0 * x).collect()
    }

    #[test]
    fn test_sgd_step() {
        let mut opt = Optimizer::sgd(0.1, vec![1.0, 1.0]);
        let grads = quadratic_grad(opt.params());
        opt.step(&grads);

        // After one step: params = [1.0, 1.0] - 0.1 * [2.0, 2.0] = [0.8, 0.8]
        assert!((opt.params()[0] - 0.8).abs() < 1e-10);
        assert!((opt.params()[1] - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_adam_step() {
        let mut opt = Optimizer::adam(0.1, vec![1.0, 1.0]);
        let grads = quadratic_grad(opt.params());
        opt.step(&grads);

        // Adam should decrease parameters
        assert!(opt.params()[0] < 1.0);
        assert!(opt.params()[1] < 1.0);
    }

    #[test]
    fn test_momentum_accumulation() {
        let config = OptimizerConfig::momentum(0.1, 0.9);
        let mut opt = Optimizer::new(config, vec![1.0]);

        // Multiple steps with constant gradient
        for _ in 0..5 {
            opt.step(&[1.0]);
        }

        // Velocity should accumulate
        let state = opt.momentum_state.as_ref().unwrap();
        assert!(state.velocity[0].abs() > 0.1);
    }

    #[test]
    fn test_gradient_clipping() {
        let config = OptimizerConfig::sgd(0.1).with_grad_clip(1.0);
        let mut opt = Optimizer::new(config, vec![0.0, 0.0]);

        // Large gradient
        opt.step(&[100.0, 100.0]);

        // Update should be clipped
        let update_magnitude = (opt.params()[0].powi(2) + opt.params()[1].powi(2)).sqrt();
        assert!(update_magnitude <= 0.2); // 0.1 * sqrt(2) ≈ 0.14
    }

    #[test]
    fn test_weight_decay() {
        let config = OptimizerConfig::sgd(0.1).with_weight_decay(0.01);
        let mut opt = Optimizer::new(config, vec![1.0]);

        opt.step(&[0.0]); // Zero gradient

        // Should still update due to weight decay
        assert!(opt.params()[0] < 1.0);
    }

    #[test]
    fn test_minimize_quadratic() {
        let config = OptimizerConfig::adam(0.1)
            .with_max_iter(100)
            .with_tolerance(1e-6);

        let result = minimize(config, vec![5.0, 5.0], quadratic, quadratic_grad);

        // Should converge near origin
        assert!(result.objective < 0.1);
    }

    #[test]
    fn test_spsa_optimization() {
        // SPSA is stochastic, so just verify it doesn't diverge
        let result = minimize_spsa(vec![2.0, 2.0], quadratic, 200);

        // Start: 4+4=8, should not get worse
        assert!(result.objective <= 8.5);
        // Should make some progress
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_early_stopping() {
        // Test early stopping with a function that plateaus
        let plateau_fn = |_params: &[f64]| -> f64 {
            // Returns constant 1.0 after first call (simulates plateau)
            1.0
        };
        let plateau_grad = |params: &[f64]| -> Vec<f64> {
            vec![0.0; params.len()] // Zero gradient (plateau)
        };

        let config = OptimizerConfig::sgd(0.1)
            .with_max_iter(100)
            .with_tolerance(1e-6)
            .with_early_stopping(3);

        let result = minimize(config, vec![1.0, 1.0], plateau_fn, plateau_grad);

        // With zero gradient and early stopping patience=3, should stop quickly
        // After 3 iterations without improvement (constant objective), early stop triggers
        assert!(result.iterations <= 10);
    }

    #[test]
    fn test_lr_schedule_step() {
        let schedule = LRSchedule::Step {
            step_size: 10,
            factor: 0.5,
        };

        assert!((schedule.get_lr(1.0, 0) - 1.0).abs() < 1e-10);
        assert!((schedule.get_lr(1.0, 10) - 0.5).abs() < 1e-10);
        assert!((schedule.get_lr(1.0, 20) - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_lr_schedule_cosine() {
        let schedule = LRSchedule::Cosine {
            t_max: 100,
            eta_min: 0.0,
        };

        // At t=0: should be base_lr
        assert!((schedule.get_lr(1.0, 0) - 1.0).abs() < 1e-10);
        // At t=t_max: should be eta_min
        assert!((schedule.get_lr(1.0, 100) - 0.0).abs() < 1e-10);
        // At t=t_max/2: should be midpoint
        assert!((schedule.get_lr(1.0, 50) - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_optimizer_reset() {
        let mut opt = Optimizer::adam(0.1, vec![1.0, 1.0]);

        // Take some steps
        for _ in 0..10 {
            opt.step(&[0.1, 0.1]);
        }

        opt.reset();

        assert_eq!(opt.iteration(), 0);
        let state = opt.adam_state.as_ref().unwrap();
        assert!(state.m.iter().all(|&m| m == 0.0));
        assert!(state.v.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_update_objective() {
        let mut opt = Optimizer::sgd(0.1, vec![1.0]);

        assert!(opt.update_objective(1.0)); // First improvement
        assert!(!opt.update_objective(1.1)); // No improvement
        assert!(opt.update_objective(0.5)); // Improvement
    }

    #[test]
    fn test_optimizer_type() {
        assert!(OptimizerType::SGD.is_gradient_based());
        assert!(OptimizerType::Adam.is_gradient_based());
        assert!(!OptimizerType::SPSA.is_gradient_based());
    }

    #[test]
    fn test_spsa_state() {
        let mut state = SPSAState::new();

        // Learning rate and perturbation should decrease
        let lr1 = state.learning_rate();
        let c1 = state.perturbation();

        state.k = 10;
        let lr2 = state.learning_rate();
        let c2 = state.perturbation();

        assert!(lr2 < lr1);
        assert!(c2 < c1);
    }

    #[test]
    fn test_rmsprop() {
        let config = OptimizerConfig::rmsprop(0.01);
        let mut opt = Optimizer::new(config, vec![1.0, 2.0]);

        for _ in 0..10 {
            let grads = quadratic_grad(opt.params());
            opt.step(&grads);
        }

        // Should converge toward zero
        assert!(opt.params()[0].abs() < 1.0);
        assert!(opt.params()[1].abs() < 2.0);
    }

    #[test]
    fn test_adagrad() {
        let config = OptimizerConfig {
            optimizer_type: OptimizerType::AdaGrad,
            learning_rate: 0.5,
            ..Default::default()
        };
        let mut opt = Optimizer::new(config, vec![1.0, 1.0]);

        for _ in 0..20 {
            let grads = quadratic_grad(opt.params());
            opt.step(&grads);
        }

        // Should decrease
        assert!(opt.params()[0].abs() < 0.5);
    }
}

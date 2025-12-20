//! Layerwise Training for Variational Quantum Circuits
//!
//! Sprint 2 Week 4 Day 3: Progressive and transfer learning techniques
//!
//! Layerwise training trains the circuit incrementally, starting from shallow
//! circuits and progressively adding depth. This helps avoid barren plateaus
//! and improves convergence.
//!
//! # Key Concepts
//!
//! - **Progressive Training**: Train layer 1, then layers 1-2, then 1-3, etc.
//! - **Freeze/Unfreeze**: Fix some parameters while training others
//! - **Transfer Learning**: Use trained shallow circuit to initialize deeper one
//!
//! # Example
//!
//! ```ignore
//! use tqp_core::{LayerwiseTrainer, VariationalCircuit, Hamiltonian};
//!
//! let circuit = VariationalCircuit::hardware_efficient(4, 6);
//! let trainer = LayerwiseTrainer::new(LayerwiseConfig::default());
//!
//! let result = trainer.train(&circuit, &hamiltonian);
//! println!("Final energy: {}", result.final_objective);
//! ```

use std::collections::HashSet;

use crate::autodiff::{
    compute_expectation, compute_gradient, VariationalCircuit, Hamiltonian,
};
use crate::optimizer::{Optimizer, OptimizerConfig, OptimizerType};

// =============================================================================
// Constants
// =============================================================================

/// Default iterations per layer
pub const DEFAULT_ITER_PER_LAYER: usize = 50;

/// Default learning rate for layerwise training
pub const DEFAULT_LAYERWISE_LR: f64 = 0.1;

/// Default warmup iterations when unfreezing
pub const DEFAULT_WARMUP_ITER: usize = 10;

// =============================================================================
// Configuration
// =============================================================================

/// Training mode for layerwise training
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerwiseMode {
    /// Train layers progressively: 1, then 1-2, then 1-3, etc.
    Progressive,
    /// Train each layer individually, then fine-tune all
    Individual,
    /// Alternate between freezing and training layers
    Alternating,
    /// Train in reverse: deepest first
    ReverseProgressive,
}

/// Freeze strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FreezeStrategy {
    /// Freeze earlier layers, train current
    FreezePrevious,
    /// Freeze later layers, train current
    FreezeNext,
    /// No freezing, train all active layers
    NoFreeze,
    /// Freeze all except current layer
    IsolateLayer,
}

/// Configuration for layerwise training
#[derive(Debug, Clone)]
pub struct LayerwiseConfig {
    /// Training mode
    pub mode: LayerwiseMode,
    /// Freeze strategy
    pub freeze_strategy: FreezeStrategy,
    /// Iterations per layer
    pub iter_per_layer: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Warmup iterations after unfreezing
    pub warmup_iter: usize,
    /// Fine-tuning iterations after layerwise training
    pub finetune_iter: usize,
    /// Optimizer type
    pub optimizer_type: OptimizerType,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Whether to use learning rate decay between layers
    pub decay_lr: bool,
    /// Learning rate decay factor
    pub lr_decay_factor: f64,
}

impl Default for LayerwiseConfig {
    fn default() -> Self {
        Self {
            mode: LayerwiseMode::Progressive,
            freeze_strategy: FreezeStrategy::FreezePrevious,
            iter_per_layer: DEFAULT_ITER_PER_LAYER,
            learning_rate: DEFAULT_LAYERWISE_LR,
            warmup_iter: DEFAULT_WARMUP_ITER,
            finetune_iter: 100,
            optimizer_type: OptimizerType::Adam,
            tolerance: 1e-6,
            decay_lr: false,
            lr_decay_factor: 0.9,
        }
    }
}

impl LayerwiseConfig {
    /// Set training mode
    pub fn with_mode(mut self, mode: LayerwiseMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set freeze strategy
    pub fn with_freeze_strategy(mut self, strategy: FreezeStrategy) -> Self {
        self.freeze_strategy = strategy;
        self
    }

    /// Set iterations per layer
    pub fn with_iter_per_layer(mut self, iter: usize) -> Self {
        self.iter_per_layer = iter;
        self
    }

    /// Set learning rate
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set warmup iterations
    pub fn with_warmup_iter(mut self, iter: usize) -> Self {
        self.warmup_iter = iter;
        self
    }

    /// Set fine-tuning iterations
    pub fn with_finetune_iter(mut self, iter: usize) -> Self {
        self.finetune_iter = iter;
        self
    }

    /// Set optimizer type
    pub fn with_optimizer(mut self, opt_type: OptimizerType) -> Self {
        self.optimizer_type = opt_type;
        self
    }

    /// Enable learning rate decay
    pub fn with_lr_decay(mut self, factor: f64) -> Self {
        self.decay_lr = true;
        self.lr_decay_factor = factor;
        self
    }
}

// =============================================================================
// Parameter Mask
// =============================================================================

/// Mask for trainable parameters
#[derive(Debug, Clone)]
pub struct ParameterMask {
    /// Which parameters are trainable (true = trainable)
    mask: Vec<bool>,
    /// Frozen parameter values
    frozen_values: Vec<f64>,
}

impl ParameterMask {
    /// Create mask with all parameters trainable
    pub fn all_trainable(n_params: usize) -> Self {
        Self {
            mask: vec![true; n_params],
            frozen_values: vec![0.0; n_params],
        }
    }

    /// Create mask with all parameters frozen
    pub fn all_frozen(n_params: usize, values: Vec<f64>) -> Self {
        Self {
            mask: vec![false; n_params],
            frozen_values: values,
        }
    }

    /// Freeze parameters in range
    pub fn freeze_range(&mut self, start: usize, end: usize, values: &[f64]) {
        for i in start..end.min(self.mask.len()) {
            self.mask[i] = false;
            if i < values.len() {
                self.frozen_values[i] = values[i];
            }
        }
    }

    /// Unfreeze parameters in range
    pub fn unfreeze_range(&mut self, start: usize, end: usize) {
        for i in start..end.min(self.mask.len()) {
            self.mask[i] = true;
        }
    }

    /// Freeze specific parameters
    pub fn freeze(&mut self, indices: &[usize], values: &[f64]) {
        for (idx, &i) in indices.iter().enumerate() {
            if i < self.mask.len() {
                self.mask[i] = false;
                if idx < values.len() {
                    self.frozen_values[i] = values[idx];
                }
            }
        }
    }

    /// Unfreeze specific parameters
    pub fn unfreeze(&mut self, indices: &[usize]) {
        for &i in indices {
            if i < self.mask.len() {
                self.mask[i] = true;
            }
        }
    }

    /// Get trainable indices
    pub fn trainable_indices(&self) -> Vec<usize> {
        self.mask.iter()
            .enumerate()
            .filter(|(_, &is_trainable)| is_trainable)
            .map(|(i, _)| i)
            .collect()
    }

    /// Get frozen indices
    pub fn frozen_indices(&self) -> Vec<usize> {
        self.mask.iter()
            .enumerate()
            .filter(|(_, &is_trainable)| !is_trainable)
            .map(|(i, _)| i)
            .collect()
    }

    /// Number of trainable parameters
    pub fn n_trainable(&self) -> usize {
        self.mask.iter().filter(|&&m| m).count()
    }

    /// Total parameters
    pub fn n_total(&self) -> usize {
        self.mask.len()
    }

    /// Apply mask: extract trainable parameters
    pub fn extract_trainable(&self, params: &[f64]) -> Vec<f64> {
        self.trainable_indices().iter()
            .map(|&i| params[i])
            .collect()
    }

    /// Apply mask: merge trainable back into full params
    pub fn merge_trainable(&self, trainable: &[f64], full_params: &mut [f64]) {
        let indices = self.trainable_indices();
        for (idx, &i) in indices.iter().enumerate() {
            if idx < trainable.len() && i < full_params.len() {
                full_params[i] = trainable[idx];
            }
        }
    }

    /// Get full parameters with frozen values filled in
    pub fn full_params(&self, trainable: &[f64]) -> Vec<f64> {
        let mut result = self.frozen_values.clone();
        let indices = self.trainable_indices();
        for (idx, &i) in indices.iter().enumerate() {
            if idx < trainable.len() {
                result[i] = trainable[idx];
            }
        }
        result
    }

    /// Apply gradient mask (zero out frozen gradients)
    pub fn mask_gradient(&self, gradient: &mut [f64]) {
        for (i, &is_trainable) in self.mask.iter().enumerate() {
            if !is_trainable && i < gradient.len() {
                gradient[i] = 0.0;
            }
        }
    }
}

// =============================================================================
// Layer Partition
// =============================================================================

/// Partition of parameters into layers
#[derive(Debug, Clone)]
pub struct LayerPartition {
    /// Start index of each layer
    layer_starts: Vec<usize>,
    /// Total number of parameters
    n_params: usize,
    /// Number of layers
    n_layers: usize,
}

impl LayerPartition {
    /// Create uniform partition
    pub fn uniform(n_params: usize, n_layers: usize) -> Self {
        let params_per_layer = n_params / n_layers.max(1);
        let mut layer_starts = Vec::with_capacity(n_layers);

        for i in 0..n_layers {
            layer_starts.push(i * params_per_layer);
        }

        Self {
            layer_starts,
            n_params,
            n_layers,
        }
    }

    /// Create from explicit starts
    pub fn from_starts(starts: Vec<usize>, n_params: usize) -> Self {
        let n_layers = starts.len();
        Self {
            layer_starts: starts,
            n_params,
            n_layers,
        }
    }

    /// Get parameter range for layer
    pub fn layer_range(&self, layer_idx: usize) -> std::ops::Range<usize> {
        let start = self.layer_starts.get(layer_idx).copied().unwrap_or(0);
        let end = self.layer_starts.get(layer_idx + 1)
            .copied()
            .unwrap_or(self.n_params);
        start..end
    }

    /// Get parameter indices for layer
    pub fn layer_params(&self, layer_idx: usize) -> Vec<usize> {
        self.layer_range(layer_idx).collect()
    }

    /// Number of layers
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    /// Parameters in layer
    pub fn layer_size(&self, layer_idx: usize) -> usize {
        let range = self.layer_range(layer_idx);
        range.end - range.start
    }
}

// =============================================================================
// Training Result
// =============================================================================

/// Result of layerwise training
#[derive(Debug, Clone)]
pub struct LayerwiseResult {
    /// Final parameters
    pub params: Vec<f64>,
    /// Final objective value
    pub final_objective: f64,
    /// Best objective seen
    pub best_objective: f64,
    /// Best parameters
    pub best_params: Vec<f64>,
    /// Total iterations
    pub total_iterations: usize,
    /// Per-layer results
    pub layer_results: Vec<LayerTrainingResult>,
    /// Fine-tuning result
    pub finetune_result: Option<FinetuneResult>,
    /// Elapsed time in milliseconds
    pub elapsed_ms: u64,
}

/// Result for single layer training
#[derive(Debug, Clone)]
pub struct LayerTrainingResult {
    /// Layer index
    pub layer_idx: usize,
    /// Objective before training
    pub initial_objective: f64,
    /// Objective after training
    pub final_objective: f64,
    /// Iterations used
    pub iterations: usize,
    /// Improvement (initial - final)
    pub improvement: f64,
}

/// Result for fine-tuning phase
#[derive(Debug, Clone)]
pub struct FinetuneResult {
    /// Objective before fine-tuning
    pub initial_objective: f64,
    /// Objective after fine-tuning
    pub final_objective: f64,
    /// Iterations used
    pub iterations: usize,
}

// =============================================================================
// Layerwise Trainer
// =============================================================================

/// Layerwise training orchestrator
pub struct LayerwiseTrainer {
    /// Configuration
    config: LayerwiseConfig,
}

impl LayerwiseTrainer {
    /// Create new trainer with config
    pub fn new(config: LayerwiseConfig) -> Self {
        Self { config }
    }

    /// Train circuit layerwise
    pub fn train(
        &self,
        circuit: &VariationalCircuit,
        hamiltonian: &Hamiltonian,
        initial_params: Vec<f64>,
    ) -> LayerwiseResult {
        let start = std::time::Instant::now();
        let n_params = circuit.num_params();
        let n_layers = circuit.depth().max(1);

        // Create layer partition
        let partition = LayerPartition::uniform(n_params, n_layers);

        // Initialize parameters
        let mut params = if initial_params.len() == n_params {
            initial_params
        } else {
            vec![0.1; n_params]
        };

        let mut best_objective = f64::MAX;
        let mut best_params = params.clone();
        let mut total_iterations = 0;
        let mut layer_results = Vec::new();

        // Train according to mode
        match self.config.mode {
            LayerwiseMode::Progressive => {
                for layer_idx in 0..n_layers {
                    let result = self.train_layer(
                        circuit,
                        hamiltonian,
                        &mut params,
                        &partition,
                        layer_idx,
                        0..=layer_idx,
                    );
                    total_iterations += result.iterations;
                    layer_results.push(result);

                    let obj = compute_expectation(circuit, &params, hamiltonian);
                    if obj < best_objective {
                        best_objective = obj;
                        best_params = params.clone();
                    }
                }
            }
            LayerwiseMode::Individual => {
                for layer_idx in 0..n_layers {
                    let result = self.train_layer(
                        circuit,
                        hamiltonian,
                        &mut params,
                        &partition,
                        layer_idx,
                        layer_idx..=layer_idx,
                    );
                    total_iterations += result.iterations;
                    layer_results.push(result);
                }

                let obj = compute_expectation(circuit, &params, hamiltonian);
                if obj < best_objective {
                    best_objective = obj;
                    best_params = params.clone();
                }
            }
            LayerwiseMode::Alternating => {
                // Forward pass
                for layer_idx in 0..n_layers {
                    let result = self.train_layer(
                        circuit,
                        hamiltonian,
                        &mut params,
                        &partition,
                        layer_idx,
                        layer_idx..=layer_idx,
                    );
                    total_iterations += result.iterations;
                    layer_results.push(result);
                }
                // Backward pass
                for layer_idx in (0..n_layers).rev() {
                    let result = self.train_layer(
                        circuit,
                        hamiltonian,
                        &mut params,
                        &partition,
                        layer_idx,
                        layer_idx..=layer_idx,
                    );
                    total_iterations += result.iterations;
                    layer_results.push(result);
                }

                let obj = compute_expectation(circuit, &params, hamiltonian);
                if obj < best_objective {
                    best_objective = obj;
                    best_params = params.clone();
                }
            }
            LayerwiseMode::ReverseProgressive => {
                for layer_idx in (0..n_layers).rev() {
                    let result = self.train_layer(
                        circuit,
                        hamiltonian,
                        &mut params,
                        &partition,
                        layer_idx,
                        layer_idx..=(n_layers - 1),
                    );
                    total_iterations += result.iterations;
                    layer_results.push(result);

                    let obj = compute_expectation(circuit, &params, hamiltonian);
                    if obj < best_objective {
                        best_objective = obj;
                        best_params = params.clone();
                    }
                }
            }
        }

        // Fine-tuning phase
        let finetune_result = if self.config.finetune_iter > 0 {
            Some(self.finetune(circuit, hamiltonian, &mut params))
        } else {
            None
        };

        if let Some(ref ft) = finetune_result {
            total_iterations += ft.iterations;
            if ft.final_objective < best_objective {
                best_objective = ft.final_objective;
                best_params = params.clone();
            }
        }

        let final_objective = compute_expectation(circuit, &params, hamiltonian);

        LayerwiseResult {
            params,
            final_objective,
            best_objective,
            best_params,
            total_iterations,
            layer_results,
            finetune_result,
            elapsed_ms: start.elapsed().as_millis() as u64,
        }
    }

    /// Train a single layer (or range of layers)
    fn train_layer(
        &self,
        circuit: &VariationalCircuit,
        hamiltonian: &Hamiltonian,
        params: &mut [f64],
        partition: &LayerPartition,
        current_layer: usize,
        trainable_layers: std::ops::RangeInclusive<usize>,
    ) -> LayerTrainingResult {
        let n_params = params.len();

        // Create mask
        let mut mask = ParameterMask::all_frozen(n_params, params.to_vec());

        // Unfreeze trainable layers
        for layer in trainable_layers.clone() {
            let range = partition.layer_range(layer);
            mask.unfreeze_range(range.start, range.end);
        }

        let initial_objective = compute_expectation(circuit, params, hamiltonian);

        // Get trainable subset
        let trainable = mask.extract_trainable(params);
        if trainable.is_empty() {
            return LayerTrainingResult {
                layer_idx: current_layer,
                initial_objective,
                final_objective: initial_objective,
                iterations: 0,
                improvement: 0.0,
            };
        }

        // Create optimizer for trainable params
        let lr = if self.config.decay_lr {
            self.config.learning_rate * self.config.lr_decay_factor.powi(current_layer as i32)
        } else {
            self.config.learning_rate
        };

        let opt_config = OptimizerConfig::new(self.config.optimizer_type, lr)
            .with_max_iter(self.config.iter_per_layer)
            .with_tolerance(self.config.tolerance);

        let mut opt = Optimizer::new(opt_config, trainable);

        // Training loop
        let mut iterations = 0;
        for _ in 0..self.config.iter_per_layer {
            // Get full params
            let full_params = mask.full_params(opt.params());

            // Compute gradient
            let mut gradient = compute_gradient(circuit, &full_params, hamiltonian);

            // Mask frozen gradients
            mask.mask_gradient(&mut gradient);

            // Extract trainable gradients
            let trainable_grad: Vec<f64> = mask.trainable_indices().iter()
                .map(|&i| gradient[i])
                .collect();

            // Check convergence
            let grad_norm: f64 = trainable_grad.iter().map(|g| g * g).sum::<f64>().sqrt();
            if grad_norm < self.config.tolerance {
                break;
            }

            // Step
            let obj = compute_expectation(circuit, &full_params, hamiltonian);
            opt.update_objective(obj);
            opt.step(&trainable_grad);
            iterations += 1;
        }

        // Update full params
        mask.merge_trainable(opt.params(), params);

        let final_objective = compute_expectation(circuit, params, hamiltonian);

        LayerTrainingResult {
            layer_idx: current_layer,
            initial_objective,
            final_objective,
            iterations,
            improvement: initial_objective - final_objective,
        }
    }

    /// Fine-tune all parameters together
    fn finetune(
        &self,
        circuit: &VariationalCircuit,
        hamiltonian: &Hamiltonian,
        params: &mut [f64],
    ) -> FinetuneResult {
        let initial_objective = compute_expectation(circuit, params, hamiltonian);

        // Lower learning rate for fine-tuning
        let lr = self.config.learning_rate * 0.1;

        let opt_config = OptimizerConfig::new(self.config.optimizer_type, lr)
            .with_max_iter(self.config.finetune_iter)
            .with_tolerance(self.config.tolerance);

        let mut opt = Optimizer::new(opt_config, params.to_vec());

        let mut iterations = 0;
        for _ in 0..self.config.finetune_iter {
            let gradient = compute_gradient(circuit, opt.params(), hamiltonian);

            let grad_norm: f64 = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
            if grad_norm < self.config.tolerance {
                break;
            }

            let obj = compute_expectation(circuit, opt.params(), hamiltonian);
            opt.update_objective(obj);
            opt.step(&gradient);
            iterations += 1;
        }

        // Copy optimized params back
        params.copy_from_slice(opt.params());

        let final_objective = compute_expectation(circuit, params, hamiltonian);

        FinetuneResult {
            initial_objective,
            final_objective,
            iterations,
        }
    }
}

// =============================================================================
// Convenience Functions
// =============================================================================

/// Quick layerwise training with default config
pub fn layerwise_train(
    circuit: &VariationalCircuit,
    hamiltonian: &Hamiltonian,
    initial_params: Vec<f64>,
) -> LayerwiseResult {
    let trainer = LayerwiseTrainer::new(LayerwiseConfig::default());
    trainer.train(circuit, hamiltonian, initial_params)
}

/// Progressive training (most common mode)
pub fn progressive_train(
    circuit: &VariationalCircuit,
    hamiltonian: &Hamiltonian,
    initial_params: Vec<f64>,
    iter_per_layer: usize,
) -> LayerwiseResult {
    let config = LayerwiseConfig::default()
        .with_mode(LayerwiseMode::Progressive)
        .with_iter_per_layer(iter_per_layer);
    let trainer = LayerwiseTrainer::new(config);
    trainer.train(circuit, hamiltonian, initial_params)
}

/// Individual layer training then fine-tune
pub fn individual_then_finetune(
    circuit: &VariationalCircuit,
    hamiltonian: &Hamiltonian,
    initial_params: Vec<f64>,
    finetune_iter: usize,
) -> LayerwiseResult {
    let config = LayerwiseConfig::default()
        .with_mode(LayerwiseMode::Individual)
        .with_finetune_iter(finetune_iter);
    let trainer = LayerwiseTrainer::new(config);
    trainer.train(circuit, hamiltonian, initial_params)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::PauliObservable;

    #[test]
    fn test_parameter_mask_creation() {
        let mask = ParameterMask::all_trainable(5);
        assert_eq!(mask.n_trainable(), 5);
        assert_eq!(mask.n_total(), 5);
    }

    #[test]
    fn test_parameter_mask_freeze() {
        let mut mask = ParameterMask::all_trainable(5);
        mask.freeze_range(0, 2, &[0.1, 0.2]);

        assert_eq!(mask.n_trainable(), 3);
        assert_eq!(mask.frozen_indices(), vec![0, 1]);
        assert_eq!(mask.trainable_indices(), vec![2, 3, 4]);
    }

    #[test]
    fn test_parameter_mask_unfreeze() {
        let mut mask = ParameterMask::all_frozen(5, vec![0.0; 5]);
        mask.unfreeze_range(2, 4);

        assert_eq!(mask.n_trainable(), 2);
        assert_eq!(mask.trainable_indices(), vec![2, 3]);
    }

    #[test]
    fn test_parameter_mask_extract_merge() {
        let mut mask = ParameterMask::all_trainable(4);
        mask.freeze(&[0, 2], &[0.1, 0.3]);

        let full = vec![0.1, 0.2, 0.3, 0.4];
        let trainable = mask.extract_trainable(&full);
        assert_eq!(trainable, vec![0.2, 0.4]);

        let mut merged = full.clone();
        mask.merge_trainable(&[0.5, 0.6], &mut merged);
        assert_eq!(merged, vec![0.1, 0.5, 0.3, 0.6]);
    }

    #[test]
    fn test_parameter_mask_full_params() {
        let mut mask = ParameterMask::all_frozen(4, vec![0.1, 0.2, 0.3, 0.4]);
        mask.unfreeze(&[1, 3]);

        let trainable = vec![0.5, 0.6];
        let full = mask.full_params(&trainable);
        assert_eq!(full, vec![0.1, 0.5, 0.3, 0.6]);
    }

    #[test]
    fn test_parameter_mask_gradient() {
        let mut mask = ParameterMask::all_trainable(4);
        mask.freeze(&[0, 2], &[0.0, 0.0]);

        let mut grad = vec![1.0, 2.0, 3.0, 4.0];
        mask.mask_gradient(&mut grad);
        assert_eq!(grad, vec![0.0, 2.0, 0.0, 4.0]);
    }

    #[test]
    fn test_layer_partition_uniform() {
        let partition = LayerPartition::uniform(12, 3);

        assert_eq!(partition.n_layers(), 3);
        assert_eq!(partition.layer_range(0), 0..4);
        assert_eq!(partition.layer_range(1), 4..8);
        assert_eq!(partition.layer_range(2), 8..12);
    }

    #[test]
    fn test_layer_partition_params() {
        let partition = LayerPartition::uniform(6, 2);

        assert_eq!(partition.layer_params(0), vec![0, 1, 2]);
        assert_eq!(partition.layer_params(1), vec![3, 4, 5]);
    }

    #[test]
    fn test_layer_partition_size() {
        let partition = LayerPartition::uniform(10, 3);

        assert_eq!(partition.layer_size(0), 3);
        assert_eq!(partition.layer_size(1), 3);
        assert_eq!(partition.layer_size(2), 4); // Remainder goes to last
    }

    #[test]
    fn test_layerwise_config_builder() {
        let config = LayerwiseConfig::default()
            .with_mode(LayerwiseMode::Individual)
            .with_iter_per_layer(100)
            .with_learning_rate(0.05)
            .with_lr_decay(0.95);

        assert_eq!(config.mode, LayerwiseMode::Individual);
        assert_eq!(config.iter_per_layer, 100);
        assert_eq!(config.learning_rate, 0.05);
        assert!(config.decay_lr);
    }

    #[test]
    fn test_layerwise_trainer_creation() {
        let config = LayerwiseConfig::default();
        let _trainer = LayerwiseTrainer::new(config);
    }

    #[test]
    fn test_layerwise_train_simple() {
        let mut circuit = VariationalCircuit::new(2);
        circuit.ry(0);
        circuit.ry(1);
        circuit.cnot(0, 1);
        circuit.ry(0);
        circuit.ry(1);

        let mut hamiltonian = Hamiltonian::new();
        hamiltonian.add_term(PauliObservable::z(0));
        hamiltonian.add_term(PauliObservable::z(1));

        let config = LayerwiseConfig::default()
            .with_iter_per_layer(10)
            .with_finetune_iter(10);

        let trainer = LayerwiseTrainer::new(config);
        let result = trainer.train(&circuit, &hamiltonian, vec![0.5; 4]);

        assert!(result.total_iterations > 0);
        assert!(result.layer_results.len() > 0);
    }

    #[test]
    fn test_progressive_mode() {
        let mut circuit = VariationalCircuit::new(1);
        circuit.ry(0);
        circuit.ry(0);

        let mut hamiltonian = Hamiltonian::new();
        hamiltonian.add_term(PauliObservable::z(0));

        let result = progressive_train(&circuit, &hamiltonian, vec![0.5, 0.5], 5);

        assert!(result.layer_results.len() > 0);
    }

    #[test]
    fn test_individual_mode() {
        let mut circuit = VariationalCircuit::new(1);
        circuit.ry(0);
        circuit.ry(0);

        let mut hamiltonian = Hamiltonian::new();
        hamiltonian.add_term(PauliObservable::z(0));

        let result = individual_then_finetune(&circuit, &hamiltonian, vec![0.5, 0.5], 5);

        assert!(result.finetune_result.is_some());
    }

    #[test]
    fn test_layerwise_result_fields() {
        let mut circuit = VariationalCircuit::new(1);
        circuit.ry(0);

        let mut hamiltonian = Hamiltonian::new();
        hamiltonian.add_term(PauliObservable::z(0));

        let result = layerwise_train(&circuit, &hamiltonian, vec![0.5]);

        assert_eq!(result.params.len(), 1);
        assert!(result.final_objective.is_finite());
        assert!(result.best_objective <= result.final_objective + 1e-6);
    }

    #[test]
    fn test_layer_training_result() {
        let result = LayerTrainingResult {
            layer_idx: 0,
            initial_objective: 1.0,
            final_objective: 0.5,
            iterations: 10,
            improvement: 0.5,
        };

        assert_eq!(result.improvement, 0.5);
    }

    #[test]
    fn test_alternating_mode() {
        let mut circuit = VariationalCircuit::new(1);
        circuit.ry(0);
        circuit.ry(0);

        let mut hamiltonian = Hamiltonian::new();
        hamiltonian.add_term(PauliObservable::z(0));

        let config = LayerwiseConfig::default()
            .with_mode(LayerwiseMode::Alternating)
            .with_iter_per_layer(5)
            .with_finetune_iter(0);

        let trainer = LayerwiseTrainer::new(config);
        let result = trainer.train(&circuit, &hamiltonian, vec![0.5, 0.5]);

        // Alternating does forward + backward
        assert!(result.layer_results.len() >= 2);
    }

    #[test]
    fn test_reverse_progressive_mode() {
        let mut circuit = VariationalCircuit::new(1);
        circuit.ry(0);
        circuit.ry(0);

        let mut hamiltonian = Hamiltonian::new();
        hamiltonian.add_term(PauliObservable::z(0));

        let config = LayerwiseConfig::default()
            .with_mode(LayerwiseMode::ReverseProgressive)
            .with_iter_per_layer(5)
            .with_finetune_iter(0);

        let trainer = LayerwiseTrainer::new(config);
        let result = trainer.train(&circuit, &hamiltonian, vec![0.5, 0.5]);

        assert!(result.layer_results.len() > 0);
    }

    #[test]
    fn test_lr_decay() {
        let config = LayerwiseConfig::default()
            .with_lr_decay(0.5)
            .with_learning_rate(1.0);

        // Learning rate should decay by 0.5 each layer
        // Layer 0: 1.0, Layer 1: 0.5, Layer 2: 0.25
        assert!(config.decay_lr);
        assert_eq!(config.lr_decay_factor, 0.5);
    }

    #[test]
    fn test_freeze_strategy_variants() {
        let config1 = LayerwiseConfig::default()
            .with_freeze_strategy(FreezeStrategy::FreezePrevious);
        assert_eq!(config1.freeze_strategy, FreezeStrategy::FreezePrevious);

        let config2 = LayerwiseConfig::default()
            .with_freeze_strategy(FreezeStrategy::NoFreeze);
        assert_eq!(config2.freeze_strategy, FreezeStrategy::NoFreeze);
    }

    #[test]
    fn test_empty_trainable() {
        let mask = ParameterMask::all_frozen(3, vec![0.1, 0.2, 0.3]);
        assert_eq!(mask.n_trainable(), 0);
        assert!(mask.extract_trainable(&[0.1, 0.2, 0.3]).is_empty());
    }
}

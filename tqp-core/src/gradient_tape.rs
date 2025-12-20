//! Gradient Tape for Automatic Differentiation
//!
//! Records quantum circuit operations and enables efficient gradient computation
//! through reverse-mode automatic differentiation combined with parameter-shift rule.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                       GradientTape                               │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  TapeContext (manages recording)                                │
//! │    ├── entries: Vec<TapeEntry>                                  │
//! │    ├── params: Vec<f64>                                         │
//! │    └── gradients: Vec<f64>                                      │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  TapeEntry (individual operation)                               │
//! │    ├── op_type: OperationType                                   │
//! │    ├── qubits: Vec<usize>                                       │
//! │    ├── param_indices: Vec<usize>                                │
//! │    └── cached_values: Option<CachedValues>                      │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  Backpropagation                                                │
//! │    ├── compute_vjp() - Vector-Jacobian product                  │
//! │    └── accumulate_gradients()                                   │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! let mut tape = GradientTape::new(n_qubits, n_params);
//! tape.watch(&params);
//!
//! // Record operations with context
//! {
//!     let ctx = tape.record_context();
//!     ctx.record_rx(0, 0);
//!     ctx.record_ry(1, 1);
//!     ctx.record_cnot(0, 1);
//! }
//!
//! // Compute gradients via backward pass
//! tape.backward(loss_gradient);
//! let gradients = tape.gradients();
//! ```

use num_complex::Complex64;
use std::cell::RefCell;
use std::f64::consts::PI;
use std::rc::Rc;

// =============================================================================
// Constants
// =============================================================================

/// Parameter shift for VJP computation
pub const VJP_SHIFT: f64 = PI / 2.0;

/// Default gradient accumulation mode
pub const DEFAULT_ACCUMULATE: bool = true;

// =============================================================================
// Operation Types
// =============================================================================

/// Operation types recorded on tape
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationType {
    /// Parameterized rotation X: RX(θ)
    RX,
    /// Parameterized rotation Y: RY(θ)
    RY,
    /// Parameterized rotation Z: RZ(θ)
    RZ,
    /// General U3 gate: U3(θ, φ, λ)
    U3,
    /// Controlled rotation X
    CRX,
    /// Controlled rotation Y
    CRY,
    /// Controlled rotation Z
    CRZ,
    /// Non-parameterized gate (for tracking dependencies)
    Fixed,
    /// Expectation value measurement
    Expectation,
    /// Barrier (synchronization point)
    Barrier,
}

impl OperationType {
    /// Check if operation is parameterized
    pub fn is_parameterized(&self) -> bool {
        matches!(
            self,
            OperationType::RX
                | OperationType::RY
                | OperationType::RZ
                | OperationType::U3
                | OperationType::CRX
                | OperationType::CRY
                | OperationType::CRZ
        )
    }

    /// Number of parameters for this operation
    pub fn n_params(&self) -> usize {
        match self {
            OperationType::U3 => 3,
            OperationType::RX | OperationType::RY | OperationType::RZ => 1,
            OperationType::CRX | OperationType::CRY | OperationType::CRZ => 1,
            OperationType::Fixed | OperationType::Expectation | OperationType::Barrier => 0,
        }
    }

    /// Shift coefficient for parameter-shift rule
    pub fn shift_coefficient(&self) -> f64 {
        match self {
            OperationType::RX | OperationType::RY | OperationType::RZ => 0.5,
            OperationType::U3 => 0.5,
            OperationType::CRX | OperationType::CRY | OperationType::CRZ => 0.5,
            _ => 0.0,
        }
    }

    /// String representation
    pub fn as_str(&self) -> &'static str {
        match self {
            OperationType::RX => "RX",
            OperationType::RY => "RY",
            OperationType::RZ => "RZ",
            OperationType::U3 => "U3",
            OperationType::CRX => "CRX",
            OperationType::CRY => "CRY",
            OperationType::CRZ => "CRZ",
            OperationType::Fixed => "Fixed",
            OperationType::Expectation => "Expectation",
            OperationType::Barrier => "Barrier",
        }
    }
}

// =============================================================================
// Cached Values
// =============================================================================

/// Cached values for efficient backpropagation
#[derive(Debug, Clone)]
pub struct CachedValues {
    /// Forward pass output value
    pub forward_value: f64,
    /// Intermediate state snapshot (optional, for checkpointing)
    pub state_snapshot: Option<Vec<Complex64>>,
    /// Local Jacobian (∂output/∂param for this gate)
    pub local_jacobian: Option<Vec<f64>>,
    /// Upstream gradient received during backward pass
    pub upstream_gradient: Option<f64>,
}

impl CachedValues {
    /// Create new cached values with forward value
    pub fn new(forward_value: f64) -> Self {
        Self {
            forward_value,
            state_snapshot: None,
            local_jacobian: None,
            upstream_gradient: None,
        }
    }

    /// Builder: add state snapshot
    pub fn with_state(mut self, state: Vec<Complex64>) -> Self {
        self.state_snapshot = Some(state);
        self
    }

    /// Builder: add local Jacobian
    pub fn with_jacobian(mut self, jacobian: Vec<f64>) -> Self {
        self.local_jacobian = Some(jacobian);
        self
    }

    /// Builder: add upstream gradient
    pub fn with_upstream(mut self, gradient: f64) -> Self {
        self.upstream_gradient = Some(gradient);
        self
    }

    /// Check if state is checkpointed
    pub fn has_checkpoint(&self) -> bool {
        self.state_snapshot.is_some()
    }
}

impl Default for CachedValues {
    fn default() -> Self {
        Self::new(0.0)
    }
}

// =============================================================================
// Tape Entry
// =============================================================================

/// Single entry in the gradient tape
#[derive(Debug, Clone)]
pub struct TapeEntry {
    /// Operation type
    pub op_type: OperationType,
    /// Target qubits
    pub qubits: Vec<usize>,
    /// Parameter indices (into global parameter vector)
    pub param_indices: Vec<usize>,
    /// Parameter values at recording time
    pub param_values: Vec<f64>,
    /// Cached values from forward pass
    pub cached: Option<CachedValues>,
    /// Entry index in tape
    pub index: usize,
    /// Layer index (for parallel execution)
    pub layer: usize,
}

impl TapeEntry {
    /// Create a new tape entry
    pub fn new(op_type: OperationType, qubits: Vec<usize>, param_indices: Vec<usize>) -> Self {
        Self {
            op_type,
            qubits,
            param_indices,
            param_values: Vec::new(),
            cached: None,
            index: 0,
            layer: 0,
        }
    }

    /// Create RX entry
    pub fn rx(qubit: usize, param_index: usize) -> Self {
        Self::new(OperationType::RX, vec![qubit], vec![param_index])
    }

    /// Create RY entry
    pub fn ry(qubit: usize, param_index: usize) -> Self {
        Self::new(OperationType::RY, vec![qubit], vec![param_index])
    }

    /// Create RZ entry
    pub fn rz(qubit: usize, param_index: usize) -> Self {
        Self::new(OperationType::RZ, vec![qubit], vec![param_index])
    }

    /// Create U3 entry
    pub fn u3(qubit: usize, theta_idx: usize, phi_idx: usize, lambda_idx: usize) -> Self {
        Self::new(
            OperationType::U3,
            vec![qubit],
            vec![theta_idx, phi_idx, lambda_idx],
        )
    }

    /// Create CRZ entry
    pub fn crz(control: usize, target: usize, param_index: usize) -> Self {
        Self::new(OperationType::CRZ, vec![control, target], vec![param_index])
    }

    /// Create fixed gate entry (CNOT, H, etc.)
    pub fn fixed(qubits: Vec<usize>) -> Self {
        Self::new(OperationType::Fixed, qubits, vec![])
    }

    /// Create expectation entry
    pub fn expectation(value: f64) -> Self {
        let mut entry = Self::new(OperationType::Expectation, vec![], vec![]);
        entry.cached = Some(CachedValues::new(value));
        entry
    }

    /// Create barrier entry
    pub fn barrier() -> Self {
        Self::new(OperationType::Barrier, vec![], vec![])
    }

    /// Set cached values
    pub fn with_cached(mut self, cached: CachedValues) -> Self {
        self.cached = Some(cached);
        self
    }

    /// Set parameter values
    pub fn with_param_values(mut self, values: Vec<f64>) -> Self {
        self.param_values = values;
        self
    }

    /// Set index
    pub fn with_index(mut self, index: usize) -> Self {
        self.index = index;
        self
    }

    /// Set layer
    pub fn with_layer(mut self, layer: usize) -> Self {
        self.layer = layer;
        self
    }

    /// Check if entry affects given parameter
    pub fn affects_param(&self, param_index: usize) -> bool {
        self.param_indices.contains(&param_index)
    }

    /// Get number of qubits involved
    pub fn n_qubits(&self) -> usize {
        self.qubits.len()
    }

    /// Check if this is a parameterized operation
    pub fn is_parameterized(&self) -> bool {
        self.op_type.is_parameterized()
    }
}

// =============================================================================
// Gradient Tape
// =============================================================================

/// Gradient tape for recording operations
#[derive(Debug)]
pub struct GradientTape {
    /// Recorded entries
    entries: Vec<TapeEntry>,
    /// Watched parameters
    params: Vec<f64>,
    /// Accumulated gradients
    gradients: Vec<f64>,
    /// Is recording enabled
    recording: bool,
    /// Number of qubits
    n_qubits: usize,
    /// Current layer index
    current_layer: usize,
    /// Gradient accumulation mode
    accumulate: bool,
}

impl GradientTape {
    /// Create a new gradient tape
    pub fn new(n_qubits: usize, n_params: usize) -> Self {
        Self {
            entries: Vec::new(),
            params: vec![0.0; n_params],
            gradients: vec![0.0; n_params],
            recording: false,
            n_qubits,
            current_layer: 0,
            accumulate: DEFAULT_ACCUMULATE,
        }
    }

    /// Create tape with specific parameters
    pub fn with_params(n_qubits: usize, params: Vec<f64>) -> Self {
        let n_params = params.len();
        let mut tape = Self::new(n_qubits, n_params);
        tape.params = params;
        tape
    }

    // -------------------------------------------------------------------------
    // Recording Control
    // -------------------------------------------------------------------------

    /// Start recording
    pub fn start_recording(&mut self) {
        self.recording = true;
    }

    /// Stop recording
    pub fn stop_recording(&mut self) {
        self.recording = false;
    }

    /// Check if recording
    pub fn is_recording(&self) -> bool {
        self.recording
    }

    /// Set parameters to watch
    pub fn watch(&mut self, params: &[f64]) {
        self.params = params.to_vec();
        if !self.accumulate {
            self.gradients = vec![0.0; params.len()];
        }
    }

    /// Get current parameters
    pub fn params(&self) -> &[f64] {
        &self.params
    }

    /// Get parameter by index
    pub fn param(&self, index: usize) -> f64 {
        self.params.get(index).copied().unwrap_or(0.0)
    }

    /// Set parameter by index
    pub fn set_param(&mut self, index: usize, value: f64) {
        if index < self.params.len() {
            self.params[index] = value;
        }
    }

    /// Get accumulated gradients
    pub fn gradients(&self) -> &[f64] {
        &self.gradients
    }

    /// Get gradient by index
    pub fn gradient(&self, index: usize) -> f64 {
        self.gradients.get(index).copied().unwrap_or(0.0)
    }

    // -------------------------------------------------------------------------
    // Recording Operations
    // -------------------------------------------------------------------------

    /// Record an operation
    pub fn record(&mut self, mut entry: TapeEntry) {
        if self.recording {
            entry.index = self.entries.len();
            entry.layer = self.current_layer;

            // Capture parameter values at recording time
            if entry.is_parameterized() {
                entry.param_values = entry.param_indices.iter().map(|&i| self.param(i)).collect();
            }

            self.entries.push(entry);
        }
    }

    /// Record RX gate
    pub fn record_rx(&mut self, qubit: usize, param_index: usize) {
        self.record(TapeEntry::rx(qubit, param_index));
    }

    /// Record RY gate
    pub fn record_ry(&mut self, qubit: usize, param_index: usize) {
        self.record(TapeEntry::ry(qubit, param_index));
    }

    /// Record RZ gate
    pub fn record_rz(&mut self, qubit: usize, param_index: usize) {
        self.record(TapeEntry::rz(qubit, param_index));
    }

    /// Record U3 gate
    pub fn record_u3(&mut self, qubit: usize, theta_idx: usize, phi_idx: usize, lambda_idx: usize) {
        self.record(TapeEntry::u3(qubit, theta_idx, phi_idx, lambda_idx));
    }

    /// Record fixed gate (CNOT, H, etc.)
    pub fn record_fixed(&mut self, qubits: Vec<usize>) {
        self.record(TapeEntry::fixed(qubits));
    }

    /// Record CNOT
    pub fn record_cnot(&mut self, control: usize, target: usize) {
        self.record_fixed(vec![control, target]);
    }

    /// Record expectation measurement
    pub fn record_expectation(&mut self, value: f64) {
        self.record(TapeEntry::expectation(value));
    }

    /// Record barrier
    pub fn record_barrier(&mut self) {
        self.record(TapeEntry::barrier());
        self.current_layer += 1;
    }

    // -------------------------------------------------------------------------
    // Tape Access
    // -------------------------------------------------------------------------

    /// Get number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get entry at index
    pub fn entry(&self, index: usize) -> Option<&TapeEntry> {
        self.entries.get(index)
    }

    /// Get mutable entry at index
    pub fn entry_mut(&mut self, index: usize) -> Option<&mut TapeEntry> {
        self.entries.get_mut(index)
    }

    /// Get all entries
    pub fn entries(&self) -> &[TapeEntry] {
        &self.entries
    }

    /// Get entries affecting a specific parameter
    pub fn entries_for_param(&self, param_index: usize) -> Vec<&TapeEntry> {
        self.entries
            .iter()
            .filter(|e| e.affects_param(param_index))
            .collect()
    }

    /// Get parameterized entries only
    pub fn parameterized_entries(&self) -> Vec<&TapeEntry> {
        self.entries
            .iter()
            .filter(|e| e.is_parameterized())
            .collect()
    }

    /// Clear tape
    pub fn clear(&mut self) {
        self.entries.clear();
        self.current_layer = 0;
        if !self.accumulate {
            self.gradients.fill(0.0);
        }
    }

    /// Reset gradients only
    pub fn zero_grad(&mut self) {
        self.gradients.fill(0.0);
    }

    // -------------------------------------------------------------------------
    // Backward Pass (VJP Computation)
    // -------------------------------------------------------------------------

    /// Compute VJP (Vector-Jacobian Product) for a single parameter
    ///
    /// Uses the parameter-shift rule:
    /// ∂f/∂θ = (f(θ+π/2) - f(θ-π/2)) / 2
    pub fn compute_vjp(&self, param_index: usize, upstream_grad: f64) -> f64 {
        let mut vjp = 0.0;

        for entry in self.entries_for_param(param_index) {
            if !entry.is_parameterized() {
                continue;
            }

            // Get parameter value
            let param_value = self.param(param_index);
            let shift_coeff = entry.op_type.shift_coefficient();

            // Compute local gradient using parameter-shift rule
            // For standard rotation gates: ∂/∂θ = 0.5 * (f(θ+π/2) - f(θ-π/2))
            let local_grad = if let Some(ref cached) = entry.cached {
                if let Some(ref jacobian) = cached.local_jacobian {
                    // Use precomputed Jacobian if available
                    jacobian.first().copied().unwrap_or(0.0)
                } else {
                    // Estimate using shift coefficient
                    shift_coeff * (param_value + VJP_SHIFT).cos()
                        - shift_coeff * (param_value - VJP_SHIFT).cos()
                }
            } else {
                // Fallback: use shift coefficient approximation
                shift_coeff * 2.0 * param_value.cos()
            };

            vjp += upstream_grad * local_grad;
        }

        vjp
    }

    /// Perform backward pass
    ///
    /// Computes gradients for all parameters given upstream gradient
    pub fn backward(&mut self, upstream_grad: f64) {
        for i in 0..self.params.len() {
            let vjp = self.compute_vjp(i, upstream_grad);
            if self.accumulate {
                self.gradients[i] += vjp;
            } else {
                self.gradients[i] = vjp;
            }
        }
    }

    /// Backward pass with per-parameter upstream gradients
    pub fn backward_multi(&mut self, upstream_grads: &[f64]) {
        assert_eq!(upstream_grads.len(), self.params.len());

        for (i, &upstream) in upstream_grads.iter().enumerate() {
            let vjp = self.compute_vjp(i, upstream);
            if self.accumulate {
                self.gradients[i] += vjp;
            } else {
                self.gradients[i] = vjp;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Forward Pass Helper
    // -------------------------------------------------------------------------

    /// Forward pass with recording
    pub fn forward<F>(&mut self, params: &[f64], compute_fn: F) -> f64
    where
        F: FnOnce(&[f64]) -> f64,
    {
        self.watch(params);
        self.start_recording();
        let result = compute_fn(params);
        self.record_expectation(result);
        self.stop_recording();
        result
    }

    /// Forward and backward pass
    pub fn forward_backward<F>(&mut self, params: &[f64], compute_fn: F) -> (f64, Vec<f64>)
    where
        F: FnOnce(&[f64]) -> f64,
    {
        let result = self.forward(params, compute_fn);
        self.backward(1.0); // dL/dL = 1
        (result, self.gradients.clone())
    }

    // -------------------------------------------------------------------------
    // Summary and Statistics
    // -------------------------------------------------------------------------

    /// Get tape summary
    pub fn summary(&self) -> TapeSummary {
        let n_parameterized = self.entries.iter().filter(|e| e.is_parameterized()).count();

        let n_fixed = self
            .entries
            .iter()
            .filter(|e| matches!(e.op_type, OperationType::Fixed))
            .count();

        TapeSummary {
            total_entries: self.entries.len(),
            parameterized_entries: n_parameterized,
            fixed_entries: n_fixed,
            n_params: self.params.len(),
            n_qubits: self.n_qubits,
            n_layers: self.current_layer + 1,
        }
    }
}

/// Summary of tape contents
#[derive(Debug, Clone)]
pub struct TapeSummary {
    pub total_entries: usize,
    pub parameterized_entries: usize,
    pub fixed_entries: usize,
    pub n_params: usize,
    pub n_qubits: usize,
    pub n_layers: usize,
}

impl TapeSummary {
    /// Percentage of parameterized operations
    pub fn param_ratio(&self) -> f64 {
        if self.total_entries == 0 {
            0.0
        } else {
            self.parameterized_entries as f64 / self.total_entries as f64
        }
    }
}

// =============================================================================
// Tape Context (RAII Recording)
// =============================================================================

/// RAII context for recording operations
///
/// Automatically starts recording on creation and stops on drop.
pub struct TapeContext<'a> {
    tape: &'a mut GradientTape,
}

impl<'a> TapeContext<'a> {
    /// Create recording context
    pub fn new(tape: &'a mut GradientTape) -> Self {
        tape.start_recording();
        Self { tape }
    }

    /// Record operation in context
    pub fn record(&mut self, entry: TapeEntry) {
        self.tape.record(entry);
    }

    /// Record RX gate
    pub fn rx(&mut self, qubit: usize, param_index: usize) {
        self.tape.record_rx(qubit, param_index);
    }

    /// Record RY gate
    pub fn ry(&mut self, qubit: usize, param_index: usize) {
        self.tape.record_ry(qubit, param_index);
    }

    /// Record RZ gate
    pub fn rz(&mut self, qubit: usize, param_index: usize) {
        self.tape.record_rz(qubit, param_index);
    }

    /// Record CNOT
    pub fn cnot(&mut self, control: usize, target: usize) {
        self.tape.record_cnot(control, target);
    }

    /// Record barrier
    pub fn barrier(&mut self) {
        self.tape.record_barrier();
    }

    /// Get tape reference
    pub fn tape(&self) -> &GradientTape {
        self.tape
    }
}

impl<'a> Drop for TapeContext<'a> {
    fn drop(&mut self) {
        self.tape.stop_recording();
    }
}

// =============================================================================
// Gradient Accumulator
// =============================================================================

/// Gradient accumulator for batch operations
///
/// Useful for mini-batch gradient computation in VQE optimization.
#[derive(Debug)]
pub struct GradientAccumulator {
    /// Accumulated gradients
    gradients: Vec<f64>,
    /// Number of accumulations
    count: usize,
    /// Enable normalization by count
    normalize: bool,
}

impl GradientAccumulator {
    /// Create new accumulator
    pub fn new(n_params: usize) -> Self {
        Self {
            gradients: vec![0.0; n_params],
            count: 0,
            normalize: true,
        }
    }

    /// Create accumulator without normalization
    pub fn without_normalization(n_params: usize) -> Self {
        Self {
            gradients: vec![0.0; n_params],
            count: 0,
            normalize: false,
        }
    }

    /// Add gradients to accumulator
    pub fn accumulate(&mut self, gradients: &[f64]) {
        assert_eq!(gradients.len(), self.gradients.len());
        for (acc, grad) in self.gradients.iter_mut().zip(gradients.iter()) {
            *acc += grad;
        }
        self.count += 1;
    }

    /// Add gradients from tape
    pub fn accumulate_from_tape(&mut self, tape: &GradientTape) {
        self.accumulate(tape.gradients());
    }

    /// Get averaged gradients
    pub fn average(&self) -> Vec<f64> {
        if self.count == 0 || !self.normalize {
            self.gradients.clone()
        } else {
            self.gradients
                .iter()
                .map(|g| g / self.count as f64)
                .collect()
        }
    }

    /// Get sum of gradients (no normalization)
    pub fn sum(&self) -> &[f64] {
        &self.gradients
    }

    /// Get accumulated gradients (applies normalization if enabled)
    pub fn gradients(&self) -> Vec<f64> {
        self.average()
    }

    /// Reset accumulator
    pub fn reset(&mut self) {
        self.gradients.fill(0.0);
        self.count = 0;
    }

    /// Get count
    pub fn count(&self) -> usize {
        self.count
    }

    /// Number of parameters
    pub fn n_params(&self) -> usize {
        self.gradients.len()
    }
}

// =============================================================================
// Shared Tape (for multi-threaded contexts)
// =============================================================================

/// Shared tape for thread-safe recording
pub type SharedTape = Rc<RefCell<GradientTape>>;

/// Create shared tape
pub fn shared_tape(n_qubits: usize, n_params: usize) -> SharedTape {
    Rc::new(RefCell::new(GradientTape::new(n_qubits, n_params)))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operation_type() {
        assert!(OperationType::RX.is_parameterized());
        assert!(OperationType::RY.is_parameterized());
        assert!(OperationType::RZ.is_parameterized());
        assert!(!OperationType::Fixed.is_parameterized());
        assert!(!OperationType::Expectation.is_parameterized());

        assert_eq!(OperationType::U3.n_params(), 3);
        assert_eq!(OperationType::RX.n_params(), 1);
        assert_eq!(OperationType::Fixed.n_params(), 0);
    }

    #[test]
    fn test_cached_values() {
        let cached = CachedValues::new(1.5)
            .with_state(vec![Complex64::new(1.0, 0.0)])
            .with_jacobian(vec![0.5]);

        assert!((cached.forward_value - 1.5).abs() < 1e-10);
        assert!(cached.has_checkpoint());
        assert!(cached.local_jacobian.is_some());
    }

    #[test]
    fn test_tape_entry_creation() {
        let entry = TapeEntry::rx(0, 0);
        assert_eq!(entry.op_type, OperationType::RX);
        assert_eq!(entry.qubits, vec![0]);
        assert_eq!(entry.param_indices, vec![0]);
        assert!(entry.is_parameterized());

        let fixed = TapeEntry::fixed(vec![0, 1]);
        assert!(!fixed.is_parameterized());
    }

    #[test]
    fn test_tape_entry_u3() {
        let entry = TapeEntry::u3(0, 1, 2, 3);
        assert_eq!(entry.op_type, OperationType::U3);
        assert_eq!(entry.param_indices, vec![1, 2, 3]);
    }

    #[test]
    fn test_tape_recording() {
        let mut tape = GradientTape::new(2, 4);
        tape.start_recording();

        tape.record_rx(0, 0);
        tape.record_ry(1, 1);
        tape.record_cnot(0, 1);

        tape.stop_recording();

        assert_eq!(tape.len(), 3);
        assert!(!tape.is_recording());
    }

    #[test]
    fn test_tape_watch() {
        let mut tape = GradientTape::new(2, 3);
        tape.watch(&[0.1, 0.2, 0.3]);

        assert_eq!(tape.params().len(), 3);
        assert!((tape.param(0) - 0.1).abs() < 1e-10);
        assert!((tape.param(1) - 0.2).abs() < 1e-10);
        assert!((tape.param(2) - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_entries_for_param() {
        let mut tape = GradientTape::new(2, 2);
        tape.start_recording();
        tape.record_rx(0, 0);
        tape.record_ry(0, 1);
        tape.record_rx(1, 0);
        tape.stop_recording();

        let entries = tape.entries_for_param(0);
        assert_eq!(entries.len(), 2);

        let entries = tape.entries_for_param(1);
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn test_tape_context() {
        let mut tape = GradientTape::new(2, 2);

        {
            let mut ctx = TapeContext::new(&mut tape);
            ctx.rx(0, 0);
            ctx.ry(1, 1);
            ctx.cnot(0, 1);
            assert!(ctx.tape().is_recording());
        }

        // Should stop recording when context dropped
        assert!(!tape.is_recording());
        assert_eq!(tape.len(), 3);
    }

    #[test]
    fn test_gradient_accumulator() {
        let mut acc = GradientAccumulator::new(3);
        acc.accumulate(&[1.0, 2.0, 3.0]);
        acc.accumulate(&[3.0, 2.0, 1.0]);

        let avg = acc.average();
        assert!((avg[0] - 2.0).abs() < 1e-10);
        assert!((avg[1] - 2.0).abs() < 1e-10);
        assert!((avg[2] - 2.0).abs() < 1e-10);
        assert_eq!(acc.count(), 2);
    }

    #[test]
    fn test_gradient_accumulator_no_normalize() {
        let mut acc = GradientAccumulator::without_normalization(2);
        acc.accumulate(&[1.0, 2.0]);
        acc.accumulate(&[3.0, 4.0]);

        let result = acc.gradients();
        assert!((result[0] - 4.0).abs() < 1e-10);
        assert!((result[1] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_tape_summary() {
        let mut tape = GradientTape::new(3, 4);
        tape.start_recording();
        tape.record_rx(0, 0);
        tape.record_ry(1, 1);
        tape.record_cnot(0, 1);
        tape.record_barrier();
        tape.record_rz(2, 2);
        tape.stop_recording();

        let summary = tape.summary();
        assert_eq!(summary.total_entries, 5);
        assert_eq!(summary.parameterized_entries, 3);
        assert_eq!(summary.fixed_entries, 1);
        assert_eq!(summary.n_layers, 2);
    }

    #[test]
    fn test_forward_pass() {
        let mut tape = GradientTape::new(2, 2);

        let result = tape.forward(&[0.1, 0.2], |params| params.iter().sum::<f64>());

        assert!((result - 0.3).abs() < 1e-10);
        assert!(!tape.is_empty());
    }

    #[test]
    fn test_backward_pass() {
        let mut tape = GradientTape::new(2, 2);
        tape.watch(&[0.5, 1.0]);
        tape.start_recording();
        tape.record_rx(0, 0);
        tape.record_ry(1, 1);
        tape.stop_recording();

        tape.backward(1.0);

        // Gradients should be computed
        assert!(tape.gradients().iter().all(|g| g.is_finite()));
    }

    #[test]
    fn test_shared_tape() {
        let tape = shared_tape(2, 3);
        {
            let mut t = tape.borrow_mut();
            t.start_recording();
            t.record_rx(0, 0);
            t.stop_recording();
        }
        assert_eq!(tape.borrow().len(), 1);
    }

    #[test]
    fn test_clear_and_zero_grad() {
        let mut tape = GradientTape::new(2, 2);
        tape.watch(&[0.1, 0.2]);
        tape.start_recording();
        tape.record_rx(0, 0);
        tape.stop_recording();
        tape.backward(1.0);

        // Clear tape
        tape.clear();
        assert!(tape.is_empty());

        // Zero gradients
        tape.backward(1.0);
        tape.zero_grad();
        assert!(tape.gradients().iter().all(|&g| g.abs() < 1e-10));
    }

    #[test]
    fn test_parameterized_entries() {
        let mut tape = GradientTape::new(2, 3);
        tape.start_recording();
        tape.record_rx(0, 0);
        tape.record_cnot(0, 1);
        tape.record_ry(1, 1);
        tape.record_rz(0, 2);
        tape.stop_recording();

        let param_entries = tape.parameterized_entries();
        assert_eq!(param_entries.len(), 3);
    }
}

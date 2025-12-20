//! Memory Policy for TQP Sparse/Dense State Vectors
//!
//! PPR-based automatic switching between Dense and Sparse representations.
//!
//! # 3P Structure
//! - Perceive: Collect state metrics (qubit count, sparsity ratio)
//! - Process: Evaluate policy rules and estimate memory
//! - Response: Execute or recommend conversion
//!
//! # Design Rationale (PPR Simulation Verified)
//! - N >= 20 qubits AND sparsity < 10% → Sparse recommended
//! - Dense memory > 1GB → Force Sparse
//! - Sparse with sparsity > 50% → Convert to Dense
//!
//! Validated with 9 simulation scenarios before implementation.

use crate::sparse::SparseStateVector;
use crate::state::TQPState;

/// Memory policy configuration
#[derive(Debug, Clone)]
pub struct PolicyConfig {
    /// Qubit threshold for considering sparse (default: 20)
    pub qubit_threshold: usize,
    /// Sparsity ratio threshold (default: 0.1 = 10%)
    pub sparsity_ratio_threshold: f64,
    /// Maximum dense memory in bytes (default: 1GB)
    pub max_dense_bytes: usize,
    /// Conversion overhead factor (default: 1.5)
    pub conversion_overhead_factor: f64,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        PolicyConfig {
            qubit_threshold: 20,
            sparsity_ratio_threshold: 0.1,
            max_dense_bytes: 1 << 30, // 1GB
            conversion_overhead_factor: 1.5,
        }
    }
}

impl PolicyConfig {
    /// Create a config optimized for memory efficiency
    pub fn memory_optimized() -> Self {
        PolicyConfig {
            qubit_threshold: 15,
            sparsity_ratio_threshold: 0.2,
            max_dense_bytes: 512 << 20, // 512MB
            conversion_overhead_factor: 1.2,
        }
    }

    /// Create a config optimized for performance
    pub fn performance_optimized() -> Self {
        PolicyConfig {
            qubit_threshold: 25,
            sparsity_ratio_threshold: 0.05,
            max_dense_bytes: 4 << 30, // 4GB
            conversion_overhead_factor: 2.0,
        }
    }
}

/// State representation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateRepresentation {
    Dense,
    Sparse,
}

/// State metrics collected during Perceive phase
#[derive(Debug, Clone)]
pub struct StateMetrics {
    pub num_qubits: usize,
    pub num_time_bins: usize,
    pub num_layers: usize,
    pub nnz: usize,
    pub current_repr: StateRepresentation,
}

impl StateMetrics {
    /// Total Hilbert space dimension
    #[inline]
    pub fn total_dim(&self) -> usize {
        self.num_layers * self.num_time_bins * (1 << self.num_qubits)
    }

    /// Sparsity ratio (nnz / total_dim)
    #[inline]
    pub fn sparsity_ratio(&self) -> f64 {
        let dim = self.total_dim();
        if dim == 0 {
            0.0
        } else {
            self.nnz as f64 / dim as f64
        }
    }

    /// Check if state is considered large
    pub fn is_large(&self, threshold: usize) -> bool {
        self.num_qubits >= threshold
    }

    /// Check if state is sparse
    pub fn is_sparse(&self, threshold: f64) -> bool {
        self.sparsity_ratio() < threshold
    }
}

/// Memory usage estimate
#[derive(Debug, Clone)]
pub struct MemoryEstimate {
    /// Dense representation memory (bytes)
    pub dense_bytes: usize,
    /// Sparse representation memory (bytes)
    pub sparse_bytes: usize,
}

impl MemoryEstimate {
    /// Calculate sparse/dense ratio
    #[inline]
    pub fn ratio(&self) -> f64 {
        if self.dense_bytes == 0 {
            f64::INFINITY
        } else {
            self.sparse_bytes as f64 / self.dense_bytes as f64
        }
    }

    /// Check if sparse is more memory efficient
    #[inline]
    pub fn sparse_is_better(&self) -> bool {
        self.sparse_bytes < self.dense_bytes
    }

    /// Memory savings if converting to sparse (can be negative)
    #[inline]
    pub fn sparse_savings(&self) -> i64 {
        self.dense_bytes as i64 - self.sparse_bytes as i64
    }
}

/// Policy decision action
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyAction {
    /// Keep current representation
    Keep,
    /// Convert to sparse representation
    ToSparse,
    /// Convert to dense representation
    ToDense,
}

/// Policy decision result
#[derive(Debug, Clone)]
pub struct PolicyDecision {
    /// Recommended action
    pub action: PolicyAction,
    /// Human-readable reason
    pub reason: String,
    /// Estimated memory change (positive = saving)
    pub memory_delta: i64,
    /// Decision confidence (0.0 - 1.0)
    pub confidence: f64,
}

/// Memory policy engine implementing PPR 3P structure
pub struct MemoryPolicy {
    config: PolicyConfig,
}

impl MemoryPolicy {
    /// Create with default configuration
    pub fn new() -> Self {
        Self::with_config(PolicyConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: PolicyConfig) -> Self {
        MemoryPolicy { config }
    }

    /// Get current configuration
    pub fn config(&self) -> &PolicyConfig {
        &self.config
    }

    // =========================================================================
    // Perceive: Collect state metrics
    // =========================================================================

    /// Perceive metrics from a dense state
    pub fn perceive_dense(&self, state: &TQPState) -> StateMetrics {
        let nnz = state
            .state_vector
            .iter()
            .filter(|c| c.norm_sqr() >= 1e-14)
            .count();

        StateMetrics {
            num_qubits: state.dims.num_qubits,
            num_time_bins: state.dims.num_time_bins,
            num_layers: state.dims.num_layers,
            nnz,
            current_repr: StateRepresentation::Dense,
        }
    }

    /// Perceive metrics from a sparse state
    pub fn perceive_sparse(&self, state: &SparseStateVector) -> StateMetrics {
        StateMetrics {
            num_qubits: state.dims.num_qubits,
            num_time_bins: state.dims.num_time_bins,
            num_layers: state.dims.num_layers,
            nnz: state.nnz(),
            current_repr: StateRepresentation::Sparse,
        }
    }

    // =========================================================================
    // Process: Evaluate policy and estimate memory
    // =========================================================================

    /// Estimate memory usage for both representations
    ///
    /// Dense: total_dim * 16 bytes (Complex64)
    /// Sparse: 48 bytes overhead + nnz * 24 bytes (usize + Complex64)
    pub fn estimate_memory(&self, metrics: &StateMetrics) -> MemoryEstimate {
        let dense_bytes = metrics.total_dim() * 16; // Complex64 = 16 bytes
        let sparse_bytes = 48 + metrics.nnz * 24; // FxHashMap overhead + elements

        MemoryEstimate {
            dense_bytes,
            sparse_bytes,
        }
    }

    /// Make policy decision based on metrics
    ///
    /// Decision Rules (PPR Verified):
    /// 1. Memory limit exceeded + dense → ToSparse (forced, highest priority)
    /// 2. Large (N >= threshold) + sparse + dense → ToSparse
    /// 3. Sparse + high density (> 50%) + within limits → ToDense
    /// 4. Otherwise → Keep
    pub fn decide(&self, metrics: &StateMetrics) -> PolicyDecision {
        let estimate = self.estimate_memory(metrics);
        let sparsity = metrics.sparsity_ratio();

        // Rule 1: Memory limit exceeded → force sparse (HIGHEST PRIORITY)
        if estimate.dense_bytes > self.config.max_dense_bytes
            && metrics.current_repr == StateRepresentation::Dense
        {
            return PolicyDecision {
                action: PolicyAction::ToSparse,
                reason: format!(
                    "Dense memory ({:.2}GB) exceeds limit ({:.2}GB)",
                    estimate.dense_bytes as f64 / 1e9,
                    self.config.max_dense_bytes as f64 / 1e9
                ),
                memory_delta: estimate.sparse_savings(),
                confidence: 1.0,
            };
        }

        // Rule 2: Large + sparse + currently dense → to_sparse
        if metrics.is_large(self.config.qubit_threshold)
            && metrics.is_sparse(self.config.sparsity_ratio_threshold)
            && metrics.current_repr == StateRepresentation::Dense
        {
            return PolicyDecision {
                action: PolicyAction::ToSparse,
                reason: format!(
                    "Large qubit count (N={}) with low sparsity ({:.4})",
                    metrics.num_qubits, sparsity
                ),
                memory_delta: estimate.sparse_savings(),
                confidence: 0.9,
            };
        }

        // Rule 3: Sparse but too dense → to_dense
        if metrics.current_repr == StateRepresentation::Sparse
            && sparsity > 0.5
            && estimate.dense_bytes <= self.config.max_dense_bytes
        {
            return PolicyDecision {
                action: PolicyAction::ToDense,
                reason: format!(
                    "High density ratio ({:.2}%), dense more efficient",
                    sparsity * 100.0
                ),
                memory_delta: -(estimate.sparse_savings()), // Negative of sparse savings
                confidence: 0.7,
            };
        }

        // Rule 4: Keep current
        PolicyDecision {
            action: PolicyAction::Keep,
            reason: "Current representation is optimal".to_string(),
            memory_delta: 0,
            confidence: 0.8,
        }
    }

    // =========================================================================
    // Response: Execute conversion
    // =========================================================================

    /// Check if conversion is recommended
    pub fn should_convert(&self, metrics: &StateMetrics) -> (bool, PolicyAction) {
        let decision = self.decide(metrics);
        match decision.action {
            PolicyAction::Keep => (false, PolicyAction::Keep),
            action => (true, action),
        }
    }

    /// Get full recommendation for a dense state
    pub fn recommend_for_dense(&self, state: &TQPState) -> PolicyDecision {
        let metrics = self.perceive_dense(state);
        self.decide(&metrics)
    }

    /// Get full recommendation for a sparse state
    pub fn recommend_for_sparse(&self, state: &SparseStateVector) -> PolicyDecision {
        let metrics = self.perceive_sparse(state);
        self.decide(&metrics)
    }
}

impl Default for MemoryPolicy {
    fn default() -> Self {
        Self::new()
    }
}

/// Unified state that can be either Dense or Sparse
pub enum UnifiedState {
    Dense(TQPState),
    Sparse(SparseStateVector),
}

impl UnifiedState {
    /// Create from dense state
    pub fn from_dense(state: TQPState) -> Self {
        UnifiedState::Dense(state)
    }

    /// Create from sparse state
    pub fn from_sparse(state: SparseStateVector) -> Self {
        UnifiedState::Sparse(state)
    }

    /// Get current representation type
    pub fn representation(&self) -> StateRepresentation {
        match self {
            UnifiedState::Dense(_) => StateRepresentation::Dense,
            UnifiedState::Sparse(_) => StateRepresentation::Sparse,
        }
    }

    /// Get total dimension
    pub fn dimension(&self) -> usize {
        match self {
            UnifiedState::Dense(s) => s.dimension(),
            UnifiedState::Sparse(s) => s.dimension(),
        }
    }

    /// Get number of non-zero elements
    pub fn nnz(&self) -> usize {
        match self {
            UnifiedState::Dense(s) => s
                .state_vector
                .iter()
                .filter(|c| c.norm_sqr() >= 1e-14)
                .count(),
            UnifiedState::Sparse(s) => s.nnz(),
        }
    }

    /// Apply policy and convert if needed
    ///
    /// Returns the action taken
    pub fn apply_policy(&mut self, policy: &MemoryPolicy) -> PolicyAction {
        let metrics = match self {
            UnifiedState::Dense(s) => policy.perceive_dense(s),
            UnifiedState::Sparse(s) => policy.perceive_sparse(s),
        };

        let decision = policy.decide(&metrics);

        match decision.action {
            PolicyAction::ToSparse => {
                if let UnifiedState::Dense(dense) = self {
                    let sparse = SparseStateVector::from_dense(dense);
                    *self = UnifiedState::Sparse(sparse);
                }
            }
            PolicyAction::ToDense => {
                if let UnifiedState::Sparse(sparse) = self {
                    let dense = sparse.to_dense();
                    *self = UnifiedState::Dense(dense);
                }
            }
            PolicyAction::Keep => {}
        }

        decision.action
    }

    /// Force conversion to dense
    pub fn to_dense_state(&self) -> TQPState {
        match self {
            UnifiedState::Dense(s) => s.clone(),
            UnifiedState::Sparse(s) => s.to_dense(),
        }
    }

    /// Force conversion to sparse
    pub fn to_sparse_state(&self) -> SparseStateVector {
        match self {
            UnifiedState::Dense(s) => SparseStateVector::from_dense(s),
            UnifiedState::Sparse(s) => s.clone(),
        }
    }
}

/// Auto-managed state that applies policy automatically
pub struct ManagedState {
    state: UnifiedState,
    policy: MemoryPolicy,
    auto_convert: bool,
}

impl ManagedState {
    /// Create a new managed dense state
    pub fn new_dense(num_qubits: usize, num_time_bins: usize, num_layers: usize) -> Self {
        let state = TQPState::new(num_qubits, num_time_bins, num_layers);
        ManagedState {
            state: UnifiedState::Dense(state),
            policy: MemoryPolicy::new(),
            auto_convert: true,
        }
    }

    /// Create with custom policy
    pub fn with_policy(state: UnifiedState, policy: MemoryPolicy) -> Self {
        ManagedState {
            state,
            policy,
            auto_convert: true,
        }
    }

    /// Enable/disable auto conversion
    pub fn set_auto_convert(&mut self, enabled: bool) {
        self.auto_convert = enabled;
    }

    /// Check and apply policy if auto_convert is enabled
    pub fn check_policy(&mut self) -> PolicyAction {
        if self.auto_convert {
            self.state.apply_policy(&self.policy)
        } else {
            PolicyAction::Keep
        }
    }

    /// Get reference to inner state
    pub fn inner(&self) -> &UnifiedState {
        &self.state
    }

    /// Get mutable reference to inner state
    pub fn inner_mut(&mut self) -> &mut UnifiedState {
        &mut self.state
    }

    /// Get current representation
    pub fn representation(&self) -> StateRepresentation {
        self.state.representation()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // PPR Simulation Scenario 1: Small Dense - Keep
    #[test]
    fn test_small_dense_keep() {
        let policy = MemoryPolicy::new();
        let metrics = StateMetrics {
            num_qubits: 5,
            num_time_bins: 1,
            num_layers: 1,
            nnz: 32,
            current_repr: StateRepresentation::Dense,
        };

        let decision = policy.decide(&metrics);
        assert_eq!(decision.action, PolicyAction::Keep);
        assert!(decision.confidence >= 0.7);
    }

    // PPR Simulation Scenario 2: Large Sparse - Convert to Sparse
    #[test]
    fn test_large_sparse_convert() {
        let policy = MemoryPolicy::new();
        let metrics = StateMetrics {
            num_qubits: 25,
            num_time_bins: 1,
            num_layers: 1,
            nnz: 1000,
            current_repr: StateRepresentation::Dense,
        };

        let decision = policy.decide(&metrics);
        assert_eq!(decision.action, PolicyAction::ToSparse);
        assert!(decision.memory_delta > 0, "Should save memory");
        assert_eq!(decision.confidence, 0.9);
    }

    // PPR Simulation Scenario 3: Sparse but Dense - Convert to Dense
    #[test]
    fn test_sparse_to_dense() {
        let policy = MemoryPolicy::new();
        let metrics = StateMetrics {
            num_qubits: 10,
            num_time_bins: 1,
            num_layers: 1,
            nnz: 800, // 78% dense
            current_repr: StateRepresentation::Sparse,
        };

        let decision = policy.decide(&metrics);
        assert_eq!(decision.action, PolicyAction::ToDense);
        assert_eq!(decision.confidence, 0.7);
    }

    // PPR Simulation Scenario 4: Medium Size - Keep Current
    #[test]
    fn test_medium_keep() {
        let policy = MemoryPolicy::new();
        let metrics = StateMetrics {
            num_qubits: 15,
            num_time_bins: 1,
            num_layers: 1,
            nnz: 16384, // 50% dense
            current_repr: StateRepresentation::Dense,
        };

        let decision = policy.decide(&metrics);
        assert_eq!(decision.action, PolicyAction::Keep);
    }

    // PPR Simulation Scenario 5: Memory Limit Exceeded
    #[test]
    fn test_memory_limit_force_sparse() {
        let policy = MemoryPolicy::new();
        let metrics = StateMetrics {
            num_qubits: 30, // 2^30 * 16 = 17GB
            num_time_bins: 1,
            num_layers: 1,
            nnz: 1000,
            current_repr: StateRepresentation::Dense,
        };

        let decision = policy.decide(&metrics);
        assert_eq!(decision.action, PolicyAction::ToSparse);
        assert_eq!(
            decision.confidence, 1.0,
            "Forced conversion should have max confidence"
        );
    }

    // PPR Simulation Scenario 6: Optimal Sparse - Keep
    #[test]
    fn test_optimal_sparse_keep() {
        let policy = MemoryPolicy::new();
        let metrics = StateMetrics {
            num_qubits: 20,
            num_time_bins: 1,
            num_layers: 1,
            nnz: 100,
            current_repr: StateRepresentation::Sparse,
        };

        let decision = policy.decide(&metrics);
        assert_eq!(decision.action, PolicyAction::Keep);
    }

    // PPR Simulation Scenario 7: Multi-layer Multi-bin
    #[test]
    fn test_multi_layer_bin() {
        let policy = MemoryPolicy::new();
        let metrics = StateMetrics {
            num_qubits: 20,
            num_time_bins: 4,
            num_layers: 2,
            nnz: 500,
            current_repr: StateRepresentation::Dense,
        };

        let decision = policy.decide(&metrics);
        assert_eq!(decision.action, PolicyAction::ToSparse);
    }

    // PPR Simulation Scenario 8: Boundary - Just Below Threshold
    #[test]
    fn test_boundary_below() {
        let policy = MemoryPolicy::new();
        let metrics = StateMetrics {
            num_qubits: 20,
            num_time_bins: 1,
            num_layers: 1,
            nnz: 104857, // ~10% but < 0.1
            current_repr: StateRepresentation::Dense,
        };

        let sparsity = metrics.sparsity_ratio();
        assert!(sparsity < 0.1, "Should be just below threshold");

        let decision = policy.decide(&metrics);
        assert_eq!(decision.action, PolicyAction::ToSparse);
    }

    // PPR Simulation Scenario 9: Boundary - Above Threshold
    #[test]
    fn test_boundary_above() {
        let policy = MemoryPolicy::new();
        let metrics = StateMetrics {
            num_qubits: 20,
            num_time_bins: 1,
            num_layers: 1,
            nnz: 104858, // > 10%
            current_repr: StateRepresentation::Dense,
        };

        let sparsity = metrics.sparsity_ratio();
        assert!(sparsity >= 0.1, "Should be at or above threshold");

        let decision = policy.decide(&metrics);
        assert_eq!(decision.action, PolicyAction::Keep);
    }

    #[test]
    fn test_memory_estimate() {
        let policy = MemoryPolicy::new();
        let metrics = StateMetrics {
            num_qubits: 10,
            num_time_bins: 1,
            num_layers: 1,
            nnz: 100,
            current_repr: StateRepresentation::Dense,
        };

        let estimate = policy.estimate_memory(&metrics);
        assert_eq!(estimate.dense_bytes, 1024 * 16); // 2^10 * 16
        assert_eq!(estimate.sparse_bytes, 48 + 100 * 24); // overhead + 100 elements
        assert!(estimate.sparse_is_better());
    }

    #[test]
    fn test_unified_state_conversion() {
        let policy = MemoryPolicy::new();

        // Create a large sparse state that should convert
        let mut state = UnifiedState::Dense(TQPState::new(20, 1, 1));

        // Initially dense
        assert_eq!(state.representation(), StateRepresentation::Dense);

        // After policy application, should still be dense (initial state is dense)
        // because nnz check on fresh state will show it as very sparse
        let action = state.apply_policy(&policy);

        // Fresh state has nnz=1, very sparse, should convert
        assert_eq!(action, PolicyAction::ToSparse);
        assert_eq!(state.representation(), StateRepresentation::Sparse);
    }

    #[test]
    fn test_config_presets() {
        let mem_opt = PolicyConfig::memory_optimized();
        assert_eq!(mem_opt.qubit_threshold, 15);
        assert!(mem_opt.max_dense_bytes < PolicyConfig::default().max_dense_bytes);

        let perf_opt = PolicyConfig::performance_optimized();
        assert_eq!(perf_opt.qubit_threshold, 25);
        assert!(perf_opt.max_dense_bytes > PolicyConfig::default().max_dense_bytes);
    }
}

use crate::state::TQPState;

/// Defines the memory management policy for the quantum state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPolicy {
    /// Use dense array representation (ndarray::Array1).
    /// Best for high-entanglement states or small systems.
    Dense,
    /// Use sparse matrix representation (e.g., CSR).
    /// Best for low-entanglement states or very large systems.
    /// (Note: Sparse implementation is a placeholder for MVP).
    Sparse,
}

/// Manages memory allocation and representation for the TQP system.
pub struct MemoryManager {
    pub current_policy: MemoryPolicy,
    pub sparsity_threshold: f64,
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryManager {
    pub fn new() -> Self {
        Self {
            current_policy: MemoryPolicy::Dense,
            sparsity_threshold: 0.1, // If non-zero elements < 10%, switch to sparse
        }
    }

    /// Decides the best memory policy based on the current state.
    pub fn decide_policy(&self, state: &TQPState) -> MemoryPolicy {
        // For MVP, we only support Dense.
        // In the future, we would check state.count_non_zero() / total_dim.

        let total_dim = state.dims.total_dim();
        if total_dim > 1_000_000 {
            // Placeholder logic: if state is huge, suggest Sparse
            // But since we don't have Sparse impl yet, return Dense with a warning log?
            // For now, just return Dense.
            MemoryPolicy::Dense
        } else {
            MemoryPolicy::Dense
        }
    }

    /// Optimizes the storage of the given state based on the decided policy.
    pub fn optimize_storage(&mut self, state: &mut TQPState) {
        let new_policy = self.decide_policy(state);

        if new_policy != self.current_policy {
            // Perform conversion (Placeholder)
            // if new_policy == Sparse { convert_to_sparse(state); }
            // else { convert_to_dense(state); }

            self.current_policy = new_policy;
        }
    }
}

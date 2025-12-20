use ndarray::Array1;
use num_complex::Complex64;

/// Dimensions of the TQP system
#[derive(Debug, Clone, Copy)]
pub struct TQPDimensions {
    pub num_qubits: usize,
    pub num_time_bins: usize,
    pub num_layers: usize,
}

impl TQPDimensions {
    pub fn new(num_qubits: usize, num_time_bins: usize, num_layers: usize) -> Self {
        Self {
            num_qubits,
            num_time_bins,
            num_layers,
        }
    }

    pub fn total_dim(&self) -> usize {
        (1 << self.num_qubits) * self.num_time_bins * self.num_layers
    }

    pub fn spatial_dim(&self) -> usize {
        1 << self.num_qubits
    }
}

/// Represents the state of the Temporal Quantum Processor.
/// The state is a tensor product of Spatial (Qubits) x FastMux (Time Bins) x DeepLogic (Layers).
///
/// Total Dimensions = 2^N (Spatial) * M (FastMux) * L (DeepLogic)
///
/// Index Mapping:
/// global_idx = layer_idx * (num_bins * spatial_dim) + bin_idx * spatial_dim + spatial_idx
#[derive(Debug, Clone)]
pub struct TQPState {
    pub dims: TQPDimensions,
    pub state_vector: Array1<Complex64>,
}

impl TQPState {
    /// Creates a new TQPState initialized to |0...0>|0>|0>
    pub fn new(num_qubits: usize, num_time_bins: usize, num_layers: usize) -> Self {
        let dims = TQPDimensions::new(num_qubits, num_time_bins, num_layers);
        let total_dim = dims.total_dim();
        let mut state_vector = Array1::<Complex64>::zeros(total_dim);
        state_vector[0] = Complex64::new(1.0, 0.0); // Initialize to |0>

        TQPState { dims, state_vector }
    }

    pub fn dimension(&self) -> usize {
        self.state_vector.len()
    }

    pub fn normalize(&mut self) {
        let norm_sq: f64 = self.state_vector.iter().map(|c| c.norm_sqr()).sum();
        if norm_sq > 1e-10 {
            let norm = norm_sq.sqrt();
            self.state_vector.mapv_inplace(|c| c / norm);
        }
    }

    pub fn probability(&self, index: usize) -> f64 {
        if index < self.dimension() {
            self.state_vector[index].norm_sqr()
        } else {
            0.0
        }
    }

    /// Calculates the marginal probability of a specific qubit being in state |1>.
    pub fn get_marginal_probability(&self, qubit_idx: usize) -> f64 {
        let mut prob = 0.0;
        let bit = 1 << qubit_idx;

        for i in 0..self.dimension() {
            let (_, _, s) = self.get_indices(i);
            if (s & bit) != 0 {
                prob += self.probability(i);
            }
        }
        prob
    }

    /// Helper: Calculates the global index from (layer, bin, spatial) indices.
    #[inline]
    pub fn get_index(&self, layer: usize, bin: usize, spatial: usize) -> usize {
        let spatial_dim = 1 << self.dims.num_qubits;
        layer * (self.dims.num_time_bins * spatial_dim) + bin * spatial_dim + spatial
    }

    /// Helper: Decomposes a global index into (layer, bin, spatial) indices.
    #[inline]
    pub fn get_indices(&self, global_idx: usize) -> (usize, usize, usize) {
        let spatial_dim = 1 << self.dims.num_qubits;
        let bin_dim = self.dims.num_time_bins;

        let spatial = global_idx % spatial_dim;
        let remaining = global_idx / spatial_dim;
        let bin = remaining % bin_dim;
        let layer = remaining / bin_dim;

        (layer, bin, spatial)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initialization() {
        let state = TQPState::new(2, 2, 2);
        assert_eq!(state.dimension(), 4 * 2 * 2); // 16
        assert_eq!(state.probability(0), 1.0);
        assert_eq!(state.probability(1), 0.0);
    }

    #[test]
    fn test_indexing() {
        let state = TQPState::new(2, 2, 2); // 2 qubits (dim 4), 2 bins, 2 layers
                                            // Index mapping: layer * 8 + bin * 4 + spatial

        let idx = state.get_index(1, 1, 3); // Layer 1, Bin 1, Spatial 3 (|11>)
                                            // 1 * 8 + 1 * 4 + 3 = 15
        assert_eq!(idx, 15);

        let (l, b, s) = state.get_indices(15);
        assert_eq!((l, b, s), (1, 1, 3));
    }
}

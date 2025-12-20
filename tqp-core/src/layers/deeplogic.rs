/// Represents the DeepLogic layer (Tier 2).
/// Handles logical layering at the microsecond scale.
#[derive(Debug, Clone)]
pub struct DeepLogic {
    pub num_layers: usize,
    pub coherence_time_us: f64,
}

impl DeepLogic {
    pub fn new(num_layers: usize, coherence_time_us: f64) -> Self {
        DeepLogic {
            num_layers,
            coherence_time_us,
        }
    }

    /// Returns the index of the next layer.
    /// Returns None if it's the last layer.
    pub fn next_layer(&self, current_layer: usize) -> Option<usize> {
        if current_layer + 1 < self.num_layers {
            Some(current_layer + 1)
        } else {
            None
        }
    }
}

/// Represents the FastMux layer (Tier 1).
/// Handles time-bin multiplexing at the nanosecond scale.
#[derive(Debug, Clone)]
pub struct FastMux {
    pub num_bins: usize,
    pub bin_duration_ns: f64,
}

impl FastMux {
    pub fn new(num_bins: usize, bin_duration_ns: f64) -> Self {
        FastMux {
            num_bins,
            bin_duration_ns,
        }
    }

    /// Returns the index of the next time bin (cyclic).
    pub fn next_bin(&self, current_bin: usize) -> usize {
        (current_bin + 1) % self.num_bins
    }

    /// Returns the index after a shift of delta bins.
    pub fn shift(&self, current_bin: usize, delta: usize) -> usize {
        (current_bin + delta) % self.num_bins
    }
}

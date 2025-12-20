use crate::state::TQPState;
use num_complex::Complex64;

/// Ghost Path Pruner (Tier 2 Noise Resilience)
/// Detects low-amplitude paths (Ghost Paths) and prunes them to recycle resources.
/// In a simulation, this means zeroing out small amplitudes to reduce computational noise/complexity.
pub struct GhostPathPruner {
    pub amplitude_threshold: f64,
    pub pruned_count: usize,
    pub recycled_energy: f64,
}

impl GhostPathPruner {
    pub fn new(amplitude_threshold: f64) -> Self {
        GhostPathPruner {
            amplitude_threshold,
            pruned_count: 0,
            recycled_energy: 0.0,
        }
    }

    /// Prunes the state by zeroing out amplitudes below the threshold.
    /// Returns the number of pruned paths.
    pub fn prune(&mut self, state: &mut TQPState) -> usize {
        let mut local_pruned = 0;
        let mut local_energy = 0.0;

        state.state_vector.mapv_inplace(|c| {
            // For MVP: identity evolution (no Hamiltonian yet)
            // TODO: Implement actual Hamiltonian
            let prob = c.norm_sqr();
            if prob > 0.0 && prob < self.amplitude_threshold {
                local_pruned += 1;
                local_energy += prob;
                Complex64::new(0.0, 0.0)
            } else {
                c
            }
        });

        self.pruned_count += local_pruned;
        self.recycled_energy += local_energy;

        // Renormalize after pruning to maintain valid quantum state
        state.normalize();

        local_pruned
    }
}

/// Decoherence Channel (T1/T2 Noise)
/// Simulates physical noise using Quantum Trajectories (Monte Carlo Wavefunction).
/// Applies Amplitude Damping (T1) and Phase Damping (T2) to the state vector.
pub struct DecoherenceChannel {
    pub t1: f64, // Relaxation time (ns)
    pub t2: f64, // Dephasing time (ns)
}

impl DecoherenceChannel {
    pub fn new(t1: f64, t2: f64) -> Self {
        DecoherenceChannel { t1, t2 }
    }

    /// Applies noise for a duration `dt` (ns) to all qubits.
    /// Uses Kraus operators and probabilistic jumps.
    pub fn apply_noise(&self, state: &mut TQPState, dt: f64) {
        // Probabilities for infinitesimal time step
        // p_decay = dt / T1
        // p_phase = dt / T2 (approx, technically T2* includes T1 effects, but treating separate for MVP)

        let p_decay = if self.t1 > 0.0 { dt / self.t1 } else { 0.0 };
        let p_phase = if self.t2 > 0.0 { dt / self.t2 } else { 0.0 };

        for q in 0..state.dims.num_qubits {
            self.apply_amplitude_damping(state, q, p_decay);
            self.apply_phase_damping(state, q, p_phase);
        }
    }

    fn apply_amplitude_damping(&self, state: &mut TQPState, qubit_idx: usize, p: f64) {
        if p <= 0.0 {
            return;
        }

        // Calculate probability of jump (decay)
        let prob_1 = state.get_marginal_probability(qubit_idx);
        let p_jump = p * prob_1;

        let r: f64 = rand::random();
        let bit = 1 << qubit_idx;

        if r < p_jump {
            // JUMP: Apply K1 = sqrt(p) |0⟩⟨1|
            let norm_factor = 1.0 / p_jump.sqrt();
            let sqrt_p = p.sqrt();
            let factor = sqrt_p * norm_factor;

            // Optimization: Create new zero vector and populate only from |1⟩ states mapping to |0⟩
            let mut result_vector = ndarray::Array1::<Complex64>::zeros(state.dimension());

            for i in 0..state.dimension() {
                let (_, _, s) = state.get_indices(i);
                if (s & bit) != 0 {
                    // This is a |1⟩ state. Map to |0⟩ state.
                    let target_i = i - bit;
                    result_vector[target_i] = state.state_vector[i] * factor;
                }
            }
            state.state_vector = result_vector;
        } else {
            // NO JUMP: Apply K0 = |0⟩⟨0| + sqrt(1-p)|1⟩⟨1|
            let norm_factor = 1.0 / (1.0 - p_jump).sqrt();
            let sqrt_1_minus_p = (1.0 - p).sqrt();

            for i in 0..state.dimension() {
                let (_, _, s) = state.get_indices(i);
                if (s & bit) != 0 {
                    // |1⟩ state: scale by sqrt(1-p)
                    state.state_vector[i] = state.state_vector[i] * sqrt_1_minus_p * norm_factor;
                } else {
                    // |0⟩ state: scale by 1
                    state.state_vector[i] *= norm_factor;
                }
            }
        }
    }

    fn apply_phase_damping(&self, state: &mut TQPState, qubit_idx: usize, p: f64) {
        if p <= 0.0 {
            return;
        }

        // Phase damping (Dephasing)
        // K0 = sqrt(1-p) I, K1 = sqrt(p) Z

        let r: f64 = rand::random();

        if r < p {
            // Apply Z
            let bit = 1 << qubit_idx;
            for i in 0..state.dimension() {
                let (_, _, s) = state.get_indices(i);
                if (s & bit) != 0 {
                    state.state_vector[i] = -state.state_vector[i];
                }
            }
        }
    }
}

/// Crosstalk Noise Model (Spatial + Temporal)
/// Simulates ZZ interaction between adjacent qubits (Spatial) and leakage between time bins (Temporal).
pub struct CrosstalkNoise {
    pub lambda_temporal: Vec<f64>, // Temporal coupling strength per time bin index
    pub coupling_map: Vec<(usize, usize)>, // Spatial connectivity graph (qubit_i, qubit_j)
    pub j_spatial: Vec<f64>,       // Spatial coupling strength J_ij for each pair
}

impl CrosstalkNoise {
    pub fn new(
        num_bins: usize,
        coupling_map: Vec<(usize, usize)>,
        default_j: f64,
        default_lambda: f64,
    ) -> Self {
        CrosstalkNoise {
            lambda_temporal: vec![default_lambda; num_bins],
            coupling_map: coupling_map.clone(),
            j_spatial: vec![default_j; coupling_map.len()],
        }
    }

    /// Returns the crosstalk Hamiltonian term for a specific time step.
    pub fn get_params(&self) -> (f64, f64) {
        // Return average strengths for reporting
        let avg_lambda = if self.lambda_temporal.is_empty() {
            0.0
        } else {
            self.lambda_temporal.iter().sum::<f64>() / self.lambda_temporal.len() as f64
        };
        let avg_j = if self.j_spatial.is_empty() {
            0.0
        } else {
            self.j_spatial.iter().sum::<f64>() / self.j_spatial.len() as f64
        };
        (avg_lambda, avg_j)
    }
}

use crate::state::TQPState;
use ndarray::Array2;
use num_complex::Complex64;

/// Op-1: Intra-Layer Gate (Spatial)
/// Applies a single-qubit gate to a specific qubit at a specific time bin and layer.
pub fn apply_spatial_gate(state: &mut TQPState, qubit_idx: usize, gate: &Array2<Complex64>) {
    // Optimized N-qubit Tensor Product Logic using ndarray reshaping.
    // We view the state vector as a tensor where the target qubit is one dimension.
    // The state vector has size L * M * 2^N.
    // We want to apply the gate to the qubit_idx-th bit of the spatial part.
    // Spatial index s = ... q_k ...
    // q_k is at bit position qubit_idx (0-indexed, LSB).
    // Stride for q_k is 2^qubit_idx.

    // Total dimension D.
    // We can reshape D into (Outer, 2, Inner).
    // Inner size = 2^qubit_idx.
    // Outer size = D / (2 * Inner).
    // The middle dimension (2) corresponds to the target qubit (0 or 1).

    let _num_qubits = state.dims.num_qubits;
    let _num_bins = state.dims.num_time_bins;
    let _num_layers = state.dims.num_layers;
    let total_dim = state.dimension();

    let inner_dim = 1 << qubit_idx;
    let outer_dim = total_dim / (2 * inner_dim);

    // Reshape state vector to (Outer, 2, Inner)
    // Note: to_shape might fail if not contiguous, but TQPState owns a contiguous Array1.
    // We use into_shape to avoid copying if possible, but we need mutable access.
    // Array1 -> Array3

    // We need to be careful with borrowing. We can't easily reshape in place and multiply in place
    // without some temporary buffer or careful iteration because it's a matrix-vector multiplication
    // along an axis.
    // U * |psi>_k
    // |0>_k -> u00|0> + u10|1>
    // |1>_k -> u01|0> + u11|1>

    // Let's use a chunked iteration which is essentially what reshape does but manually to be safe
    // and avoid complex ndarray type wrangling for now, or use `axis_iter_mut`.

    // Actually, reshaping is the most elegant way if it works.
    let mut tensor_view = state
        .state_vector
        .view_mut()
        .into_shape((outer_dim, 2, inner_dim))
        .expect("Failed to reshape state vector");

    let u00 = gate[[0, 0]];
    let u01 = gate[[0, 1]];
    let u10 = gate[[1, 0]];
    let u11 = gate[[1, 1]];

    // Iterate over Outer and Inner dimensions
    // We can use Zip to iterate efficiently
    // But a simple nested loop over axis 0 and 2 is fine for MVP optimization.

    for mut chunk in tensor_view.outer_iter_mut() {
        for i in 0..inner_dim {
            let alpha = chunk[[0, i]];
            let beta = chunk[[1, i]];

            chunk[[0, i]] = u00 * alpha + u01 * beta;
            chunk[[1, i]] = u10 * alpha + u11 * beta;
        }
    }
}

/// Op-1b: Intra-Layer 2-Qubit Gate (Spatial)
/// Applies a two-qubit gate to two specific qubits.
/// Currently simplified to assume adjacent or handled by swap (but we implement general logic).
/// For MVP Phase 3, we'll implement a general 2-qubit gate logic.
pub fn apply_spatial_gate_2q(
    state: &mut TQPState,
    qubit1: usize,
    qubit2: usize,
    gate: &Array2<Complex64>,
) {
    // General N-qubit logic for 2-qubit gate is complex with reshaping if qubits are not adjacent.
    // We can use the iterative approach for general case.
    // Iterating over all basis states and applying the 4x4 matrix to the target pair.

    let num_qubits = state.dims.num_qubits;
    let _dim = state.dimension();
    let spatial_dim = 1 << num_qubits;

    let bit1 = 1 << qubit1;
    let bit2 = 1 << qubit2;

    // We iterate over all indices.
    // But we can optimize by iterating over "other" qubits.
    // Number of pairs of pairs = D / 4.

    // Let's use the naive iteration over full dimension for simplicity of implementation in MVP,
    // but skip indices that don't match the "00" pattern of targets to avoid double counting.
    // Actually, we need to find the 4 indices for each "rest" configuration.

    // Mask for other qubits
    let _mask = !(bit1 | bit2);

    // We iterate i from 0 to spatial_dim.
    // If i has 0 at bit1 and 0 at bit2, it is a base index.
    // Then we form i00, i01, i10, i11.

    // To iterate efficiently:
    // We can iterate 'k' from 0 to spatial_dim/4.
    // We insert 0s at qubit1 and qubit2 positions.

    // Helper to insert bits:
    // If we have k, and we want to insert 0 at p1 and p2 (p1 < p2).
    // Split k into 3 parts: high, mid, low.

    let (p1, p2) = if qubit1 < qubit2 {
        (qubit1, qubit2)
    } else {
        (qubit2, qubit1)
    };

    // Iterate over layers and bins
    let num_bins = state.dims.num_time_bins;
    let num_layers = state.dims.num_layers;

    for l in 0..num_layers {
        for m in 0..num_bins {
            for k in 0..(spatial_dim >> 2) {
                // Insert 0 at p1:
                // k = high_mid | low
                // k_p1 = (high_mid << 1) | low
                // Insert 0 at p2:
                // k_p1_p2 = (high << 1) | mid_low

                // Let's do it in two steps
                let mask1 = (1 << p1) - 1;
                let low = k & mask1;
                let high_mid = k >> p1;
                let k_p1 = (high_mid << (p1 + 1)) | low;

                let mask2 = (1 << p2) - 1;
                let low2 = k_p1 & mask2;
                let high2 = k_p1 >> p2;
                let idx00_spatial = (high2 << (p2 + 1)) | low2;

                let _idx01_spatial = idx00_spatial | bit1; // This logic depends on which is which
                                                           // Wait, idx00 is 0 at p1 and 0 at p2.
                                                           // We want basis states |00>, |01>, |10>, |11> w.r.t (qubit1, qubit2).
                                                           // If qubit1 is target 0 and qubit2 is target 1 (in gate matrix convention).

                let idx00_s = idx00_spatial;
                let idx01_s = idx00_spatial | (1 << qubit2); // |0>_1 |1>_2
                let idx10_s = idx00_spatial | (1 << qubit1); // |1>_1 |0>_2
                let idx11_s = idx00_spatial | (1 << qubit1) | (1 << qubit2); // |1>_1 |1>_2

                // Note: The gate matrix is usually defined as |00>, |01>, |10>, |11> where first qubit is control/first.
                // We need to match this.
                // Let's assume gate is ordered as q1, q2.
                // So |0>_q1 |0>_q2 is index 0 of gate.
                // |0>_q1 |1>_q2 is index 1.

                let i00 = state.get_index(l, m, idx00_s);
                let i01 = state.get_index(l, m, idx01_s);
                let i10 = state.get_index(l, m, idx10_s);
                let i11 = state.get_index(l, m, idx11_s);

                let v00 = state.state_vector[i00];
                let v01 = state.state_vector[i01];
                let v10 = state.state_vector[i10];
                let v11 = state.state_vector[i11];
                // Wait, if we pass (q1, q2), we expect gate to act on |q1 q2>.
                // My idx construction:
                // idx01_s sets bit at qubit2. So it is |0>_q1 |1>_q2.
                // idx10_s sets bit at qubit1. So it is |1>_q1 |0>_q2.
                // This matches standard tensor order if q1 is "more significant" or "first"?
                // Usually |q1 q2> = |q1> (x) |q2>.
                // If q1 is index 0 (LSB), then |10> means q1=0, q2=1? No, usually binary string.
                // Let's assume standard quantum convention: |q_n ... q_0>.
                // But here we specify q1, q2 explicitly.
                // Let's map 0->00, 1->01, 2->10, 3->11 based on (q1, q2) state.
                // So v01 corresponds to q1=0, q2=1.
                // v10 corresponds to q1=1, q2=0.

                // Apply 4x4 matrix
                // [v00']   [u00 u01 u02 u03] [v00]
                // [v01'] = [u10 ...        ] [v01]
                // [v10']   [...            ] [v10]
                // [v11']   [...         u33] [v11]

                let mut res = [Complex64::default(); 4];
                let src = [v00, v01, v10, v11];

                for r in 0..4 {
                    for c in 0..4 {
                        res[r] += gate[[r, c]] * src[c];
                    }
                }

                state.state_vector[i00] = res[0];
                state.state_vector[i01] = res[1];
                state.state_vector[i10] = res[2];
                state.state_vector[i11] = res[3];
            }
        }
    }
}

/// Op-2: Fast-Mux Shift
/// Shifts the quantum state across time bins.
/// |τ_m⟩ -> |τ_{(m+Δ) mod M}⟩
pub fn fast_mux_shift(state: &mut TQPState, delta: usize) {
    let num_bins = state.dims.num_time_bins;
    let spatial_dim = 1 << state.dims.num_qubits;
    let layer_dim = state.dims.num_layers;

    let mut new_vector = state.state_vector.clone();

    // Iterate over all layers
    for l in 0..layer_dim {
        // Iterate over all time bins
        for m in 0..num_bins {
            let target_m = (m + delta) % num_bins;

            // For each spatial basis state
            for s in 0..spatial_dim {
                let src_idx = state.get_index(l, m, s);
                let dst_idx = state.get_index(l, target_m, s);

                new_vector[dst_idx] = state.state_vector[src_idx];
            }
        }
    }
    state.state_vector = new_vector;
}

/// Op-3: Deep-Logic Shift
/// Shifts the quantum state across logical layers.
/// |λ_l⟩ -> |λ_{l+Δ}⟩ (with boundary check or cyclic for MVP)
pub fn deep_logic_shift(state: &mut TQPState, delta: usize) {
    let num_bins = state.dims.num_time_bins;
    let spatial_dim = 1 << state.dims.num_qubits;
    let layer_dim = state.dims.num_layers;

    let mut new_vector = state.state_vector.clone();

    for l in 0..layer_dim {
        let target_l = (l + delta) % layer_dim; // Cyclic for MVP simplicity

        for m in 0..num_bins {
            for s in 0..spatial_dim {
                let src_idx = state.get_index(l, m, s);
                let dst_idx = state.get_index(target_l, m, s);

                new_vector[dst_idx] = state.state_vector[src_idx];
            }
        }
    }
    state.state_vector = new_vector;
}

/// Op-4: Temporal Entangle
/// Entangles a qubit across two different time bins.
/// This is the core "Time as a Resource" operator.
/// E_time |q_i⟩(|τ_m1⟩ + |τ_m2⟩) -> Entangled State
pub fn temporal_entangle(state: &mut TQPState, _qubit_idx: usize, bin1: usize, bin2: usize) {
    // For MVP: Apply a Hadamard-like operation between bin1 and bin2 for the specific qubit.
    // If qubit is in state |0> at bin1, it becomes (|0, bin1> + |0, bin2>) / sqrt(2)

    let spatial_dim = 1 << state.dims.num_qubits;
    let layer_dim = state.dims.num_layers;
    let sqrt2 = 2.0_f64.sqrt();

    // We only want to entangle if the qubit is in a specific state?
    // Or is it a global operation on the time bins conditioned on the qubit?
    // Definition: E_time |q_i⟩(|τ_{m₁}⟩ + |τ_{m₂}⟩)|λ_l⟩
    // It mixes the time bins.

    // Simplified interpretation for MVP:
    // For every spatial state s, if s has qubit_idx, we mix the amplitudes of (bin1, s) and (bin2, s).
    // Actually, usually entanglement implies CNOT-like behavior.
    // Let's implement a "Time-Splitter" (Hadamard on Time) for all spatial states for now.

    for l in 0..layer_dim {
        for s in 0..spatial_dim {
            let idx1 = state.get_index(l, bin1, s);
            let idx2 = state.get_index(l, bin2, s);

            let val1 = state.state_vector[idx1];
            let val2 = state.state_vector[idx2];

            // Apply H-like mixing
            state.state_vector[idx1] = (val1 + val2) / sqrt2;
            state.state_vector[idx2] = (val1 - val2) / sqrt2;
        }
    }
}

/// Op-5: Measure
/// Measures the state and returns the collapsed state index.
pub fn measure(state: &mut TQPState) -> usize {
    let r: f64 = rand::random();
    let mut cumulative_prob = 0.0;

    for i in 0..state.dimension() {
        cumulative_prob += state.probability(i);
        if r <= cumulative_prob {
            // Collapse state (simplified: just return index, don't update vector for now)
            return i;
        }
    }
    state.dimension() - 1
}

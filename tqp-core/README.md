# tqp-core

The core simulation engine for TQP, written in Rust.

## Features

- **State Vector Simulation:** Supports arbitrary qubit ($N$), time-bin ($M$), and layer ($L$) dimensions.
- **Temporal OS Integration:**
  - `AdaptiveScheduler`: Optimizes gate execution order.
  - `MemoryManager`: Manages state memory allocation.
- **Noise Models:**
  - `DecoherenceChannel`: T1/T2 relaxation and dephasing.
  - `CrosstalkNoise`: Spatial crosstalk simulation.
- **Pulse Solver:** Hamiltonian simulation using RK4 integration.

## Usage

```rust
use tqp_core::state::TQPState;
use tqp_core::ops;
use num_complex::Complex64;

fn main() {
    // Initialize state: 2 qubits, 1 time-bin, 1 layer
    let mut state = TQPState::new(2, 1, 1);
    
    // Apply Hadamard to qubit 0
    // ... gate definition ...
    // ops::apply_spatial_gate(&mut state, 0, &h_gate);
    
    // Measure
    let result = ops::measure(&mut state);
    println!("Result: {}", result);
}
```

import tqp
import math

def test_tqp_bindings():
    print("Testing TQP Python Bindings...")
    
    # 1. Initialize State: 1 Qubit, 2 Time Bins, 1 Layer
    state = tqp.PyTQPState(1, 2, 1)
    print(f"State Dimension: {state.dimension()}")
    
    # 2. Apply X Gate to Qubit 0
    # |0> -> |1>
    tqp.PyOps.apply_x(state, 0)
    print("Applied X Gate.")
    
    # 3. Check Probabilities
    # Index 0 (|0, bin0>) should be 0
    # Index 1 (|1, bin0>) should be 1 (assuming spatial dim=2, bin0=0)
    # Wait, indexing: layer * (bins * spatial) + bin * spatial + spatial_idx
    # Bin 0, Spatial 0 (|0>) -> Index 0
    # Bin 0, Spatial 1 (|1>) -> Index 1
    
    prob0 = state.probability(0)
    prob1 = state.probability(1)
    print(f"Prob(0): {prob0}")
    print(f"Prob(1): {prob1}")
    
    if abs(prob1 - 1.0) < 1e-6:
        print("PASS: X Gate applied correctly.")
    else:
        print("FAIL: X Gate failed.")
        
    # 4. Temporal Entangle
    # Reset state to |0>
    state = tqp.PyTQPState(1, 2, 1)
    tqp.PyOps.temporal_entangle(state, 0, 0, 1)
    print("Applied Temporal Entangle (Bin 0, Bin 1).")
    
    # Bin 0, Spatial 0 -> Index 0
    # Bin 1, Spatial 0 -> Index 2
    prob_bin0 = state.probability(0)
    prob_bin1 = state.probability(2)
    
    print(f"Prob(Bin0): {prob_bin0}")
    print(f"Prob(Bin1): {prob_bin1}")
    
    if abs(prob_bin0 - 0.5) < 1e-6 and abs(prob_bin1 - 0.5) < 1e-6:
        print("PASS: Temporal Entanglement successful.")
    else:
        print("FAIL: Temporal Entanglement failed.")

if __name__ == "__main__":
    try:
        test_tqp_bindings()
    except ImportError:
        print("Error: 'tqp' module not found. Make sure to build with 'maturin develop' or link the .so/.pyd file.")

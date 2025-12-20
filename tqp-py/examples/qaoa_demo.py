import tqp
import math
import cmath

# Simple MaxCut QAOA Demo on a 3-node linear graph: 0-1-2
# Edges: (0,1), (1,2)
# MaxCut: Cut both edges -> 0(A), 1(B), 2(A) -> 101 or 010.
# Hamiltonian H_C = 0.5 * sum_{(i,j)} (1 - Z_i Z_j)
# We want to maximize <H_C>, or minimize <Z_i Z_j>.

def rzz_gate(theta):
    # Rzz(theta) = exp(-i * theta/2 * Z \otimes Z)
    # Diag: [e^{-i theta/2}, e^{i theta/2}, e^{i theta/2}, e^{-i theta/2}]
    val1 = cmath.exp(-1j * theta / 2)
    val2 = cmath.exp(1j * theta / 2)
    
    return [
        [val1, 0, 0, 0],
        [0, val2, 0, 0],
        [0, 0, val2, 0],
        [0, 0, 0, val1]
    ]

def rx_gate(theta):
    # Rx(theta) = exp(-i * theta/2 * X)
    # [cos(theta/2), -i sin(theta/2)]
    # [-i sin(theta/2), cos(theta/2)]
    c = math.cos(theta / 2)
    s = -1j * math.sin(theta / 2)
    return [
        [c, s],
        [s, c]
    ]

def qaoa_demo():
    print("Starting QAOA Demo (MaxCut on 3-node linear graph)...")
    
    # Parameters
    p = 1 # Depth
    gamma = 2.0 # Initial guess
    beta = 1.0 # Initial guess
    steps = 10
    lr = 0.1
    history = []
    
    # Edges
    edges = [(0, 1), (1, 2)]
    
    # Initialize State: 3 Qubits, 1 Bin, 1 Layer
    # |+++>
    state = tqp.PyTQPState(3, 1, 1)
    
    # Apply Hadamard to all
    h_gate = [
        [complex(1/math.sqrt(2), 0), complex(1/math.sqrt(2), 0)],
        [complex(1/math.sqrt(2), 0), complex(-1/math.sqrt(2), 0)]
    ]
    
    for i in range(3):
        tqp.PyOps.apply_spatial_gate(state, i, h_gate)
        
    print("State Initialized to |+++>.")
    
    # Optimization Loop (Simplified Gradient Descent)
    for step in range(steps):
        # 1. Re-initialize state
        state = tqp.PyTQPState(3, 1, 1)
        for i in range(3):
            tqp.PyOps.apply_spatial_gate(state, i, h_gate)
            
        # 2. Apply QAOA Ansatz
        # Layer 1: U_C(gamma) = prod e^{-i gamma H_ij}
        # H_ij = (1 - Z_i Z_j)/2. Constant factor doesn't matter for state evolution, only phase.
        # We implement e^{-i gamma (-Z_i Z_j)} = e^{i gamma Z_i Z_j}.
        # This corresponds to Rzz(-2 * gamma).
        
        for (u, v) in edges:
            tqp.PyOps.apply_spatial_gate_2q(state, u, v, rzz_gate(-2.0 * gamma))
            
        # Layer 2: U_B(beta) = prod e^{-i beta X_i} = Rx(2 * beta)
        for i in range(3):
            tqp.PyOps.apply_spatial_gate(state, i, rx_gate(2.0 * beta))
            
        # 3. Calculate Expectation Value <H_C>
        # H_C = sum (1 - Z_i Z_j)/2
        # We want to MAXIMIZE this.
        # <H_C> = sum (1 - <Z_i Z_j>)/2
        # We need <Z_i Z_j>.
        # Our `expval_z` gives <Z_i>.
        # We don't have <Z_i Z_j> directly exposed.
        # But we can measure probability distribution and calculate it.
        
        # Calculate <Z_0 Z_1> and <Z_1 Z_2> from probabilities
        dim = state.dimension()
        exp_z0z1 = 0.0
        exp_z1z2 = 0.0
        
        for idx in range(dim):
            prob = state.probability(idx)
            # idx maps to spatial state s (since 1 bin, 1 layer)
            # s = q2 q1 q0
            s = idx 
            q0 = (s >> 0) & 1
            q1 = (s >> 1) & 1
            q2 = (s >> 2) & 1
            
            # Z eigenvalue is +1 if 0, -1 if 1.
            z0 = 1 if q0 == 0 else -1
            z1 = 1 if q1 == 0 else -1
            z2 = 1 if q2 == 0 else -1
            
            exp_z0z1 += prob * (z0 * z1)
            exp_z1z2 += prob * (z1 * z2)
            
        cost = 0.5 * ((1 - exp_z0z1) + (1 - exp_z1z2))
        print(f"Step {step}: Cost = {cost:.4f}, Gamma = {gamma:.2f}, Beta = {beta:.2f}")
        
        history.append({
            "step": step,
            "cost": cost,
            "gamma": gamma,
            "beta": beta
        })
        
        # Simple update rule (Gradient free / manual for demo)
        gamma += 0.01
        beta -= 0.01

    print("QAOA Demo Completed.")
    
    # Export data for visualization
    import json
    with open("qaoa_results.json", "w") as f:
        json.dump(history, f, indent=2)
    print("Results saved to qaoa_results.json")

if __name__ == "__main__":
    qaoa_demo()

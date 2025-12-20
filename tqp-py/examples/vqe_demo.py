import tqp
import numpy as np
import math

def vqe_demo():
    print("=== TQP VQE Demo: H2 Molecule (Simplified) ===")
    
    # 1. Define Hamiltonian (H2 at bond distance 0.74A, mapped to 2 qubits via Jordan-Wigner)
    # Simplified Hamiltonian: H = g0*I + g1*Z0 + g2*Z1 + g3*Z0Z1 + g4*X0X1 + g5*Y0Y1
    # Coefficients (approximate for demo):
    g0 = -0.4804
    g1 = 0.3435
    g2 = -0.4347
    g3 = 0.5716
    g4 = 0.0910
    g5 = 0.0910

    # 2. Define Ansatz (Hardware Efficient / RyRz)
    def ansatz(theta, state):
        # Layer 1: Ry rotations
        # Ry(theta) = [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]
        
        def ry_matrix(angle):
            c = math.cos(angle / 2)
            s = math.sin(angle / 2)
            return [[complex(c, 0), complex(-s, 0)], [complex(s, 0), complex(c, 0)]]

        tqp.PyOps.apply_spatial_gate(state, 0, ry_matrix(theta[0]))
        tqp.PyOps.apply_spatial_gate(state, 1, ry_matrix(theta[1]))

        # Layer 2: CNOT (Entanglement)
        # CNOT 0->1
        cnot = [
            [complex(1,0), complex(0,0), complex(0,0), complex(0,0)],
            [complex(0,0), complex(1,0), complex(0,0), complex(0,0)],
            [complex(0,0), complex(0,0), complex(0,0), complex(1,0)],
            [complex(0,0), complex(0,0), complex(1,0), complex(0,0)]
        ]
        tqp.PyOps.apply_spatial_gate_2q(state, 0, 1, cnot)

        # Layer 3: Ry rotations
        tqp.PyOps.apply_spatial_gate(state, 0, ry_matrix(theta[2]))
        tqp.PyOps.apply_spatial_gate(state, 1, ry_matrix(theta[3]))

    # 3. Expectation Value Calculation
    def calculate_energy(theta):
        state = tqp.PyTQPState(2, 1, 1) # 2 qubits, 1 bin, 1 layer
        ansatz(theta, state)
        
        # Calculate <H> = sum(gi * <Pi>)
        # Note: tqp currently only supports Z expectation easily via expval_z.
        # For X and Y terms, we need basis rotation.
        # <X> = <H Z H>
        # <Y> = <Sdag H Z H S>
        
        # For MVP, let's approximate or implement full expval in Rust later.
        # Here we use a simplified cost function just using Z terms for demonstration
        # if full Pauli measurement isn't ready.
        # BUT, let's try to do it right with what we have.
        
        # Term Z0:
        z0 = state.expval_z(0)
        
        # Term Z1:
        z1 = state.expval_z(1)
        
        # Term Z0Z1:
        # We need correlation <Z0Z1>. 
        # P(00) + P(11) - P(01) - P(10)
        # We can get probabilities.
        p0 = state.probability(0) # |00>
        p1 = state.probability(1) # |01>
        p2 = state.probability(2) # |10>
        p3 = state.probability(3) # |11>
        z0z1 = (p0 + p3) - (p1 + p2)
        
        # For X0X1 and Y0Y1, we need to rotate basis and measure.
        # Since we can't clone state easily in this loop without cost, 
        # we would typically re-prepare state.
        # For this demo, we will ignore off-diagonal terms or assume they are small
        # OR we re-run ansatz for each basis.
        
        # Let's re-run for X basis
        state_x = tqp.PyTQPState(2, 1, 1)
        ansatz(theta, state_x)
        # Apply H to both to measure X
        h_gate = [[complex(0.707,0), complex(0.707,0)], [complex(0.707,0), complex(-0.707,0)]]
        tqp.PyOps.apply_spatial_gate(state_x, 0, h_gate)
        tqp.PyOps.apply_spatial_gate(state_x, 1, h_gate)
        
        px0 = state_x.probability(0)
        px1 = state_x.probability(1)
        px2 = state_x.probability(2)
        px3 = state_x.probability(3)
        x0x1 = (px0 + px3) - (px1 + px2)

        # Let's re-run for Y basis
        state_y = tqp.PyTQPState(2, 1, 1)
        ansatz(theta, state_y)
        # Apply Sdag H to measure Y. Sdag = [[1,0],[0,-i]]
        # H = [[1,1],[1,-1]]/sqrt(2)
        # Sdag H = [[1,1], [-i, i]] / sqrt(2)
        sdagh = [
            [complex(0.707,0), complex(0.707,0)], 
            [complex(0, -0.707), complex(0, 0.707)]
        ]
        tqp.PyOps.apply_spatial_gate(state_y, 0, sdagh)
        tqp.PyOps.apply_spatial_gate(state_y, 1, sdagh)
        
        py0 = state_y.probability(0)
        py1 = state_y.probability(1)
        py2 = state_y.probability(2)
        py3 = state_y.probability(3)
        y0y1 = (py0 + py3) - (py1 + py2)

        energy = g0 + g1*z0 + g2*z1 + g3*z0z1 + g4*x0x1 + g5*y0y1
        return energy

    # 4. Optimization Loop (Simple Gradient Descent)
    theta = [0.1, 0.1, 0.1, 0.1] # Initial params
    lr = 0.1
    steps = 50
    
    print(f"Initial Energy: {calculate_energy(theta):.4f}")
    
    for i in range(steps):
        # Finite difference gradient
        grad = []
        current_e = calculate_energy(theta)
        
        for j in range(len(theta)):
            temp_theta = theta[:]
            temp_theta[j] += 0.01
            e_plus = calculate_energy(temp_theta)
            g = (e_plus - current_e) / 0.01
            grad.append(g)
            
        # Update
        for j in range(len(theta)):
            theta[j] -= lr * grad[j]
            
        if i % 10 == 0:
            print(f"Step {i}: Energy = {current_e:.4f}")
            
    print(f"Final Energy: {calculate_energy(theta):.4f}")
    print(f"Optimal Params: {theta}")

if __name__ == "__main__":
    vqe_demo()

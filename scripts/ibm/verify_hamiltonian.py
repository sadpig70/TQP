"""
BeH₂ 해밀토니안 검증 스크립트

로컬 시뮬레이션으로 HF 상태의 이론적 에너지를 계산하여
하드웨어 결과와 비교합니다.
"""

import numpy as np

def calculate_hf_energy_theory():
    """HF 상태 |000011⟩에서 이론적 에너지 계산"""
    
    # HF 상태: q0=1, q1=1, q2=0, q3=0, q4=0, q5=0
    # Z eigenvalue: |0⟩ → +1, |1⟩ → -1
    # 
    # Qiskit 규칙: Pauli string은 little-endian
    # "IIIIIZ" = Z on qubit 0 (rightmost)
    # "IIIIZI" = Z on qubit 1
    # "ZIIIII" = Z on qubit 5 (leftmost)
    
    # HF 상태 Z expectation values
    z_expectations = {
        0: -1,  # q0 = |1⟩
        1: -1,  # q1 = |1⟩
        2: +1,  # q2 = |0⟩
        3: +1,  # q3 = |0⟩
        4: +1,  # q4 = |0⟩
        5: +1,  # q5 = |0⟩
    }
    
    # 해밀토니안 항목 (Qiskit little-endian)
    # "IIIIIZ" means Z_0, "ZIIIII" means Z_5
    pauli_terms = [
        ("IIIIII", -15.4947),  # Identity
        ("IIIIIZ", 0.1562),    # Z0 → -1
        ("IIIIZI", 0.1562),    # Z1 → -1
        ("IIIIZZ", 0.0782),    # Z0Z1 → (-1)*(-1) = +1
        ("IIIZII", -0.1089),   # Z2 → +1
        ("IIZIII", -0.1089),   # Z3 → +1
        ("IZIIII", -0.0534),   # Z4 → +1
        ("ZIIIII", -0.0534),   # Z5 → +1
        ("IIIZIZ", 0.0423),    # Z0Z2 → (-1)*(+1) = -1
        ("IIIZZI", 0.0423),    # Z1Z2 → (-1)*(+1) = -1
        ("IIZIZI", 0.0512),    # Z1Z3 → (-1)*(+1) = -1
        ("IIZIIZ", 0.0512),    # Z0Z3 → (-1)*(+1) = -1
        ("IIZZII", 0.0389),    # Z2Z3 → (+1)*(+1) = +1
        ("IZIIZI", 0.0267),    # Z1Z4 → (-1)*(+1) = -1
        ("IZIIIZ", 0.0267),    # Z0Z4 → (-1)*(+1) = -1
        ("IZIZII", 0.0334),    # Z2Z4 → (+1)*(+1) = +1
        ("IZZIII", 0.0334),    # Z3Z4 → (+1)*(+1) = +1
        ("ZIIZII", 0.0267),    # Z2Z5 → (+1)*(+1) = +1
        ("ZIIIZI", 0.0267),    # Z1Z5 → (-1)*(+1) = -1
        ("ZIIIIZ", 0.0312),    # Z0Z5 → (-1)*(+1) = -1
        ("ZIZIII", 0.0334),    # Z3Z5 → (+1)*(+1) = +1
        ("ZZIIII", 0.0223),    # Z4Z5 → (+1)*(+1) = +1
    ]
    
    def get_expectation(pauli_string):
        """Pauli string의 expectation value 계산"""
        result = 1.0
        for i, p in enumerate(reversed(pauli_string)):  # little-endian
            if p == 'Z':
                result *= z_expectations[i]
            # I는 +1이므로 무시
        return result
    
    # 에너지 계산
    energy = 0.0
    print("=" * 60)
    print("HF 상태 |000011⟩에서 해밀토니안 검증")
    print("=" * 60)
    print(f"\n{'Pauli':<10} {'계수':>10} {'<Pauli>':>10} {'기여':>12}")
    print("-" * 45)
    
    for pauli, coeff in pauli_terms:
        exp_val = get_expectation(pauli)
        contribution = coeff * exp_val
        energy += contribution
        print(f"{pauli:<10} {coeff:>10.4f} {exp_val:>10.0f} {contribution:>12.4f}")
    
    print("-" * 45)
    print(f"{'합계':<10} {'':<10} {'':<10} {energy:>12.4f} Ha")
    
    return energy


def verify_with_qiskit_simulation():
    """Qiskit 시뮬레이터로 검증"""
    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import SparsePauliOp, Statevector
        
        print("\n" + "=" * 60)
        print("Qiskit 시뮬레이션 검증")
        print("=" * 60)
        
        # HF 회로
        qc = QuantumCircuit(6)
        qc.x(0)
        qc.x(1)
        
        # 상태벡터
        sv = Statevector.from_instruction(qc)
        
        # 해밀토니안
        pauli_terms = [
            ("IIIIII", -15.4947),
            ("IIIIIZ", 0.1562),
            ("IIIIZI", 0.1562),
            ("IIIIZZ", 0.0782),
            ("IIIZII", -0.1089),
            ("IIZIII", -0.1089),
            ("IZIIII", -0.0534),
            ("ZIIIII", -0.0534),
            ("IIIZIZ", 0.0423),
            ("IIIZZI", 0.0423),
            ("IIZIZI", 0.0512),
            ("IIZIIZ", 0.0512),
            ("IIZZII", 0.0389),
            ("IZIIZI", 0.0267),
            ("IZIIIZ", 0.0267),
            ("IZIZII", 0.0334),
            ("IZZIII", 0.0334),
            ("ZIIZII", 0.0267),
            ("ZIIIZI", 0.0267),
            ("ZIIIIZ", 0.0312),
            ("ZIZIII", 0.0334),
            ("ZZIIII", 0.0223),
        ]
        
        hamiltonian = SparsePauliOp.from_list(pauli_terms)
        
        # 기댓값 계산
        energy = sv.expectation_value(hamiltonian).real
        
        print(f"\n시뮬레이션 에너지: {energy:.6f} Ha")
        print(f"HF 참조값:        -15.5614 Ha")
        print(f"차이:             {(energy - (-15.5614))*1000:.2f} mHa")
        
        return energy
        
    except ImportError as e:
        print(f"Qiskit 없음: {e}")
        return None


if __name__ == "__main__":
    # 이론적 계산
    theory_energy = calculate_hf_energy_theory()
    
    print(f"\n이론적 HF 에너지: {theory_energy:.4f} Ha")
    print(f"문헌 HF 참조값:   -15.5614 Ha")
    print(f"차이:             {(theory_energy - (-15.5614))*1000:.2f} mHa")
    
    # Qiskit 시뮬레이션
    sim_energy = verify_with_qiskit_simulation()

"""
검증된 2-qubit H₂ 해밀토니안 테스트

문헌 출처: arXiv (STO-3G, R=0.735Å)
H = -1.052373*II - 0.397937*ZI - 0.397937*IZ + 0.011280*ZZ + 0.180931*XX

이 해밀토니안은 Bravyi-Kitaev 또는 Parity 변환을 사용한
축소된 2-qubit 표현입니다.
"""

from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit import QuantumCircuit

print("=" * 50)
print("검증된 2-qubit H₂ 해밀토니안")
print("=" * 50)

# 문헌에서 가져온 정확한 계수
h2_2qubit_terms = [
    ("II", -1.052373),
    ("IZ", -0.397937),
    ("ZI", -0.397937),
    ("ZZ", +0.011280),
    ("XX", +0.180931),
]

H = SparsePauliOp.from_list(h2_2qubit_terms)

print(f"\n해밀토니안 항 개수: {len(H)}")
print(f"큐비트 수: {H.num_qubits}")

# 다양한 상태 테스트
print("\n" + "=" * 50)
print("다양한 상태 테스트")
print("=" * 50)

# |00>, |01>, |10>, |11>
for pattern in ["00", "01", "10", "11"]:
    qc = QuantumCircuit(2)
    for i, b in enumerate(reversed(pattern)):
        if b == "1":
            qc.x(i)
    sv = Statevector.from_instruction(qc)
    e = sv.expectation_value(H).real
    print(f"|{pattern}⟩: E = {e:.6f} Ha")

# HF 상태는 |01⟩ 또는 |10⟩ (1전자 점유)
# 하지만 H₂는 2전자이므로 싱글릿 상태가 필요

print("\n" + "=" * 50)
print("진정한 ground state를 위한 VQE 필요")
print("=" * 50)

# Ground state는 XX 항 때문에 computational basis가 아님
# 정확한 ground state 계산
import numpy as np
from numpy.linalg import eigvalsh

H_matrix = H.to_matrix()
eigenvalues = eigvalsh(H_matrix)
print(f"\n고유값 (에너지 준위):")
for i, e in enumerate(sorted(eigenvalues)):
    print(f"  E_{i} = {e:.6f} Ha")

print(f"\n기저 상태 에너지: {min(eigenvalues):.6f} Ha")
print(f"참조 E_FCI: -1.1373 Ha")
print(f"오차: {(min(eigenvalues) - (-1.1373))*1000:.2f} mHa")

# HF 에너지는 |01⟩ 또는 |10⟩ 상태
qc_hf = QuantumCircuit(2)
qc_hf.x(1)  # |10⟩
sv_hf = Statevector.from_instruction(qc_hf)
e_hf = sv_hf.expectation_value(H).real
print(f"\nHF 상태 |10⟩ 에너지: {e_hf:.6f} Ha")
print(f"참조 E_HF: -1.1173 Ha")
print(f"오차: {(e_hf - (-1.1173))*1000:.2f} mHa")

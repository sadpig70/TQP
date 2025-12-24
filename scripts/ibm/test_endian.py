"""HF 상태 및 해밀토니안 검증 - 간단한 버전"""

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp

print("=" * 50)
print("Qiskit Endianness 테스트")
print("=" * 50)

# HF 상태 |0011⟩ 테스트
qc = QuantumCircuit(4)
qc.x(0)  # q0 = |1>
qc.x(1)  # q1 = |1>
# q2, q3 = |0>

sv = Statevector.from_instruction(qc)

# 상태 확인
print("\n상태벡터 (비영 계수):")
for i, amp in enumerate(sv.data):
    if abs(amp) > 0.01:
        binary = format(i, '04b')
        print(f"  |{binary}> (i={i}): {amp}")
        print(f"  → q3={binary[0]}, q2={binary[1]}, q1={binary[2]}, q0={binary[3]}")

# Z 기댓값 확인 (개별)
print("\n개별 큐빗 Z 기댓값:")
for i in range(4):
    pauli_str = 'I'*i + 'Z' + 'I'*(3-i)
    z_op = SparsePauliOp.from_list([(pauli_str, 1.0)])
    ev = sv.expectation_value(z_op).real
    print(f"  {pauli_str} (Z on q{i}): {ev:+.1f}")

print("\n해석:")
print("  Z expectation value = +1: 큐비트 = |0>")
print("  Z expectation value = -1: 큐비트 = |1>")

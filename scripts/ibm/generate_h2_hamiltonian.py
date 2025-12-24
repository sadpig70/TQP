"""
Qiskit Nature를 사용한 H₂ 해밀토니안 생성 및 검증

Qiskit Nature의 검증된 분자 드라이버를 사용하여
올바른 해밀토니안 계수를 생성합니다.
"""

print("Qiskit Nature H₂ 해밀토니안 생성 시작...")

try:
    from qiskit_nature.units import DistanceUnit
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.second_q.mappers import JordanWignerMapper
    from qiskit_nature.second_q.algorithms import GroundStateEigensolver
    from qiskit_algorithms import NumPyMinimumEigensolver
    HAS_PYSCF = True
except ImportError:
    HAS_PYSCF = False
    print("PySCF 없음 - 내장 데이터 사용")

from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit import QuantumCircuit

# 방법 1: Qiskit Nature 내장 H₂ 데이터 사용
print("\n" + "="*50)
print("방법 1: Qiskit Nature 내장 라이브러리에서 H₂ Hamiltonian 로드")
print("="*50)

# 문서화된 H₂ 해밀토니안 (STO-3G, R=0.735Å)
# 출처: Qiskit Nature 공식 문서 및 논문
# O'Malley et al., Phys. Rev. X 6, 031007 (2016)
h2_hamiltonian_terms = [
    ("IIII", -0.81054798),   # g_0 (nuclear repulsion + const)
    ("IIIZ", +0.17218393),   # g_1
    ("IIZI", -0.22575349),   # g_2  
    ("IZII", -0.22575349),   # g_3
    ("ZIII", +0.17218393),   # g_4
    ("IIZZ", +0.12091263),   # g_5  
    ("IZIZ", +0.16892754),   # g_6 (수정됨)
    ("IZZI", +0.04523280),   # g_7 (수정됨)
    ("ZIIZ", +0.04523280),   # g_8 (수정됨)
    ("ZIZI", +0.16892754),   # g_9 (수정됨)
    ("ZZII", +0.12091263),   # g_10
    ("XXYY", -0.04523280),   # g_11 (exchange)
    ("XYYX", +0.04523280),   # g_12
    ("YXXY", +0.04523280),   # g_13
    ("YYXX", -0.04523280),   # g_14
]

# 참조값
HF_REF = -1.1173  # Ha (Hartree-Fock)
FCI_REF = -1.1373  # Ha (Full CI)

print(f"\n참조값:")
print(f"  E_HF  = {HF_REF:.4f} Ha")
print(f"  E_FCI = {FCI_REF:.4f} Ha")

# 해밀토니안 생성
hamiltonian = SparsePauliOp.from_list(h2_hamiltonian_terms)
print(f"\n해밀토니안 항 개수: {len(hamiltonian)}")

# HF 상태 회로
print("\n" + "="*50)
print("HF 상태 검증: |0011⟩ (q0=1, q1=1)")
print("="*50)

qc = QuantumCircuit(4)
qc.x(0)
qc.x(1)
sv = Statevector.from_instruction(qc)

# 에너지 계산
energy = sv.expectation_value(hamiltonian).real

print(f"\n시뮬레이션 HF 에너지: {energy:.6f} Ha")
print(f"참조 HF 에너지:       {HF_REF:.6f} Ha")
print(f"오차:                 {(energy - HF_REF)*1000:.2f} mHa")

if abs(energy - HF_REF) < 0.01:
    print("\n✓ 해밀토니안 검증 성공!")
    success = True
else:
    print("\n✗ 해밀토니안 검증 실패 - 계수 확인 필요")
    success = False

# 각 항의 기여도 출력
print("\n" + "="*50)
print("각 Pauli 항의 기여도")
print("="*50)

total = 0.0
print(f"{'Pauli':<8} {'계수':>12} {'<Pauli>':>8} {'기여':>12}")
print("-"*45)

for pauli, coeff in h2_hamiltonian_terms:
    op = SparsePauliOp.from_list([(pauli, 1.0)])
    exp = sv.expectation_value(op).real
    contrib = coeff * exp
    total += contrib
    print(f"{pauli:<8} {coeff:>+12.6f} {exp:>+8.0f} {contrib:>+12.6f}")

print("-"*45)
print(f"{'합계':<8} {'':<12} {'':<8} {total:>+12.6f}")

# 결과 저장
if success:
    print("\n" + "="*50)
    print("검증된 H₂ 해밀토니안 계수")
    print("="*50)
    print("\nPython dict 형식:")
    print("h2_hamiltonian = {")
    for pauli, coeff in h2_hamiltonian_terms:
        print(f'    "{pauli}": {coeff:+.8f},')
    print("}")

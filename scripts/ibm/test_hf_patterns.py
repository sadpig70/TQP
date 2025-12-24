"""HF 점유 패턴 테스트 - 올바른 매핑 찾기"""

from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit import QuantumCircuit

# H2 해밀토니안 (O'Malley et al.)
h2_terms = [
    ('IIII', -0.81054798),
    ('IIIZ', +0.17218393),
    ('IIZI', -0.22575349),
    ('IZII', -0.22575349),
    ('ZIII', +0.17218393),
    ('IIZZ', +0.12091263),
    ('IZIZ', +0.16892754),
    ('IZZI', +0.04523280),
    ('ZIIZ', +0.04523280),
    ('ZIZI', +0.16892754),
    ('ZZII', +0.12091263),
    ('XXYY', -0.04523280),
    ('XYYX', +0.04523280),
    ('YXXY', +0.04523280),
    ('YYXX', -0.04523280),
]
H = SparsePauliOp.from_list(h2_terms)

print('다양한 HF 상태 테스트:')
print('='*60)
print(f'참조값: E_HF = -1.1173 Ha')
print('='*60)

# 4가지 점유 패턴 테스트
results = []
for pattern in ['0011', '0110', '1100', '1001', '0101', '1010']:
    qc = QuantumCircuit(4)
    occupied = []
    for i, b in enumerate(reversed(pattern)):
        if b == '1':
            qc.x(i)
            occupied.append(i)
    sv = Statevector.from_instruction(qc)
    e = sv.expectation_value(H).real
    results.append((pattern, occupied, e))
    mark = '✓' if abs(e - (-1.1173)) < 0.02 else ' '
    print(f'{mark} |{pattern}> qc.x({occupied}): E = {e:.6f} Ha, diff = {(e-(-1.1173))*1000:+.1f} mHa')

print()
best = min(results, key=lambda x: abs(x[2] - (-1.1173)))
print(f'가장 근접: |{best[0]}> with E = {best[2]:.6f} Ha')

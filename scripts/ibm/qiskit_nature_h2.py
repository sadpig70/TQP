"""Qiskit Nature 내장 H₂ 예제에서 정확한 해밀토니안 추출"""

print("Qiskit Nature에서 H₂ 해밀토니안 직접 생성...")

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.problems import ElectronicStructureProblem

# PySCF 드라이버 확인
try:
    driver = PySCFDriver(
        atom="H 0 0 0; H 0 0 0.735",
        basis="sto3g",
        charge=0,
        spin=0
    )
    problem = driver.run()
    print("✓ PySCF 드라이버 성공")
    
    # Jordan-Wigner 매핑
    mapper = JordanWignerMapper()
    hamiltonian = mapper.map(problem.hamiltonian.second_q_op())
    
    print(f"\n해밀토니안 항 개수: {len(hamiltonian)}")
    print(f"큐비트 수: {hamiltonian.num_qubits}")
    
    # 계수 출력
    print("\nPauli 항목:")
    for pauli, coeff in sorted(hamiltonian.to_list(), key=lambda x: -abs(x[1])):
        if abs(coeff) > 1e-8:
            print(f"  {pauli}: {coeff.real:+.8f}")
    
    # HF 상태에서 에너지 검증
    from qiskit.quantum_info import Statevector
    from qiskit import QuantumCircuit
    
    # Qiskit Nature의 HF bitstring
    hf_bitstring = problem.reference_energy
    print(f"\nHartree-Fock 에너지 (from problem): {problem.reference_energy:.6f} Ha")
    
    # HF 상태 회로 (2전자 점유)
    qc = QuantumCircuit(4)
    qc.x(0)
    qc.x(1)
    sv = Statevector.from_instruction(qc)
    
    e = sv.expectation_value(hamiltonian).real
    print(f"시뮬레이션 HF 에너지: {e:.6f} Ha")
    
except ImportError as e:
    print(f"✗ PySCF 없음: {e}")
    print("\n대안: qiskit-nature 내장 테스트 데이터 사용")
    
    # 내장 테스트 데이터
    from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
    from qiskit_nature.second_q.operators import ElectronicIntegrals
    from qiskit_nature.second_q.operators.tensor_ordering import to_chemist_ordering
    import numpy as np
    
    print("PySCF 없이 진행 불가. pip install pyscf 필요.")

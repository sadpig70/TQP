"""
BeH₂ 4-qubit 해밀토니안 생성 (CASSCF(2,2) 기반)

PySCF CASSCF(2,2) 결과를 OpenFermion으로 변환하여
4-qubit Pauli 해밀토니안을 생성합니다.

사용법:
    wsl -d Ubuntu -- python3 /mnt/d/SynProject/Engine/QC/tqp/scripts/ibm/generate_beh2_4qubit.py
"""

import json
import numpy as np

def get_beh2_geometry():
    """BeH₂ 선형 분자 기하 구조"""
    return """
    Be  0.0000  0.0000  0.0000
    H   0.0000  0.0000  1.3260
    H   0.0000  0.0000 -1.3260
    """

def run_casscf_and_extract():
    """CASSCF(2,2) 계산 및 해밀토니안 추출"""
    from pyscf import gto, scf, mcscf, ao2mo
    
    print("=" * 60)
    print("BeH₂ 4-qubit Hamiltonian Generation")
    print("=" * 60)
    
    # 분자 정의
    mol = gto.Mole()
    mol.atom = get_beh2_geometry()
    mol.basis = 'sto-3g'
    mol.charge = 0
    mol.spin = 0
    mol.build()
    
    print(f"\nMolecule: BeH₂")
    print(f"Basis: {mol.basis}")
    print(f"Electrons: {mol.nelectron}")
    
    # HF 계산
    mf = scf.RHF(mol)
    mf.verbose = 0
    e_hf = mf.kernel()
    print(f"\nE_HF: {e_hf:.6f} Ha")
    
    # CASSCF(2,2) 계산
    mc = mcscf.CASSCF(mf, 2, 2)
    mc.verbose = 0
    e_casscf = mc.kernel()[0]
    print(f"E_CASSCF(2,2): {e_casscf:.6f} Ha")
    
    # 활성 공간 적분 추출
    n_orb = 2  # 활성 궤도 수
    n_qubits = 2 * n_orb  # 스핀 궤도 = 4 qubits
    
    # 1-전자 적분
    h1_result = mc.get_h1cas()
    h1e = h1_result[0] if isinstance(h1_result, tuple) else h1_result
    
    # 2-전자 적분 (ao2mo로 직접 변환)
    mo_cas = mc.mo_coeff[:, mc.ncore:mc.ncore+n_orb]
    h2e = ao2mo.kernel(mol, mo_cas)
    h2e = ao2mo.restore(1, h2e, n_orb)  # 4D 텐서로 복원
    
    # 핵간 반발 + 코어 에너지
    e_nuc = mol.energy_nuc()
    try:
        e_core = mc.energy_core
    except AttributeError:
        e_cas = mc.e_cas if hasattr(mc, 'e_cas') else 0.0
        e_core = mc.e_tot - e_cas - e_nuc
    
    constant = e_nuc + e_core
    
    print(f"\nExtracted:")
    print(f"  h1e shape: {h1e.shape}")
    print(f"  h2e shape: {h2e.shape}")
    print(f"  Constant (E_nuc + E_core): {constant:.6f} Ha")
    
    return h1e, h2e, constant, e_hf, e_casscf, n_qubits

def jordan_wigner_transform(h1e, h2e, constant, n_qubits):
    """Jordan-Wigner 변환으로 Pauli 해밀토니안 생성"""
    try:
        from openfermion import jordan_wigner
        from openfermion.ops import InteractionOperator
        from openfermion.transforms import get_fermion_operator
        
        n_orb = h1e.shape[0]
        
        # OpenFermion InteractionOperator 생성
        int_op = InteractionOperator(constant, h1e, 0.5 * h2e)
        fermion_op = get_fermion_operator(int_op)
        qubit_op = jordan_wigner(fermion_op)
        
        # Pauli 항 추출
        pauli_terms = {}
        for term, coeff in qubit_op.terms.items():
            if abs(coeff) > 1e-10:
                pauli_str = pauli_term_to_string(term, n_qubits)
                pauli_terms[pauli_str] = float(coeff.real)
        
        print(f"\nJordan-Wigner 변환 완료:")
        print(f"  Total Pauli terms: {len(pauli_terms)}")
        
        return pauli_terms
        
    except ImportError:
        print("Error: OpenFermion이 설치되지 않았습니다.")
        return None

def pauli_term_to_string(term, n_qubits):
    """OpenFermion 항을 Pauli 문자열로 변환"""
    paulis = ['I'] * n_qubits
    for qubit, op in term:
        paulis[qubit] = op
    return ''.join(paulis)

def validate_hamiltonian(pauli_terms, e_hf):
    """해밀토니안 검증: HF 상태 에너지 계산"""
    # HF 상태: |0011⟩ (CASSCF(2,2), 2 전자)
    hf_energy = 0.0
    
    for pauli_str, coeff in pauli_terms.items():
        # HF 상태에서의 기대값 계산
        expectation = 1.0
        for i, p in enumerate(pauli_str):
            # |0011⟩: q0=1, q1=1, q2=0, q3=0
            qubit_state = 1 if i < 2 else 0
            if p == 'Z':
                expectation *= (1 - 2 * qubit_state)
            elif p in ['X', 'Y']:
                expectation = 0
                break
        hf_energy += coeff * expectation
    
    print(f"\n검증:")
    print(f"  HF 상태 에너지 (계산): {hf_energy:.6f} Ha")
    print(f"  HF 참조값: {e_hf:.6f} Ha")
    print(f"  차이: {abs(hf_energy - e_hf)*1000:.2f} mHa")
    
    return hf_energy

def save_hamiltonian(pauli_terms, e_hf, e_casscf, filename="beh2_4qubit_hamiltonian.json"):
    """해밀토니안을 JSON으로 저장"""
    output = {
        "molecule": "BeH2",
        "basis": "STO-3G",
        "active_space": "CASSCF(2,2)",
        "n_qubits": 4,
        "e_hf": e_hf,
        "e_casscf": e_casscf,
        "pauli_terms": pauli_terms
    }
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n저장됨: {filename}")
    return filename

def generate_qiskit_code(pauli_terms):
    """Qiskit SparsePauliOp 코드 생성"""
    print("\n" + "=" * 60)
    print("Qiskit 코드 (복사용)")
    print("=" * 60)
    print("from qiskit.quantum_info import SparsePauliOp\n")
    print("pauli_terms = [")
    
    # 상위 10개만 출력
    sorted_terms = sorted(pauli_terms.items(), key=lambda x: abs(x[1]), reverse=True)
    for pauli, coeff in sorted_terms[:15]:
        print(f'    ("{pauli}", {coeff:.6f}),')
    if len(pauli_terms) > 15:
        print(f"    # ... {len(pauli_terms) - 15} more terms")
    print("]")
    print("hamiltonian = SparsePauliOp.from_list(pauli_terms)")

def main():
    # 1. CASSCF 계산 및 적분 추출
    h1e, h2e, constant, e_hf, e_casscf, n_qubits = run_casscf_and_extract()
    
    # 2. Jordan-Wigner 변환
    pauli_terms = jordan_wigner_transform(h1e, h2e, constant, n_qubits)
    
    if pauli_terms is None:
        return
    
    # 3. 해밀토니안 검증
    validate_hamiltonian(pauli_terms, e_hf)
    
    # 4. 저장
    save_hamiltonian(pauli_terms, e_hf, e_casscf)
    
    # 5. Qiskit 코드 출력
    generate_qiskit_code(pauli_terms)
    
    print("\n" + "=" * 60)
    print("완료")
    print("=" * 60)

if __name__ == "__main__":
    main()

"""
BeH₂ 해밀토니안 생성 (openfermion-pyscf 사용)

openfermion-pyscf를 사용하여 정확한 Jordan-Wigner 변환 수행

사용법:
    wsl -d Ubuntu -- python3 /mnt/d/SynProject/Engine/QC/tqp/scripts/ibm/generate_beh2_openfermion.py
"""

import json
import numpy as np

def generate_beh2_hamiltonian():
    """openfermion-pyscf로 BeH₂ 해밀토니안 생성"""
    from openfermion import jordan_wigner, get_fermion_operator
    from openfermion.chem import MolecularData
    from openfermionpyscf import run_pyscf
    
    print("=" * 60)
    print("BeH₂ Hamiltonian (openfermion-pyscf)")
    print("=" * 60)
    
    # 분자 정의
    geometry = [
        ('Be', (0.0, 0.0, 0.0)),
        ('H', (0.0, 0.0, 1.326)),
        ('H', (0.0, 0.0, -1.326)),
    ]
    
    basis = 'sto-3g'
    multiplicity = 1
    charge = 0
    
    # MolecularData 객체 생성
    molecule = MolecularData(
        geometry=geometry,
        basis=basis,
        multiplicity=multiplicity,
        charge=charge,
        description="BeH2_linear"
    )
    
    # PySCF로 계산 실행
    print("\nPySCF 계산 중...")
    molecule = run_pyscf(molecule, run_scf=True, run_fci=True)
    
    print(f"HF 에너지: {molecule.hf_energy:.6f} Ha")
    print(f"FCI 에너지: {molecule.fci_energy:.6f} Ha")
    print(f"상관 에너지: {(molecule.fci_energy - molecule.hf_energy)*1000:.2f} mHa")
    print(f"큐비트 수 (스핀-궤도): {molecule.n_qubits}")
    
    # 분자 해밀토니안 추출
    molecular_hamiltonian = molecule.get_molecular_hamiltonian()
    
    # Fermion 연산자로 변환
    fermion_hamiltonian = get_fermion_operator(molecular_hamiltonian)
    
    # Jordan-Wigner 변환
    qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)
    
    # Pauli 항 추출
    n_qubits = molecule.n_qubits
    pauli_terms = {}
    
    for term, coeff in qubit_hamiltonian.terms.items():
        if abs(coeff) > 1e-10:
            pauli_str = pauli_term_to_string(term, n_qubits)
            pauli_terms[pauli_str] = float(coeff.real)
    
    print(f"\n변환 결과:")
    print(f"  Pauli 항 수: {len(pauli_terms)}")
    
    # 상위 항 출력
    sorted_terms = sorted(pauli_terms.items(), key=lambda x: abs(x[1]), reverse=True)
    print("\n상위 15개 항:")
    for pauli, coeff in sorted_terms[:15]:
        print(f"  {pauli}: {coeff:+.6f}")
    
    # HF 에너지 검증
    hf_energy = compute_hf_expectation(pauli_terms, n_qubits, molecule.n_electrons)
    print(f"\nHF 상태 검증:")
    print(f"  계산된 에너지: {hf_energy:.6f} Ha")
    print(f"  참조 HF 에너지: {molecule.hf_energy:.6f} Ha")
    print(f"  차이: {abs(hf_energy - molecule.hf_energy)*1000:.2f} mHa")
    
    # 결과 저장
    output = {
        "molecule": "BeH2",
        "basis": basis,
        "n_qubits": n_qubits,
        "n_electrons": molecule.n_electrons,
        "hf_energy": molecule.hf_energy,
        "fci_energy": molecule.fci_energy,
        "n_pauli_terms": len(pauli_terms),
        "pauli_terms": pauli_terms
    }
    
    filename = "beh2_openfermion_hamiltonian.json"
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n저장됨: {filename}")
    
    # Qiskit 코드 생성
    generate_qiskit_code(sorted_terms, n_qubits)
    
    return output

def pauli_term_to_string(term, n_qubits):
    """OpenFermion 항을 Pauli 문자열로 변환"""
    paulis = ['I'] * n_qubits
    for qubit, op in term:
        paulis[qubit] = op
    return ''.join(paulis)

def compute_hf_expectation(pauli_terms, n_qubits, n_electrons):
    """HF 상태에서의 해밀토니안 기대값"""
    # HF 상태: 처음 n_electrons 스핀-궤도 점유
    hf_state = [1] * n_electrons + [0] * (n_qubits - n_electrons)
    
    hf_energy = 0.0
    for pauli_str, coeff in pauli_terms.items():
        expectation = 1.0
        for i, p in enumerate(pauli_str):
            state = hf_state[i] if i < len(hf_state) else 0
            if p == 'Z':
                expectation *= (1 - 2 * state)
            elif p in ['X', 'Y']:
                expectation = 0
                break
        hf_energy += coeff * expectation
    
    return hf_energy

def generate_qiskit_code(sorted_terms, n_qubits):
    """Qiskit SparsePauliOp 코드 생성"""
    print("\n" + "=" * 60)
    print("Qiskit 코드")
    print("=" * 60)
    print("from qiskit.quantum_info import SparsePauliOp\n")
    print("pauli_terms = [")
    
    for pauli, coeff in sorted_terms[:20]:
        print(f'    ("{pauli}", {coeff:.6f}),')
    if len(sorted_terms) > 20:
        print(f"    # ... {len(sorted_terms) - 20} more terms")
    print("]")
    print("hamiltonian = SparsePauliOp.from_list(pauli_terms)")

if __name__ == "__main__":
    generate_beh2_hamiltonian()

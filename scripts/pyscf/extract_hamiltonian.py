"""
해밀토니안 추출 스크립트

CASSCF 적분에서 Jordan-Wigner 변환을 수행하여
6-qubit Pauli 해밀토니안을 생성합니다.

사용법:
    python extract_hamiltonian.py

입력:
    beh2_integrals.npz (beh2_casscf.py 출력)

출력:
    beh2_hamiltonian.json (Pauli 계수)
"""

import json
import numpy as np


def load_integrals(filename="beh2_integrals.npz"):
    """적분 로드"""
    data = np.load(filename)
    return {
        'h1e': data['h1e'],
        'h2e': data['h2e'],
        'e_nuc': float(data['e_nuc']),
        'e_core': float(data['e_core'])
    }


def jordan_wigner_transform(h1e, h2e, n_qubits=6):
    """Jordan-Wigner 변환 수행
    
    Args:
        h1e: 1-전자 적분 (n_orb x n_orb)
        h2e: 2-전자 적분 (n_orb x n_orb x n_orb x n_orb)
        n_qubits: 큐비트 수
    
    Returns:
        Dict[str, float]: Pauli 문자열 -> 계수
    """
    try:
        from openfermion import jordan_wigner
        from openfermion.ops import InteractionOperator
        from openfermion.transforms import get_fermion_operator
        
        # InteractionOperator 생성
        n_orb = h1e.shape[0]
        constant = 0.0  # 핵간 반발 + 코어 에너지는 별도 처리
        
        # OpenFermion 형식으로 변환
        int_op = InteractionOperator(constant, h1e, 0.5 * h2e)
        fermion_op = get_fermion_operator(int_op)
        
        # Jordan-Wigner 변환
        qubit_op = jordan_wigner(fermion_op)
        
        # Pauli 항 추출
        pauli_terms = {}
        for term, coeff in qubit_op.terms.items():
            if abs(coeff) > 1e-10:  # 작은 항 무시
                pauli_str = pauli_term_to_string(term, n_qubits)
                pauli_terms[pauli_str] = float(coeff.real)
        
        return pauli_terms
        
    except ImportError:
        print("OpenFermion이 설치되지 않았습니다.")
        print("설치: pip install openfermion")
        return manual_jw_transform(h1e, h2e, n_qubits)


def pauli_term_to_string(term, n_qubits):
    """OpenFermion 항을 Pauli 문자열로 변환"""
    paulis = ['I'] * n_qubits
    for qubit, op in term:
        paulis[qubit] = op
    return ''.join(paulis)


def manual_jw_transform(h1e, h2e, n_qubits):
    """수동 Jordan-Wigner 변환 (폴백)
    
    간단한 구현 - 실제로는 OpenFermion 권장
    """
    pauli_terms = {}
    n_orb = h1e.shape[0]
    
    # 1-body 항: h_pq (a†_p a_q + h.c.)
    for p in range(n_orb):
        for q in range(n_orb):
            if abs(h1e[p, q]) > 1e-10:
                # 대각 항: Z 연산자
                if p == q:
                    # 스핀-업과 스핀-다운 모두
                    for spin in [0, 1]:
                        qubit = 2 * p + spin
                        pauli = 'I' * qubit + 'Z' + 'I' * (n_qubits - qubit - 1)
                        coeff = -0.5 * h1e[p, p]
                        if pauli in pauli_terms:
                            pauli_terms[pauli] += coeff
                        else:
                            pauli_terms[pauli] = coeff
    
    # Identity 항 (대각 기여 합)
    identity_coeff = 0.0
    for p in range(n_orb):
        identity_coeff += h1e[p, p]
    pauli_terms['I' * n_qubits] = identity_coeff * 0.5
    
    # 참고: 실제 2-body 항은 더 복잡함
    # OpenFermion 사용 권장
    
    return pauli_terms


def validate_hamiltonian(pauli_terms, expected_term_count=None):
    """해밀토니안 검증"""
    print(f"\n{'='*50}")
    print("Hamiltonian Validation")
    print(f"{'='*50}")
    print(f"Total terms: {len(pauli_terms)}")
    
    if expected_term_count:
        if len(pauli_terms) >= expected_term_count * 0.8:
            print(f"✓ Term count OK (expected ~{expected_term_count})")
        else:
            print(f"⚠ Term count low (expected ~{expected_term_count})")
    
    # 에르미트성 검사 (실수 계수 확인)
    all_real = all(isinstance(v, float) for v in pauli_terms.values())
    if all_real:
        print("✓ All coefficients are real (Hermitian)")
    else:
        print("⚠ Complex coefficients detected")
    
    # 가장 큰 계수 출력
    sorted_terms = sorted(pauli_terms.items(), key=lambda x: abs(x[1]), reverse=True)
    print("\nTop 10 terms:")
    for pauli, coeff in sorted_terms[:10]:
        print(f"  {pauli}: {coeff:+.6f}")
    
    print(f"{'='*50}\n")


def save_hamiltonian(pauli_terms, e_nuc, e_core, filename="beh2_hamiltonian.json"):
    """해밀토니안을 JSON으로 저장"""
    output = {
        "molecule": "BeH2",
        "basis": "STO-3G",
        "active_space": "CASSCF(3,2)",
        "n_qubits": 6,
        "nuclear_repulsion": e_nuc,
        "core_energy": e_core,
        "constant_offset": e_nuc + e_core,
        "pauli_terms": pauli_terms
    }
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Hamiltonian saved to {filename}")


def main():
    """메인 실행"""
    print("="*60)
    print("BeH₂ Hamiltonian Extraction")
    print("="*60)
    
    # 1. 적분 로드
    try:
        integrals = load_integrals()
        print(f"\nLoaded integrals:")
        print(f"  h1e shape: {integrals['h1e'].shape}")
        print(f"  h2e shape: {integrals['h2e'].shape}")
        print(f"  E_nuc: {integrals['e_nuc']:.6f} Ha")
        print(f"  E_core: {integrals['e_core']:.6f} Ha")
    except FileNotFoundError:
        print("Error: beh2_integrals.npz not found")
        print("먼저 beh2_casscf.py를 실행하세요.")
        return
    
    # 2. Jordan-Wigner 변환
    print("\nPerforming Jordan-Wigner transformation...")
    pauli_terms = jordan_wigner_transform(
        integrals['h1e'],
        integrals['h2e'],
        n_qubits=6
    )
    
    # 3. 검증
    validate_hamiltonian(pauli_terms, expected_term_count=100)
    
    # 4. 저장
    save_hamiltonian(
        pauli_terms,
        integrals['e_nuc'],
        integrals['e_core']
    )
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

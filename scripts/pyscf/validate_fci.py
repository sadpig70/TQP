"""
FCI 검증 스크립트

생성된 Pauli 해밀토니안이 올바른 FCI 에너지를 재현하는지 검증합니다.

사용법:
    python validate_fci.py

입력:
    beh2_hamiltonian.json (extract_hamiltonian.py 출력)
"""

import json
import numpy as np


def load_hamiltonian(filename="beh2_hamiltonian.json"):
    """해밀토니안 로드"""
    with open(filename, 'r') as f:
        return json.load(f)


def pauli_string_to_matrix(pauli_str):
    """Pauli 문자열을 행렬로 변환"""
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    pauli_map = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
    
    result = np.array([[1]], dtype=complex)
    for char in pauli_str:
        result = np.kron(result, pauli_map[char])
    
    return result


def build_hamiltonian_matrix(hamiltonian_data):
    """전체 해밀토니안 행렬 구축"""
    n_qubits = hamiltonian_data['n_qubits']
    dim = 2 ** n_qubits
    
    H = np.zeros((dim, dim), dtype=complex)
    
    # 상수 항 추가
    H += hamiltonian_data['constant_offset'] * np.eye(dim)
    
    # Pauli 항 추가
    for pauli_str, coeff in hamiltonian_data['pauli_terms'].items():
        H += coeff * pauli_string_to_matrix(pauli_str)
    
    return H


def validate_hermiticity(H):
    """에르미트성 검사"""
    diff = np.max(np.abs(H - H.conj().T))
    is_hermitian = diff < 1e-10
    print(f"Hermiticity check: {'PASS' if is_hermitian else 'FAIL'} (max diff: {diff:.2e})")
    return is_hermitian


def compute_ground_state(H):
    """기저 상태 에너지 계산"""
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    
    ground_energy = eigenvalues[0]
    ground_state = eigenvectors[:, 0]
    
    return ground_energy, ground_state, eigenvalues


def analyze_ground_state(ground_state, n_qubits):
    """기저 상태 분석"""
    dim = 2 ** n_qubits
    
    print("\nGround state analysis:")
    
    # 가장 큰 진폭 찾기
    amplitudes = np.abs(ground_state) ** 2
    sorted_idx = np.argsort(amplitudes)[::-1]
    
    print("Top 5 basis states:")
    for i in sorted_idx[:5]:
        basis = format(i, f'0{n_qubits}b')
        prob = amplitudes[i]
        if prob > 0.001:
            print(f"  |{basis}⟩: {prob:.4f}")


def compare_with_reference(computed_energy, reference_fci=-15.5952):
    """참조 에너지와 비교"""
    error_ha = abs(computed_energy - reference_fci)
    error_mha = error_ha * 1000
    
    print(f"\n{'='*50}")
    print("Energy Comparison")
    print(f"{'='*50}")
    print(f"Computed:  {computed_energy:.6f} Ha")
    print(f"Reference: {reference_fci:.6f} Ha")
    print(f"Error:     {error_mha:.3f} mHa")
    
    if error_mha < 0.1:
        print("✓ Excellent agreement (< 0.1 mHa)")
        return True
    elif error_mha < 1.0:
        print("✓ Good agreement (< 1 mHa)")
        return True
    elif error_mha < 10.0:
        print("⚠ Moderate agreement (< 10 mHa)")
        return True
    else:
        print("✗ Poor agreement (> 10 mHa)")
        return False


def main():
    """메인 실행"""
    print("="*60)
    print("BeH₂ FCI Validation")
    print("="*60)
    
    # 1. 해밀토니안 로드
    try:
        hamiltonian_data = load_hamiltonian()
        print(f"\nLoaded Hamiltonian:")
        print(f"  Molecule: {hamiltonian_data['molecule']}")
        print(f"  Qubits: {hamiltonian_data['n_qubits']}")
        print(f"  Pauli terms: {len(hamiltonian_data['pauli_terms'])}")
        print(f"  Constant offset: {hamiltonian_data['constant_offset']:.6f} Ha")
    except FileNotFoundError:
        print("Error: beh2_hamiltonian.json not found")
        print("먼저 extract_hamiltonian.py를 실행하세요.")
        return
    
    # 2. 해밀토니안 행렬 구축
    print("\nBuilding Hamiltonian matrix...")
    H = build_hamiltonian_matrix(hamiltonian_data)
    print(f"Matrix shape: {H.shape}")
    
    # 3. 에르미트성 검사
    validate_hermiticity(H)
    
    # 4. 기저 상태 계산
    print("\nDiagonalizing Hamiltonian...")
    ground_energy, ground_state, all_energies = compute_ground_state(H)
    
    print(f"\nFirst 5 eigenvalues (Ha):")
    for i, e in enumerate(all_energies[:5]):
        print(f"  E_{i}: {e:.6f}")
    
    # 5. 기저 상태 분석
    analyze_ground_state(ground_state, hamiltonian_data['n_qubits'])
    
    # 6. 참조값과 비교
    success = compare_with_reference(ground_energy)
    
    print(f"\n{'='*60}")
    print(f"VALIDATION: {'PASSED' if success else 'FAILED'}")
    print(f"{'='*60}")
    
    return success


if __name__ == "__main__":
    main()

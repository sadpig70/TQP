"""
H₂ 분자 IBM VQE 테스트 - ENDIANNESS 수정판

Qiskit Pauli string 규칙:
  - "IIIZ" = Z on q3 (rightmost = highest index qubit)
  - "ZIII" = Z on q0 (leftmost = lowest index qubit)

HF 상태 매핑:
  - 활성 공간: 4 spin-orbitals (q0, q1, q2, q3)
  - 점유: q0=↑, q1=↓ (bonding orbital)
  - 비점유: q2=↑, q3=↓ (antibonding orbital)
  
회로에서 점유 상태 표시:
  - qc.x(0), qc.x(1) → q0=|1⟩, q1=|1⟩
  
Pauli 문자열에서:
  - 점유 큐비트 (q0, q1)의 Z expectation = -1
  - 비점유 큐비트 (q2, q3)의 Z expectation = +1
"""

import json
from pathlib import Path

try:
    from qiskit import QuantumCircuit
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator
    from qiskit.quantum_info import SparsePauliOp, Statevector
    IBM_AVAILABLE = True
    print("✓ Qiskit/IBM Runtime 로드 성공")
except ImportError as e:
    print(f"✗ IBM Runtime 로드 실패: {e}")
    IBM_AVAILABLE = False


def load_api_credentials():
    script_dir = Path(__file__).parent
    filepath = script_dir / "../../tqp-ibm-apikey.json"
    with open(filepath.resolve()) as f:
        data = json.load(f)
    return data.get("apikey"), data.get("name", "TQP")


def create_h2_hf_circuit():
    """H₂ HF 상태 회로 (4-qubit)"""
    qc = QuantumCircuit(4)
    qc.x(0)  # q0 = |1⟩ (occupied)
    qc.x(1)  # q1 = |1⟩ (occupied)
    # q2, q3 = |0⟩ (virtual)
    qc.barrier()
    return qc


def create_h2_hamiltonian_fixed():
    """H₂ 해밀토니안 (endianness 수정됨)
    
    Qiskit 규칙: 
      - Pauli string index: "P_q3 P_q2 P_q1 P_q0"
      - 즉 rightmost = q0, leftmost = q3
    
    HF 상태 |q3=0, q2=0, q1=1, q0=1⟩:
      - Z_q0 exp = -1, Z_q1 exp = -1
      - Z_q2 exp = +1, Z_q3 exp = +1
    
    참조: Qiskit Nature H₂ STO-3G at R=0.735Å
    """
    # H₂ 해밀토니안 (올바른 부호)
    pauli_terms = [
        ("IIII", -0.8105),   # Identity + nuclear repulsion
        ("IIIZ", +0.1721),   # Z_q0 → -1 → contrib = -0.1721
        ("IIZI", -0.2257),   # Z_q1 → -1 → contrib = +0.2257
        ("IZII", -0.2257),   # Z_q2 → +1 → contrib = -0.2257
        ("ZIII", +0.1721),   # Z_q3 → +1 → contrib = +0.1721
        ("IIZZ", +0.1209),   # Z_q0Z_q1 → +1 → contrib = +0.1209
        ("IZIZ", +0.0453),   # Z_q0Z_q2 → -1 → contrib = -0.0453
        ("IZZI", +0.1657),   # Z_q1Z_q2 → -1 → contrib = -0.1657
        ("ZIIZ", +0.1657),   # Z_q0Z_q3 → -1 → contrib = -0.1657
        ("ZIZI", +0.0453),   # Z_q1Z_q3 → -1 → contrib = -0.0453
        ("ZZII", +0.1209),   # Z_q2Z_q3 → +1 → contrib = +0.1209
        ("XXYY", -0.0453),   # Exchange
        ("XYYX", +0.0453),
        ("YXXY", +0.0453),
        ("YYXX", -0.0453),
    ]
    
    return SparsePauliOp.from_list(pauli_terms)


def verify_locally():
    """로컬에서 이론값 검증"""
    print("\n" + "=" * 50)
    print("H₂ 로컬 시뮬레이션 검증")
    print("=" * 50)
    
    qc = create_h2_hf_circuit()
    sv = Statevector.from_instruction(qc)
    
    hamiltonian = create_h2_hamiltonian_fixed()
    energy = sv.expectation_value(hamiltonian).real
    
    # 참조값 (Qiskit Nature 문서)
    hf_reference = -1.1167
    fci_reference = -1.1373
    
    print(f"시뮬레이션 HF 에너지: {energy:.6f} Ha")
    print(f"문헌 HF 참조값:       {hf_reference:.6f} Ha")
    print(f"문헌 FCI 참조값:      {fci_reference:.6f} Ha")
    print(f"HF 오차:              {(energy - hf_reference)*1000:.2f} mHa")
    
    if abs(energy - hf_reference) < 0.01:  # 10 mHa 이내
        print("✓ 해밀토니안 검증 통과!")
        return True
    else:
        print("✗ 해밀토니안 검증 실패 - 계수 확인 필요")
        return False


def run_ibm(backend_name: str = "ibm_torino"):
    """IBM에서 실행"""
    if not IBM_AVAILABLE:
        print("IBM Runtime이 설치되지 않았습니다.")
        return None
    
    # 먼저 로컬 검증
    if not verify_locally():
        print("로컬 검증 실패. IBM 실행 중단.")
        return None
    
    api_key, _ = load_api_credentials()
    
    print("\n" + "=" * 50)
    print(f"H₂ IBM 실행: {backend_name}")
    print("=" * 50)
    
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=api_key)
    backend = service.backend(backend_name)
    
    circuit = create_h2_hf_circuit()
    hamiltonian = create_h2_hamiltonian_fixed()
    
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(circuit)
    isa_observable = hamiltonian.apply_layout(isa_circuit.layout)
    
    print(f"작업 제출 중...")
    estimator = Estimator(backend)
    job = estimator.run([(isa_circuit, isa_observable)])
    print(f"Job ID: {job.job_id()}")
    print("결과 대기 중...")
    
    result = job.result()
    energy = result[0].data.evs
    std_error = result[0].data.stds
    
    print(f"\n측정 에너지: {energy:.6f} ± {std_error:.6f} Ha")
    print(f"HF 참조값:   -1.1167 Ha")
    print(f"오차:        {(energy - (-1.1167))*1000:.2f} mHa")
    
    return energy


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="ibm_torino")
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args()
    
    if args.local_only:
        verify_locally()
    else:
        run_ibm(args.backend)

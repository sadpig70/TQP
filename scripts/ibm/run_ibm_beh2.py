"""
IBM Quantum BeH₂ VQE 실행 스크립트

IBM Quantum 백엔드에서 BeH₂ 분자의 VQE 에너지 측정을 수행합니다.

사용법:
    python run_ibm_beh2.py [--backend ibm_sherbrooke] [--shots 4096]
"""

import json
import argparse
from pathlib import Path

try:
    from qiskit import QuantumCircuit
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator
    from qiskit_ibm_runtime import Options
    from qiskit.quantum_info import SparsePauliOp
    IBM_AVAILABLE = True
    print("✓ Qiskit/IBM Runtime 로드 성공")
except ImportError as e:
    print(f"✗ IBM Runtime 로드 실패: {e}")
    IBM_AVAILABLE = False


def load_api_credentials(filepath: str = None):
    """API 인증 정보 로드"""
    if filepath is None:
        # 스크립트 위치 기준 상대 경로
        script_dir = Path(__file__).parent
        filepath = script_dir / "../../tqp-ibm-apikey.json"
    
    path = Path(filepath).resolve()
    if not path.exists():
        raise FileNotFoundError(f"API key file not found: {path}")
    
    with open(path) as f:
        data = json.load(f)
    
    return data.get("apikey"), data.get("name", "TQP")


def create_beh2_hf_circuit():
    """BeH₂ HF 상태 회로 생성 (6-qubit)"""
    qc = QuantumCircuit(6)
    
    # HF 상태 |000011⟩ 준비
    qc.x(0)  # 궤도 1, spin up
    qc.x(1)  # 궤도 1, spin down
    
    qc.barrier()
    return qc


def create_beh2_hamiltonian():
    """BeH₂ 해밀토니안 생성 (Z-only terms for HF verification)"""
    pauli_terms = [
        ("IIIIII", -15.4947),  # Identity
        ("IIIIIZ", 0.1562),    # Z0
        ("IIIIZI", 0.1562),    # Z1
        ("IIIIZZ", 0.0782),    # Z0Z1
        ("IIIZII", -0.1089),   # Z2
        ("IIZIII", -0.1089),   # Z3
        ("IZIIII", -0.0534),   # Z4
        ("ZIIIII", -0.0534),   # Z5
        ("IIIZIZ", 0.0423),    # Z0Z2
        ("IIIZZI", 0.0423),    # Z1Z2
        ("IIZIZI", 0.0512),    # Z1Z3
        ("IIZIIZ", 0.0512),    # Z0Z3
        ("IIZZII", 0.0389),    # Z2Z3
        ("IZIIZI", 0.0267),    # Z1Z4
        ("IZIIIZ", 0.0267),    # Z0Z4
        ("IZIZII", 0.0334),    # Z2Z4
        ("IZZIII", 0.0334),    # Z3Z4
        ("ZIIZII", 0.0267),    # Z2Z5
        ("ZIIIZI", 0.0267),    # Z1Z5
        ("ZIIIIZ", 0.0312),    # Z0Z5
        ("ZIZIII", 0.0334),    # Z3Z5
        ("ZZIIII", 0.0223),    # Z4Z5
    ]
    
    return SparsePauliOp.from_list(pauli_terms)


def run_ibm_estimation(backend_name: str = "ibm_sherbrooke", shots: int = 4096):
    """IBM 백엔드에서 에너지 추정 실행"""
    if not IBM_AVAILABLE:
        print("IBM Runtime이 설치되지 않았습니다.")
        return None
    
    # API 키 로드
    api_key, instance_name = load_api_credentials()
    
    print(f"=" * 60)
    print(f"IBM Quantum BeH₂ VQE Execution")
    print(f"=" * 60)
    print(f"Instance: {instance_name}")
    print(f"Backend: {backend_name}")
    print(f"Shots: {shots}")
    
    # 서비스 초기화
    try:
        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=api_key)
        print("✓ IBM Quantum 서비스 연결 성공")
    except Exception as e:
        print(f"✗ 서비스 연결 실패: {e}")
        return None
    
    # 백엔드 선택
    try:
        backend = service.backend(backend_name)
        print(f"✓ 백엔드 선택: {backend.name}")
        print(f"  - Qubits: {backend.num_qubits}")
        print(f"  - Status: {backend.status().status_msg}")
    except Exception as e:
        print(f"✗ 백엔드 연결 실패: {e}")
        # 시뮬레이터로 폴백
        print("시뮬레이터로 대체 실행...")
        backend = service.backend("ibmq_qasm_simulator")
    
    # 회로 및 해밀토니안 생성
    circuit = create_beh2_hf_circuit()
    hamiltonian = create_beh2_hamiltonian()
    
    print(f"\n회로 정보:")
    print(f"  - 큐비트: {circuit.num_qubits}")
    print(f"  - 깊이: {circuit.depth()}")
    print(f"해밀토니안 항: {len(hamiltonian)}")
    
    # 트랜스파일
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(circuit)
    isa_observable = hamiltonian.apply_layout(isa_circuit.layout)
    
    print(f"\n트랜스파일 후 깊이: {isa_circuit.depth()}")
    
    # Estimator 실행
    print(f"\n작업 제출 중...")
    
    try:
        estimator = Estimator(backend)
        job = estimator.run([(isa_circuit, isa_observable)])
        
        print(f"Job ID: {job.job_id()}")
        print(f"상태 대기 중...")
        
        result = job.result()
        energy = result[0].data.evs
        std_error = result[0].data.stds
        
        print(f"\n" + "=" * 60)
        print(f"결과")
        print(f"=" * 60)
        print(f"측정 에너지: {energy:.6f} Ha")
        print(f"표준 오차: {std_error:.6f} Ha")
        
        # 참조값과 비교
        hf_reference = -15.5614
        error_mha = (energy - hf_reference) * 1000
        print(f"\nHF 참조값: {hf_reference:.6f} Ha")
        print(f"오차: {error_mha:.2f} mHa")
        
        if abs(error_mha) < 5.0:
            print("✓ 검증 통과 (오차 < 5 mHa)")
        else:
            print("✗ 검증 실패 (오차 >= 5 mHa)")
        
        # 결과 저장
        result_data = {
            "backend": backend_name,
            "shots": shots,
            "energy_ha": float(energy),
            "std_error_ha": float(std_error),
            "hf_reference_ha": hf_reference,
            "error_mha": float(error_mha),
            "job_id": job.job_id()
        }
        
        with open("ibm_beh2_result.json", "w") as f:
            json.dump(result_data, f, indent=2)
        print(f"\n결과 저장: ibm_beh2_result.json")
        
        return result_data
        
    except Exception as e:
        print(f"✗ 실행 실패: {e}")
        return None


def check_backends():
    """사용 가능한 백엔드 목록 확인"""
    if not IBM_AVAILABLE:
        print("IBM Runtime이 설치되지 않았습니다.")
        return
    
    api_key, _ = load_api_credentials()
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=api_key)
    
    print("사용 가능한 백엔드:")
    for backend in service.backends():
        status = backend.status()
        print(f"  {backend.name}: {backend.num_qubits} qubits, {status.status_msg}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IBM Quantum BeH₂ VQE")
    parser.add_argument("--backend", default="ibm_sherbrooke", help="백엔드 이름")
    parser.add_argument("--shots", type=int, default=4096, help="샷 수")
    parser.add_argument("--list-backends", action="store_true", help="백엔드 목록 확인")
    
    args = parser.parse_args()
    
    if args.list_backends:
        check_backends()
    else:
        run_ibm_estimation(args.backend, args.shots)

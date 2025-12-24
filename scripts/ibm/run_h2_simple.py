"""
2-qubit H₂ IBM VQE - 단순화 버전

MemoryError 회피를 위해 트랜스파일을 간소화
"""

import json
from pathlib import Path

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator
    from qiskit.quantum_info import SparsePauliOp
    print("✓ Qiskit/IBM Runtime 로드 성공")
except ImportError as e:
    print(f"✗ 로드 실패: {e}")
    exit(1)


def load_api_credentials():
    script_dir = Path(__file__).parent
    filepath = script_dir / "../../tqp-ibm-apikey.json"
    with open(filepath.resolve()) as f:
        return json.load(f).get("apikey")


def run_h2_simple(backend_name: str = "ibm_fez"):
    """단순화된 H₂ IBM 실행"""
    api_key = load_api_credentials()
    
    print("\n" + "=" * 50)
    print(f"2-qubit H₂ IBM 실행 (단순화): {backend_name}")
    print("=" * 50)
    
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=api_key)
    backend = service.backend(backend_name)
    print(f"✓ 백엔드 연결: {backend.name}")
    
    # 간단한 회로
    qc = QuantumCircuit(2)
    qc.x(1)
    qc.measure_all()
    
    # 간단한 transpile
    print("트랜스파일 중...")
    transpiled = transpile(qc, backend, optimization_level=0)
    print(f"트랜스파일 완료")
    
    # 해밀토니안 (2-qubit)
    h2_terms = [
        ("II", -1.052373),
        ("IZ", -0.397937),
        ("ZI", -0.397937),
        ("ZZ", +0.011280),
        ("XX", +0.180931),
    ]
    hamiltonian = SparsePauliOp.from_list(h2_terms)
    
    # Layout 적용
    isa_observable = hamiltonian.apply_layout(transpiled.layout)
    
    print("\n작업 제출 중...")
    estimator = Estimator(backend)
    
    # measurement가 없는 회로로 다시 만들기
    qc_no_measure = QuantumCircuit(2)
    qc_no_measure.x(1)
    transpiled_no_measure = transpile(qc_no_measure, backend, optimization_level=0)
    isa_observable = hamiltonian.apply_layout(transpiled_no_measure.layout)
    
    job = estimator.run([(transpiled_no_measure, isa_observable)])
    print(f"Job ID: {job.job_id()}")
    print("결과 대기 중...")
    
    result = job.result()
    energy = result[0].data.evs
    std_error = result[0].data.stds
    
    hf_ref = -1.0637  # 로컬 시뮬레이션 값
    print(f"\n" + "=" * 50)
    print("결과")
    print("=" * 50)
    print(f"측정 에너지: {energy:.6f} ± {std_error:.6f} Ha")
    print(f"시뮬레이션 HF: {hf_ref:.6f} Ha")
    print(f"오차:         {(energy - hf_ref)*1000:.2f} mHa")
    
    return energy


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="ibm_fez")
    args = parser.parse_args()
    
    run_h2_simple(args.backend)

"""
2-qubit H₂ IBM VQE 실행

검증된 2-qubit H₂ 해밀토니안으로 IBM 하드웨어 테스트
"""

import json
from pathlib import Path

try:
    from qiskit import QuantumCircuit
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
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


def create_h2_hf_circuit():
    """H₂ HF 상태: |10⟩ (1전자 점유)"""
    qc = QuantumCircuit(2)
    qc.x(1)  # q1 점유
    qc.barrier()
    return qc


def create_h2_hamiltonian_2qubit():
    """검증된 2-qubit H₂ 해밀토니안"""
    return SparsePauliOp.from_list([
        ("II", -1.052373),
        ("IZ", -0.397937),
        ("ZI", -0.397937),
        ("ZZ", +0.011280),
        ("XX", +0.180931),
    ])


def run_ibm(backend_name: str = "ibm_torino"):
    """IBM에서 실행"""
    api_key = load_api_credentials()
    
    print("\n" + "=" * 50)
    print(f"2-qubit H₂ IBM 실행: {backend_name}")
    print("=" * 50)
    
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=api_key)
    backend = service.backend(backend_name)
    
    circuit = create_h2_hf_circuit()
    hamiltonian = create_h2_hamiltonian_2qubit()
    
    print(f"큐비트: {circuit.num_qubits}")
    print(f"해밀토니안 항: {len(hamiltonian)}")
    
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(circuit)
    isa_observable = hamiltonian.apply_layout(isa_circuit.layout)
    
    print(f"트랜스파일 후 깊이: {isa_circuit.depth()}")
    
    print("\n작업 제출 중...")
    estimator = Estimator(backend)
    job = estimator.run([(isa_circuit, isa_observable)])
    print(f"Job ID: {job.job_id()}")
    print("결과 대기 중...")
    
    result = job.result()
    energy = result[0].data.evs
    std_error = result[0].data.stds
    
    hf_ref = -1.1173
    print(f"\n" + "=" * 50)
    print("결과")
    print("=" * 50)
    print(f"측정 에너지: {energy:.6f} ± {std_error:.6f} Ha")
    print(f"HF 참조값:   {hf_ref:.6f} Ha")
    print(f"오차:        {(energy - hf_ref)*1000:.2f} mHa")
    
    # 결과 저장
    result_data = {
        "backend": backend_name,
        "molecule": "H2",
        "qubits": 2,
        "energy_ha": float(energy),
        "std_error_ha": float(std_error),
        "hf_reference_ha": hf_ref,
        "job_id": job.job_id()
    }
    
    with open("h2_2qubit_result.json", "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"\n결과 저장: h2_2qubit_result.json")
    
    return energy


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="ibm_torino")
    args = parser.parse_args()
    
    run_ibm(args.backend)

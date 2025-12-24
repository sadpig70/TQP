"""
H₂ 2-qubit IBM VQE - 통계 검증 버전

3회 반복 실행으로 통계적 유의성 확보

사용법:
    python run_h2_statistical.py --backend ibm_torino --trials 3
"""

import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator
    from qiskit.quantum_info import SparsePauliOp
    print("✓ Qiskit/IBM Runtime 로드 성공")
except ImportError as e:
    print(f"✗ 로드 실패: {e}")
    exit(1)


def load_api_credentials() -> str:
    """API 인증 로드"""
    script_dir = Path(__file__).parent
    filepath = script_dir / "../../tqp-ibm-apikey.json"
    
    if not filepath.resolve().exists():
        raise FileNotFoundError(f"API key file not found: {filepath.resolve()}")
    
    with open(filepath.resolve()) as f:
        data = json.load(f)
    
    api_key = data.get("apikey")
    if not api_key:
        raise ValueError("API key is empty or missing in config file")
    
    return api_key


def create_h2_hf_circuit():
    """H₂ HF 상태 회로 (|01⟩)"""
    qc = QuantumCircuit(2)
    qc.x(1)  # |01⟩ 상태
    return qc


def create_h2_hamiltonian():
    """H₂ 2-qubit 해밀토니안 (R=0.735Å)"""
    # 검증된 문헌값
    h2_terms = [
        ("II", -1.052373),
        ("IZ", -0.397937),
        ("ZI", -0.397937),
        ("ZZ", +0.011280),
        ("XX", +0.180931),
    ]
    return SparsePauliOp.from_list(h2_terms)


def run_single_estimation(backend, circuit, hamiltonian, trial_num):
    """단일 추정 실행"""
    print(f"\n--- Trial {trial_num} ---")
    
    # 트랜스파일
    transpiled = transpile(circuit, backend, optimization_level=1)
    isa_observable = hamiltonian.apply_layout(transpiled.layout)
    
    # Estimator 실행
    estimator = Estimator(backend)
    job = estimator.run([(transpiled, isa_observable)])
    
    print(f"Job ID: {job.job_id()}")
    print("결과 대기 중...")
    
    result = job.result()
    energy = float(result[0].data.evs)
    std_error = float(result[0].data.stds)
    
    print(f"에너지: {energy:.6f} ± {std_error:.6f} Ha")
    
    return {
        "trial": trial_num,
        "energy": energy,
        "std_error": std_error,
        "job_id": job.job_id()
    }


def compute_statistics(results: list) -> dict:
    """통계 분석"""
    if not results:
        raise ValueError("No results to compute statistics")
    
    energies = [r["energy"] for r in results]
    n = len(energies)
    
    mean = np.mean(energies)
    
    # N=1일 때 표준편차 계산 불가 - 개별 std_error 사용
    if n == 1:
        std = results[0].get("std_error", 0.0)
        sem = std
    else:
        std = np.std(energies, ddof=1)  # 표본 표준편차
        sem = std / np.sqrt(n)  # 표준오차
    
    # 95% 신뢰구간 (t-분포)
    t_values = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
    t_value = t_values.get(n, 1.96)  # N>5면 정규분포 근사
    
    ci_lower = mean - t_value * sem
    ci_upper = mean + t_value * sem
    
    return {
        "mean": float(mean),
        "std": float(std),
        "sem": float(sem),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "n_trials": n
    }


def run_statistical_validation(backend_name="ibm_torino", n_trials=3):
    """통계 검증 실행"""
    api_key = load_api_credentials()
    
    print("\n" + "=" * 60)
    print(f"H₂ 2-qubit 통계 검증")
    print("=" * 60)
    print(f"백엔드: {backend_name}")
    print(f"반복 횟수: {n_trials}")
    
    # 서비스 연결
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=api_key)
    backend = service.backend(backend_name)
    print(f"✓ 백엔드 연결: {backend.name} ({backend.num_qubits} qubits)")
    
    # 회로 및 해밀토니안
    circuit = create_h2_hf_circuit()
    hamiltonian = create_h2_hamiltonian()
    
    # 반복 실행
    results = []
    for i in range(n_trials):
        try:
            result = run_single_estimation(backend, circuit, hamiltonian, i+1)
            results.append(result)
        except Exception as e:
            print(f"Trial {i+1} 실패: {e}")
    
    if not results:
        print("모든 시도 실패")
        return None
    
    # 통계 분석
    stats = compute_statistics(results)
    
    # 참조값 비교
    hf_ref = -1.0637  # 로컬 시뮬레이션 HF
    fci_ref = -1.1373  # 정확한 FCI
    
    print("\n" + "=" * 60)
    print("통계 분석 결과")
    print("=" * 60)
    print(f"평균 에너지: {stats['mean']:.6f} Ha")
    print(f"표준편차: {stats['std']:.6f} Ha ({stats['std']*1000:.2f} mHa)")
    print(f"표준오차: {stats['sem']:.6f} Ha")
    print(f"95% 신뢰구간: [{stats['ci_95_lower']:.6f}, {stats['ci_95_upper']:.6f}] Ha")
    
    print(f"\n참조값 비교:")
    print(f"  HF 참조: {hf_ref:.6f} Ha")
    print(f"  오차 (vs HF): {(stats['mean'] - hf_ref)*1000:.2f} mHa")
    print(f"  FCI 참조: {fci_ref:.6f} Ha")
    print(f"  오차 (vs FCI): {(stats['mean'] - fci_ref)*1000:.2f} mHa")
    
    # 검증 결과
    error_vs_hf = abs(stats['mean'] - hf_ref) * 1000
    if error_vs_hf < 10:
        print("\n✓ 검증 통과 (오차 < 10 mHa)")
    else:
        print(f"\n✗ 검증 실패 (오차 = {error_vs_hf:.2f} mHa)")
    
    # 결과 저장
    output = {
        "timestamp": datetime.now().isoformat(),
        "backend": backend_name,
        "n_trials": n_trials,
        "trials": results,
        "statistics": stats,
        "reference": {
            "hf": hf_ref,
            "fci": fci_ref
        },
        "error_mha_vs_hf": (stats['mean'] - hf_ref) * 1000
    }
    
    output_file = "h2_statistical_results.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n결과 저장: {output_file}")
    
    return output


def run_local_simulation():
    """로컬 시뮬레이션으로 검증"""
    print("\n" + "=" * 60)
    print("로컬 시뮬레이션 검증")
    print("=" * 60)
    
    circuit = create_h2_hf_circuit()
    hamiltonian = create_h2_hamiltonian()
    
    # HF 상태 기대값 계산
    # |01⟩: q0=0, q1=1
    hf_energy = 0.0
    for pauli, coeff in hamiltonian.to_list():
        expectation = 1.0
        for i, p in enumerate(pauli):
            state = [0, 1][i]  # q0=0, q1=1
            if p == 'Z':
                expectation *= (1 - 2*state)
            elif p in ['X', 'Y']:
                expectation = 0
                break
        hf_energy += coeff * expectation
    
    print(f"HF 상태 에너지: {hf_energy:.6f} Ha")
    return hf_energy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="H₂ 통계 검증")
    parser.add_argument("--backend", default="ibm_torino", help="IBM 백엔드")
    parser.add_argument("--trials", type=int, default=3, help="반복 횟수")
    parser.add_argument("--local", action="store_true", help="로컬 시뮬레이션만")
    
    args = parser.parse_args()
    
    if args.local:
        run_local_simulation()
    else:
        run_statistical_validation(args.backend, args.trials)

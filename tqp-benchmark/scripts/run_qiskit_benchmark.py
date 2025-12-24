"""
TQP vs Qiskit 벤치마크 비교 스크립트

TQP와 Qiskit Aer 시뮬레이터의 상태벡터 연산 성능을 비교합니다.

사용법:
    python run_qiskit_benchmark.py
"""

import time
import json
from dataclasses import dataclass
from typing import List, Dict

try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    print("Warning: Qiskit not available")
    QISKIT_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """벤치마크 결과"""
    n_qubits: int
    n_gates: int
    mean_time_us: float
    std_time_us: float
    median_time_us: float  # 추가
    iqr_time_us: float     # 추가
    trials: int


def benchmark_qiskit_hadamard_chain(n_qubits: int, trials: int = 10) -> BenchmarkResult:
    """Qiskit으로 H 게이트 체인 벤치마크 (10회 반복, 중앙값/IQR)"""
    if not QISKIT_AVAILABLE:
        return BenchmarkResult(n_qubits, n_qubits, 0, 0, 0, 0, 0)
    
    simulator = AerSimulator(method='statevector')
    
    times = []
    for _ in range(trials):
        # 회로 생성
        qc = QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            qc.h(i)
        qc.save_statevector()
        
        # 실행
        start = time.perf_counter()
        job = simulator.run(qc)
        result = job.result()
        end = time.perf_counter()
        
        times.append((end - start) * 1e6)  # μs
    
    import statistics
    import numpy as np
    
    times_arr = np.array(times)
    q1, median, q3 = np.percentile(times_arr, [25, 50, 75])
    
    return BenchmarkResult(
        n_qubits=n_qubits,
        n_gates=n_qubits,
        mean_time_us=statistics.mean(times),
        std_time_us=statistics.stdev(times) if len(times) > 1 else 0,
        median_time_us=median,
        iqr_time_us=q3 - q1,
        trials=trials
    )


def benchmark_qiskit_cnot_chain(n_qubits: int, trials: int = 10) -> BenchmarkResult:
    """Qiskit으로 CNOT 체인 벤치마크"""
    if not QISKIT_AVAILABLE:
        return BenchmarkResult(n_qubits, n_qubits, 0, 0, 0)
    
    simulator = AerSimulator(method='statevector')
    
    times = []
    for _ in range(trials):
        qc = QuantumCircuit(n_qubits)
        qc.h(0)  # 초기 상태
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        qc.save_statevector()
        
        start = time.perf_counter()
        job = simulator.run(qc)
        result = job.result()
        end = time.perf_counter()
        
        times.append((end - start) * 1e6)
    
    import statistics
    return BenchmarkResult(
        n_qubits=n_qubits,
        n_gates=n_qubits,
        mean_time_us=statistics.mean(times),
        std_time_us=statistics.stdev(times) if len(times) > 1 else 0,
        trials=trials
    )


def run_benchmark_suite():
    """전체 벤치마크 스위트 실행"""
    results = {
        "simulator": "Qiskit Aer",
        "method": "statevector",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hadamard_chain": [],
        "cnot_chain": []
    }
    
    print("=" * 60)
    print("Qiskit Aer Statevector Benchmark")
    print("=" * 60)
    
    # H 체인 벤치마크
    print("\n[H Chain Benchmark]")
    for n_qubits in [4, 8, 12, 16]:
        result = benchmark_qiskit_hadamard_chain(n_qubits)
        print(f"  N={n_qubits:2d}: {result.mean_time_us:8.2f} ± {result.std_time_us:6.2f} μs")
        results["hadamard_chain"].append({
            "n_qubits": result.n_qubits,
            "mean_us": result.mean_time_us,
            "std_us": result.std_time_us
        })
    
    # CNOT 체인 벤치마크
    print("\n[CNOT Chain Benchmark]")
    for n_qubits in [4, 8, 12, 16]:
        result = benchmark_qiskit_cnot_chain(n_qubits)
        print(f"  N={n_qubits:2d}: {result.mean_time_us:8.2f} ± {result.std_time_us:6.2f} μs")
        results["cnot_chain"].append({
            "n_qubits": result.n_qubits,
            "mean_us": result.mean_time_us,
            "std_us": result.std_time_us
        })
    
    # 결과 저장
    output_file = "qiskit_benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n결과 저장: {output_file}")
    
    return results


if __name__ == "__main__":
    run_benchmark_suite()

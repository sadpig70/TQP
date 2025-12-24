"""
TQP vs Qiskit 확장 벤치마크 스크립트

N=4-24 범위, 10회 반복, 중앙값 ± IQR 통계 프로토콜
메모리 피크 측정 포함

사용법:
    python run_extended_benchmark.py
"""

import time
import json
import statistics
import tracemalloc
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from pathlib import Path

import numpy as np

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
    trials: int
    # 시간 통계 (마이크로초)
    mean_time_us: float
    std_time_us: float
    median_time_us: float
    iqr_time_us: float
    min_time_us: float
    max_time_us: float
    # 메모리 통계 (바이트)
    peak_memory_bytes: Optional[int] = None


def benchmark_qiskit_hadamard_chain(n_qubits: int, trials: int = 10) -> BenchmarkResult:
    """Qiskit H 게이트 체인 벤치마크 (통계 + 메모리)"""
    if not QISKIT_AVAILABLE:
        return BenchmarkResult(n_qubits, n_qubits, trials, 0, 0, 0, 0, 0, 0, None)
    
    simulator = AerSimulator(method='statevector')
    times = []
    peak_memory = 0
    
    for trial in range(trials):
        # 회로 생성
        qc = QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            qc.h(i)
        qc.save_statevector()
        
        # 메모리 추적 (첫 실행만)
        if trial == 0:
            tracemalloc.start()
        
        # 실행
        start = time.perf_counter()
        job = simulator.run(qc)
        result = job.result()
        end = time.perf_counter()
        
        if trial == 0:
            current, peak = tracemalloc.get_traced_memory()
            peak_memory = peak
            tracemalloc.stop()
        
        times.append((end - start) * 1e6)  # μs
    
    times_arr = np.array(times)
    q1, median, q3 = np.percentile(times_arr, [25, 50, 75])
    
    return BenchmarkResult(
        n_qubits=n_qubits,
        n_gates=n_qubits,
        trials=trials,
        mean_time_us=statistics.mean(times),
        std_time_us=statistics.stdev(times) if len(times) > 1 else 0,
        median_time_us=median,
        iqr_time_us=q3 - q1,
        min_time_us=min(times),
        max_time_us=max(times),
        peak_memory_bytes=peak_memory
    )


def benchmark_qiskit_qft(n_qubits: int, trials: int = 10) -> BenchmarkResult:
    """Qiskit QFT 벤치마크"""
    if not QISKIT_AVAILABLE:
        return BenchmarkResult(n_qubits, 0, trials, 0, 0, 0, 0, 0, 0, None)
    
    from qiskit.circuit.library import QFT
    
    simulator = AerSimulator(method='statevector')
    times = []
    peak_memory = 0
    n_gates = 0
    
    for trial in range(trials):
        qc = QuantumCircuit(n_qubits)
        qc.h(0)  # 초기 상태
        qc.append(QFT(n_qubits), range(n_qubits))
        qc.save_statevector()
        
        if trial == 0:
            n_gates = qc.count_ops().get('h', 0) + qc.count_ops().get('cp', 0)
            tracemalloc.start()
        
        start = time.perf_counter()
        job = simulator.run(qc)
        result = job.result()
        end = time.perf_counter()
        
        if trial == 0:
            current, peak = tracemalloc.get_traced_memory()
            peak_memory = peak
            tracemalloc.stop()
        
        times.append((end - start) * 1e6)
    
    times_arr = np.array(times)
    q1, median, q3 = np.percentile(times_arr, [25, 50, 75])
    
    return BenchmarkResult(
        n_qubits=n_qubits,
        n_gates=n_gates,
        trials=trials,
        mean_time_us=statistics.mean(times),
        std_time_us=statistics.stdev(times) if len(times) > 1 else 0,
        median_time_us=median,
        iqr_time_us=q3 - q1,
        min_time_us=min(times),
        max_time_us=max(times),
        peak_memory_bytes=peak_memory
    )


def format_time(us: float) -> str:
    """시간 포맷팅"""
    if us < 1000:
        return f"{us:.1f} μs"
    elif us < 1_000_000:
        return f"{us/1000:.2f} ms"
    else:
        return f"{us/1_000_000:.2f} s"


def format_memory(bytes_val: int) -> str:
    """메모리 포맷팅"""
    if bytes_val < 1024:
        return f"{bytes_val} B"
    elif bytes_val < 1024**2:
        return f"{bytes_val/1024:.1f} KB"
    elif bytes_val < 1024**3:
        return f"{bytes_val/1024**2:.1f} MB"
    else:
        return f"{bytes_val/1024**3:.2f} GB"


def run_extended_benchmark():
    """확장 벤치마크 실행"""
    print("=" * 70)
    print("TQP vs Qiskit Extended Benchmark (PRX Quantum)")
    print("=" * 70)
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Trials: 10, Statistics: Median ± IQR")
    print()
    
    results = {
        "simulator": "Qiskit Aer",
        "method": "statevector",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "trials": 10,
        "hadamard_chain": [],
        "qft": []
    }
    
    # 확장된 큐비트 범위
    qubit_range = [4, 8, 12, 16, 18, 20, 22, 24]
    
    # 예상 실행 시간 (N=24는 ~10초 예상)
    expected_max_time_per_trial = {
        4: 0.1, 8: 0.2, 12: 0.5, 16: 2.0,
        18: 5.0, 20: 15.0, 22: 60.0, 24: 240.0
    }
    
    # H 체인 벤치마크
    print("[1/2] Hadamard Chain Benchmark")
    print("-" * 70)
    print(f"{'N':>4} | {'Median':>12} | {'IQR':>12} | {'Memory':>10} | Status")
    print("-" * 70)
    
    for n_qubits in qubit_range:
        try:
            expected_time = expected_max_time_per_trial.get(n_qubits, 300)
            start_total = time.time()
            
            result = benchmark_qiskit_hadamard_chain(n_qubits, trials=10)
            
            elapsed = time.time() - start_total
            
            # 시간 초과 체크 (예상의 2배)
            if elapsed > expected_time * 2:
                print(f"{n_qubits:>4} | {'TIMEOUT':>12} | {'-':>12} | {'-':>10} | ⚠️ Exceeded {expected_time*2:.1f}s")
                break
            
            mem_str = format_memory(result.peak_memory_bytes) if result.peak_memory_bytes else "-"
            print(f"{n_qubits:>4} | {format_time(result.median_time_us):>12} | "
                  f"{format_time(result.iqr_time_us):>12} | {mem_str:>10} | ✅")
            
            results["hadamard_chain"].append(asdict(result))
            
        except Exception as e:
            print(f"{n_qubits:>4} | ERROR: {str(e)[:40]}")
            break
    
    # QFT 벤치마크
    print()
    print("[2/2] QFT Benchmark")
    print("-" * 70)
    print(f"{'N':>4} | {'Median':>12} | {'IQR':>12} | {'Memory':>10} | Status")
    print("-" * 70)
    
    for n_qubits in qubit_range:
        try:
            expected_time = expected_max_time_per_trial.get(n_qubits, 300) * 2  # QFT는 더 느림
            start_total = time.time()
            
            result = benchmark_qiskit_qft(n_qubits, trials=10)
            
            elapsed = time.time() - start_total
            
            if elapsed > expected_time * 2:
                print(f"{n_qubits:>4} | {'TIMEOUT':>12} | {'-':>12} | {'-':>10} | ⚠️ Exceeded")
                break
            
            mem_str = format_memory(result.peak_memory_bytes) if result.peak_memory_bytes else "-"
            print(f"{n_qubits:>4} | {format_time(result.median_time_us):>12} | "
                  f"{format_time(result.iqr_time_us):>12} | {mem_str:>10} | ✅")
            
            results["qft"].append(asdict(result))
            
        except Exception as e:
            print(f"{n_qubits:>4} | ERROR: {str(e)[:40]}")
            break
    
    # 결과 저장
    output_dir = Path(__file__).parent.parent
    output_file = output_dir / "extended_benchmark_results.json"
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print()
    print("=" * 70)
    print(f"결과 저장: {output_file}")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    run_extended_benchmark()

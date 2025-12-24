#!/usr/bin/env python3
"""
TQP 공정 벤치마크 스크립트
- Qiskit warm-up 포함
- E2E vs Kernel 시간 분리
- N=14-20 Crossover 정밀 분석
"""

import time
import numpy as np
from pathlib import Path

# Qiskit imports
try:
    from qiskit import QuantumCircuit
    from qiskit_aer import Aer
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("Warning: Qiskit not available")

# 설정
WARMUP_RUNS = 10
MEASUREMENT_RUNS = 30
N_RANGE = range(14, 21)  # N=14-20 정밀 측정

def measure_qiskit_e2e(n_qubits: int, n_runs: int) -> list:
    """Qiskit 전체 E2E 시간 측정 (Python overhead 포함)"""
    if not QISKIT_AVAILABLE:
        return []
    
    backend = Aer.get_backend('statevector_simulator')
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    
    times = []
    for _ in range(n_runs):
        start = time.perf_counter_ns()
        result = backend.run(qc).result()
        end = time.perf_counter_ns()
        times.append((end - start) / 1000)  # μs
    
    return times

def measure_qiskit_kernel_only(n_qubits: int, n_runs: int) -> list:
    """Qiskit 커널 시간만 측정 (warm-up 후)"""
    if not QISKIT_AVAILABLE:
        return []
    
    backend = Aer.get_backend('statevector_simulator')
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    
    # Warm-up: 10회 실행
    for _ in range(WARMUP_RUNS):
        backend.run(qc).result()
    
    # 실제 측정
    times = []
    for _ in range(n_runs):
        start = time.perf_counter_ns()
        result = backend.run(qc).result()
        end = time.perf_counter_ns()
        times.append((end - start) / 1000)  # μs
    
    return times

def analyze_results(cold_times: list, warm_times: list) -> dict:
    """통계 분석"""
    if not cold_times or not warm_times:
        return {}
    
    cold_arr = np.array(cold_times)
    warm_arr = np.array(warm_times)
    
    return {
        "cold_median": np.median(cold_arr),
        "cold_iqr": np.percentile(cold_arr, 75) - np.percentile(cold_arr, 25),
        "warm_median": np.median(warm_arr),
        "warm_iqr": np.percentile(warm_arr, 75) - np.percentile(warm_arr, 25),
        "python_overhead": np.median(cold_arr) - np.median(warm_arr),
        "overhead_pct": (np.median(cold_arr) - np.median(warm_arr)) / np.median(cold_arr) * 100
    }

def run_crossover_analysis():
    """N=14-20 Crossover 정밀 분석"""
    print("=" * 60)
    print("TQP 공정 벤치마크: Crossover 정밀 분석")
    print("=" * 60)
    print(f"Warm-up: {WARMUP_RUNS}회, 측정: {MEASUREMENT_RUNS}회")
    print()
    
    results = {}
    
    for n in N_RANGE:
        print(f"N={n} 측정 중...")
        
        # Cold (warm-up 없이)
        cold_times = measure_qiskit_e2e(n, MEASUREMENT_RUNS)
        
        # Warm (warm-up 후)
        warm_times = measure_qiskit_kernel_only(n, MEASUREMENT_RUNS)
        
        stats = analyze_results(cold_times, warm_times)
        results[n] = stats
        
        if stats:
            print(f"  Cold: {stats['cold_median']:.1f} ± {stats['cold_iqr']:.1f} μs")
            print(f"  Warm: {stats['warm_median']:.1f} ± {stats['warm_iqr']:.1f} μs")
            print(f"  Python overhead: {stats['python_overhead']:.1f} μs ({stats['overhead_pct']:.1f}%)")
    
    return results

def save_results(results: dict, output_path: Path):
    """결과 저장"""
    with open(output_path, 'w') as f:
        f.write("# Qiskit Aer Fair Benchmark Results\n")
        f.write(f"# Warm-up: {WARMUP_RUNS}, Measurements: {MEASUREMENT_RUNS}\n")
        f.write("# N, Cold_Median(μs), Cold_IQR, Warm_Median(μs), Warm_IQR, Overhead(μs), Overhead(%)\n")
        
        for n, stats in results.items():
            if stats:
                f.write(f"{n}, {stats['cold_median']:.2f}, {stats['cold_iqr']:.2f}, ")
                f.write(f"{stats['warm_median']:.2f}, {stats['warm_iqr']:.2f}, ")
                f.write(f"{stats['python_overhead']:.2f}, {stats['overhead_pct']:.1f}\n")
    
    print(f"\n결과 저장: {output_path}")

if __name__ == "__main__":
    # 벤치마크 실행
    results = run_crossover_analysis()
    
    # 결과 저장
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)
    save_results(results, output_dir / "qiskit_fair_benchmark.csv")
    
    print("\n" + "=" * 60)
    print("벤치마크 완료!")
    print("=" * 60)

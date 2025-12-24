"""
TQP vs Qiskit 벤치마크 시각화 스크립트

스케일링 곡선, 비교 차트, 논문용 Figure 생성

사용법:
    python visualize_benchmark.py
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 한글 폰트 설정 (Windows)
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# 스타일 설정
plt.style.use('seaborn-v0_8-whitegrid')

# 벤치마크 데이터 (측정 결과)
DATA = {
    "n_qubits": [4, 8, 12, 16, 20, 22, 24],
    "tqp_us": [0.197, 2.63, 56.6, 1408, 35086, 154640, 656620],
    "qiskit_us": [600, 678, 816, 2930, 21930, 75080, 327540],
}


def plot_scaling_comparison():
    """TQP vs Qiskit 스케일링 비교 (log-log)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n = DATA["n_qubits"]
    tqp = DATA["tqp_us"]
    qiskit = DATA["qiskit_us"]
    
    # 로그 스케일 플롯
    ax.semilogy(n, tqp, 'o-', label='TQP', linewidth=2, markersize=8, color='#2196F3')
    ax.semilogy(n, qiskit, 's--', label='Qiskit Aer', linewidth=2, markersize=8, color='#FF5722')
    
    # 크로스오버 포인트 표시
    ax.axvline(x=17, color='gray', linestyle=':', alpha=0.7)
    ax.annotate('Crossover\n(N≈17)', xy=(17, 5000), fontsize=10, color='gray', ha='center')
    
    # 축 설정
    ax.set_xlabel('Number of Qubits (N)', fontsize=12)
    ax.set_ylabel('Execution Time (μs, log scale)', fontsize=12)
    ax.set_title('TQP vs Qiskit Aer: Hadamard Chain Benchmark', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.set_xticks(n)
    ax.grid(True, alpha=0.3)
    
    # 저장
    output_path = Path(__file__).parent / "benchmark_scaling.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"저장: {output_path}")
    
    return fig


def plot_speedup_ratio():
    """TQP 속도 우위 비율"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    n = DATA["n_qubits"]
    tqp = np.array(DATA["tqp_us"])
    qiskit = np.array(DATA["qiskit_us"])
    
    # 속도 비율 (Qiskit/TQP, >1이면 TQP 우위)
    speedup = qiskit / tqp
    
    # 색상 (TQP 우위: 파랑, Qiskit 우위: 주황)
    colors = ['#2196F3' if s >= 1 else '#FF5722' for s in speedup]
    
    bars = ax.bar(range(len(n)), speedup, color=colors, edgecolor='black', linewidth=0.5)
    
    # 기준선
    ax.axhline(y=1, color='black', linestyle='-', linewidth=1.5)
    
    # 레이블
    ax.set_xticks(range(len(n)))
    ax.set_xticklabels([f'N={x}' for x in n])
    ax.set_ylabel('Speedup Ratio (Qiskit / TQP)', fontsize=12)
    ax.set_title('TQP Speedup vs Qiskit Aer', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    
    # 값 표시
    for i, (bar, s) in enumerate(zip(bars, speedup)):
        height = bar.get_height()
        label = f'{s:.0f}x' if s >= 1 else f'{1/s:.1f}x slower'
        ax.annotate(label, xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords='offset points',
                   ha='center', va='bottom', fontsize=9)
    
    # 범례 추가
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2196F3', label='TQP Faster'),
        Patch(facecolor='#FF5722', label='Qiskit Faster')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # 저장
    output_path = Path(__file__).parent / "benchmark_speedup.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"저장: {output_path}")
    
    return fig


def plot_combined_figure():
    """논문용 결합 Figure"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    n = DATA["n_qubits"]
    tqp = np.array(DATA["tqp_us"])
    qiskit = np.array(DATA["qiskit_us"])
    speedup = qiskit / tqp
    
    # (a) 스케일링 비교
    ax1 = axes[0]
    ax1.semilogy(n, tqp, 'o-', label='TQP', linewidth=2, markersize=8, color='#2196F3')
    ax1.semilogy(n, qiskit, 's--', label='Qiskit Aer', linewidth=2, markersize=8, color='#FF5722')
    ax1.axvline(x=17, color='gray', linestyle=':', alpha=0.7)
    ax1.set_xlabel('Number of Qubits (N)', fontsize=12)
    ax1.set_ylabel('Execution Time (μs)', fontsize=12)
    ax1.set_title('(a) Performance Scaling', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_xticks(n)
    ax1.grid(True, alpha=0.3)
    
    # (b) 속도 비율
    ax2 = axes[1]
    colors = ['#2196F3' if s >= 1 else '#FF5722' for s in speedup]
    bars = ax2.bar(range(len(n)), speedup, color=colors, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=1, color='black', linestyle='-', linewidth=1.5)
    ax2.set_xticks(range(len(n)))
    ax2.set_xticklabels([f'{x}' for x in n])
    ax2.set_xlabel('Number of Qubits (N)', fontsize=12)
    ax2.set_ylabel('Speedup (Qiskit / TQP)', fontsize=12)
    ax2.set_title('(b) Relative Performance', fontsize=12, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 저장
    output_path = Path(__file__).parent / "benchmark_combined.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"저장: {output_path}")
    
    return fig


def main():
    print("=" * 60)
    print("TQP Benchmark Visualization")
    print("=" * 60)
    
    # 개별 차트
    plot_scaling_comparison()
    plot_speedup_ratio()
    
    # 논문용 결합 Figure
    plot_combined_figure()
    
    print("\n완료! 생성된 파일:")
    print("  - benchmark_scaling.png")
    print("  - benchmark_speedup.png")
    print("  - benchmark_combined.png (논문용)")
    
    plt.show()


if __name__ == "__main__":
    main()

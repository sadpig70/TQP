"""
TQP Time-bin/Layer 스케일링 시각화 스크립트

TQP 고유 기능인 시간 확장 스케일링 O(M), O(L) 시각화

사용법:
    python visualize_timebin.py
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

# 한글 폰트 설정 (Windows)
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

# 벤치마크 결과 (N=16 고정)
TIMEBIN_DATA = {
    "M": [1, 2, 4, 8, 16],
    "time_us": [1484, 3000, 6358, 16016, 30659],  # μs (Criterion 중앙값)
}

LAYER_DATA = {
    "L": [1, 2, 4, 8],
    "time_us": [12.3, 25.9, 51.0, 99.0],  # μs
}

COMBINED_DATA = {
    "config": ["1x1", "2x2", "4x2", "2x4", "4x4"],
    "M": [1, 2, 4, 2, 4],
    "L": [1, 2, 2, 4, 4],
    "time_us": [2.89, 10.42, 20.92, 21.48, 44.79],
}


def plot_timebin_scaling():
    """Time-bin 스케일링 O(M) 시각화"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    m = TIMEBIN_DATA["M"]
    t = np.array(TIMEBIN_DATA["time_us"]) / 1000  # ms
    
    ax.plot(m, t, 'o-', linewidth=2, markersize=10, color='#2196F3', label='TQP (N=16)')
    
    # 이론적 선형 스케일링 (M=1 기준)
    theoretical = t[0] * np.array(m)
    ax.plot(m, theoretical, '--', color='gray', alpha=0.7, label='O(M) Linear')
    
    ax.set_xlabel('Time-bins (M)', fontsize=12)
    ax.set_ylabel('Execution Time (ms)', fontsize=12)
    ax.set_title('TQP Time-bin Scaling: O(M) Linear', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xticks(m)
    ax.grid(True, alpha=0.3)
    
    # 스케일링 비율 주석
    for i in range(1, len(m)):
        ratio = t[i] / t[0]
        ax.annotate(f'{ratio:.1f}x', xy=(m[i], t[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    output_path = Path(__file__).parent / "timebin_scaling.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"저장: {output_path}")
    
    return fig


def plot_layer_scaling():
    """Layer 스케일링 O(L) 시각화"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    l = LAYER_DATA["L"]
    t = LAYER_DATA["time_us"]
    
    ax.plot(l, t, 's-', linewidth=2, markersize=10, color='#4CAF50', label='TQP (N=10)')
    
    # 이론적 선형 스케일링
    theoretical = t[0] * np.array(l)
    ax.plot(l, theoretical, '--', color='gray', alpha=0.7, label='O(L) Linear')
    
    ax.set_xlabel('Layers (L)', fontsize=12)
    ax.set_ylabel('Execution Time (μs)', fontsize=12)
    ax.set_title('TQP Layer Scaling: O(L) Linear', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xticks(l)
    ax.grid(True, alpha=0.3)
    
    # 스케일링 비율 주석
    for i in range(1, len(l)):
        ratio = t[i] / t[0]
        ax.annotate(f'{ratio:.1f}x', xy=(l[i], t[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    output_path = Path(__file__).parent / "layer_scaling.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"저장: {output_path}")
    
    return fig


def plot_combined_scaling():
    """결합 스케일링 (M × L) 시각화"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    configs = COMBINED_DATA["config"]
    m = COMBINED_DATA["M"]
    l = COMBINED_DATA["L"]
    t = COMBINED_DATA["time_us"]
    
    # 이론적 M×L 곱
    theoretical_product = np.array(m) * np.array(l)
    
    x = range(len(configs))
    
    bars = ax.bar(x, t, color='#9C27B0', edgecolor='black', linewidth=0.5, alpha=0.8)
    
    # 이론적 예상 (1x1 기준 선형)
    expected = t[0] * theoretical_product
    ax.plot(x, expected, 'o--', color='#FF5722', markersize=8, label='O(M×L) Expected')
    
    ax.set_xticks(x)
    ax.set_xticklabels([f'M={m[i]}\nL={l[i]}' for i in range(len(configs))])
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Execution Time (μs)', fontsize=12)
    ax.set_title('TQP Combined Scaling: O(M × L)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 값 표시
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    
    output_path = Path(__file__).parent / "combined_scaling.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"저장: {output_path}")
    
    return fig


def plot_temporal_combined_figure():
    """논문용 시간 확장 결합 Figure"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # (a) Time-bin Scaling
    ax1 = axes[0]
    m = TIMEBIN_DATA["M"]
    t_m = np.array(TIMEBIN_DATA["time_us"]) / 1000  # ms
    
    ax1.plot(m, t_m, 'o-', linewidth=2, markersize=10, color='#2196F3', label='Measured')
    theoretical_m = t_m[0] * np.array(m)
    ax1.plot(m, theoretical_m, '--', color='gray', alpha=0.7, label='O(M) Linear')
    ax1.set_xlabel('Time-bins (M)', fontsize=12)
    ax1.set_ylabel('Time (ms)', fontsize=12)
    ax1.set_title('(a) Time-bin Scaling (N=16)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_xticks(m)
    ax1.grid(True, alpha=0.3)
    
    # (b) Layer Scaling
    ax2 = axes[1]
    l = LAYER_DATA["L"]
    t_l = LAYER_DATA["time_us"]
    
    ax2.plot(l, t_l, 's-', linewidth=2, markersize=10, color='#4CAF50', label='Measured')
    theoretical_l = t_l[0] * np.array(l)
    ax2.plot(l, theoretical_l, '--', color='gray', alpha=0.7, label='O(L) Linear')
    ax2.set_xlabel('Layers (L)', fontsize=12)
    ax2.set_ylabel('Time (μs)', fontsize=12)
    ax2.set_title('(b) Layer Scaling (N=10)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.set_xticks(l)
    ax2.grid(True, alpha=0.3)
    
    output_path = Path(__file__).parent / "temporal_scaling_combined.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"저장: {output_path}")
    
    return fig


def main():
    print("=" * 60)
    print("TQP Temporal Scaling Visualization")
    print("=" * 60)
    
    # 스케일링 계산
    m_vals = TIMEBIN_DATA["M"]
    t_vals = TIMEBIN_DATA["time_us"]
    print(f"\n[Time-bin Scaling] N=16 고정")
    for i, (m, t) in enumerate(zip(m_vals, t_vals)):
        ratio = t / t_vals[0]
        print(f"  M={m:2d}: {t:8.0f} μs  ({ratio:.1f}x)")
    
    l_vals = LAYER_DATA["L"]
    t_l = LAYER_DATA["time_us"]
    print(f"\n[Layer Scaling] N=10 고정")
    for i, (l, t) in enumerate(zip(l_vals, t_l)):
        ratio = t / t_l[0]
        print(f"  L={l:2d}: {t:8.1f} μs  ({ratio:.1f}x)")
    
    # 시각화
    plot_timebin_scaling()
    plot_layer_scaling()
    plot_combined_scaling()
    plot_temporal_combined_figure()
    
    print("\n완료! 생성된 파일:")
    print("  - timebin_scaling.png")
    print("  - layer_scaling.png")
    print("  - combined_scaling.png")
    print("  - temporal_scaling_combined.png (논문용)")


if __name__ == "__main__":
    main()

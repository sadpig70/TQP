"""
TQP vs Spinoza (Rust-to-Rust) Benchmark Visualization Script

Generates figures for Section 3.4.2 and 5.1.1 comparing native Rust performance.
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

# Font settings for Windows/Korean support
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

# Data from Table 3.4.2 and 5.1.1
# State Initialization Time (μs)
INIT_DATA = {
    "n_qubits": [4, 8, 12, 16],
    "tqp": [0.0618, 0.1160, 1.247, 320.0],    # TQP (Rust)
    "spinoza": [0.1145, 0.1838, 1.807, 79.9], # Spinoza (Rust)
    "mps": [1.2, 4.5, 8.5, np.nan],            # MPS (chi=256) - N=16 absent/estimated
}

# Gate Operation Time (μs) comparison (Table 3.4.2)
GATE_DATA = {
    "labels": ["Hadamard (N=16)", "H-X-Z Seq (N=12)"],
    "tqp": [8.5, 1.8],
    "spinoza": [76.5, 29.6],
    "speedup": [9.0, 16.4]
}


def plot_init_comparison():
    """State Initialization Comparison (Log Scale) with MPS"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    n = INIT_DATA["n_qubits"]
    tqp = INIT_DATA["tqp"]
    spinoza = INIT_DATA["spinoza"]
    mps = INIT_DATA["mps"]
    
    # Plot lines
    ax.semilogy(n, tqp, 'o-', label='TQP (Rust)', linewidth=2.5, markersize=8, color='#2196F3')
    ax.semilogy(n, spinoza, 's--', label='Spinoza (Rust)', linewidth=2.5, markersize=8, color='#9C27B0')
    ax.semilogy(n[:3], mps[:3], '^:', label='MPS (χ=256)', linewidth=2, markersize=8, color='#FF5722')
    
    # Crossover annotation
    ax.axvline(x=14, color='gray', linestyle=':', alpha=0.5)
    ax.annotate('Crossover\n(N≈14)', xy=(14, 10), fontsize=10, color='gray', ha='center')

    ax.set_xlabel('Number of Qubits (N)', fontsize=12)
    ax.set_ylabel('Initialization Time (μs, log scale)', fontsize=12)
    ax.set_title('Rust-to-Rust: State Initialization', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_xticks(n)
    ax.grid(True, alpha=0.3, which='both')
    
    # Save
    output_path = Path(__file__).parent / "spinoza_init_comparison.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    return fig

def plot_gate_comparison():
    """Gate Operation Speedup Bar Chart"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    labels = GATE_DATA["labels"]
    tqp = GATE_DATA["tqp"]
    spinoza = GATE_DATA["spinoza"]
    
    x = np.arange(len(labels))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, tqp, width, label='TQP', color='#2196F3', edgecolor='black', linewidth=0.5)
    rects2 = ax.bar(x + width/2, spinoza, width, label='Spinoza', color='#9C27B0', edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Execution Time (μs)', fontsize=12)
    ax.set_title('Gate Operation Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add speedup labels above bars
    for i, (r1, r2, s) in enumerate(zip(rects1, rects2, GATE_DATA["speedup"])):
        height = max(r1.get_height(), r2.get_height())
        ax.annotate(f'{s}x Faster',
                    xy=(x[i], height),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', va='bottom', fontweight='bold', color='#2196F3')
                    
    # Add value labels
    ax.bar_label(rects1, padding=3, fmt='%.1f')
    ax.bar_label(rects2, padding=3, fmt='%.1f')
    
    # Save
    output_path = Path(__file__).parent / "spinoza_gate_comparison.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    return fig

def plot_combined_spinoza():
    """Combined Figure for Paper"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # (a) Init
    ax1 = axes[0]
    n = INIT_DATA["n_qubits"]
    tqp = INIT_DATA["tqp"]
    spinoza = INIT_DATA["spinoza"]
    mps = INIT_DATA["mps"]
    
    ax1.semilogy(n, tqp, 'o-', label='TQP (Rust)', linewidth=2.5, markersize=8, color='#2196F3')
    ax1.semilogy(n, spinoza, 's--', label='Spinoza (Rust)', linewidth=2.5, markersize=8, color='#9C27B0')
    ax1.semilogy(n[:3], mps[:3], '^:', label='MPS (χ=256)', linewidth=2, markersize=8, color='#FF5722')
    ax1.axvline(x=14, color='gray', linestyle=':', alpha=0.5)
    ax1.annotate('Crossover (N≈14)', xy=(14, 10), fontsize=9, color='gray', ha='center', va='bottom', rotation=90)
    ax1.set_xlabel('Number of Qubits (N)', fontsize=12)
    ax1.set_ylabel('Time (μs, log scale)', fontsize=12)
    ax1.set_title('(a) State Initialization (Rust vs Rust)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.set_xticks(n)
    ax1.grid(True, alpha=0.3)
    
    # (b) Gate
    ax2 = axes[1]
    labels = GATE_DATA["labels"]
    tqp_g = GATE_DATA["tqp"]
    spinoza_g = GATE_DATA["spinoza"]
    x = np.arange(len(labels))
    width = 0.35
    
    rects1 = ax2.bar(x - width/2, tqp_g, width, label='TQP', color='#2196F3', edgecolor='black', linewidth=0.5)
    rects2 = ax2.bar(x + width/2, spinoza_g, width, label='Spinoza', color='#9C27B0', edgecolor='black', linewidth=0.5)
    
    ax2.set_ylabel('Time (μs)', fontsize=12)
    ax2.set_title('(b) Gate Operations (SIMD Advantage)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    for i, s in enumerate(GATE_DATA["speedup"]):
        height = max(tqp_g[i], spinoza_g[i])
        ax2.annotate(f'{s}x Faster', xy=(x[i], height), xytext=(0, 3), textcoords='offset points', ha='center', fontweight='bold', color='#2196F3')

    output_path = Path(__file__).parent / "spinoza_comparison.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    return fig

if __name__ == "__main__":
    print("Generating Spinoza Comparison Figures...")
    plot_init_comparison()
    plot_gate_comparison()
    plot_combined_spinoza()
    print("Done.")

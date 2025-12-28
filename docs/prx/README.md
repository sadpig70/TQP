# TQP PRX Paper Project

**Status:** In Development  
**Branch:** prx-paper  
**Target:** PRX Quantum Submission

---

## Overview

This directory centralizes all materials for PRX Quantum paper submission.

## Directory Structure

```
docs/prx/
├── README.md               # This document
├── PRX_Submission_Package.md   # Cover letter and checklist
├── arxiv/                  # arXiv preprint version
│   ├── TQP_PRX.tex
│   └── references.bib
├── paper/                  # Main paper
│   ├── draft_v1.md         # Markdown draft (v5)
│   ├── TQP_PRX.tex         # LaTeX version
│   ├── TQP_PRX.pdf         # Compiled PDF
│   └── references.bib      # Bibliography
└── figures/                # Paper figures
```

## Milestones

| Milestone | Target Week | Completion Criteria | Status |
|-----------|-------------|---------------------|--------|
| **M1** | Week 2 | BeH₂ Hamiltonian verified, TQP integrated | ✅ Complete |
| **M2** | Week 4 | IBM hardware validation, error < 10 mHa | ✅ H₂ + LiH |
| **M3** | Week 6 | Benchmarks complete, theory draft | ✅ Complete |
| **M4** | Week 9 | PRX Quantum submission | 🔄 Ready |

## Work Tracks

| Track | Duration | Lead | Focus |
|-------|----------|------|-------|
| **A (Experiment)** | W1-W4 | J.W. Yu | BeH₂ → Hardware |
| **B (Benchmark)** | W3-W5 | AI Collab | Simulator comparison |
| **C (Theory)** | W4-W6 | J.W. Yu + AI | Resource theory proof |
| **D (Writing)** | W5-W9 | J.W. Yu | Draft → Final |

## Key Results

| Metric | Value | Status |
|--------|-------|--------|
| H₂ (2-qubit) Error | -4.2 mHa | ✅ |
| LiH (4-qubit) Error | 1.77 mHa | ✅ Near chemical accuracy |
| Spinoza Comparison | 1.4-1.9× init, 9-17× gate | ✅ |
| Crossover Point | N ≈ 14-16 qubits | ✅ |

## Reference Documents

- [TQP_Integrated_Specification_3.5.md](../TQP_Integrated_Specification_3.5.md) - TQP v3.5 Technical Spec
- [PRX_Paper_Improvement_Plan.md](../PRX_Paper_Improvement_Plan.md) - Improvement Plan v2.0
- [PRX_Submission_Package.md](PRX_Submission_Package.md) - Cover Letter

## Change Log

| Date | Description |
|------|-------------|
| 2025-12-21 | Initial structure created |
| 2025-12-22 | M1 complete, M2 H₂ validation success (ibm_torino, -4.2 mHa) |
| 2025-12-23 | LiH 4-qubit validation (1.77 mHa) |
| 2025-12-27 | v5 revision complete, submission prep |

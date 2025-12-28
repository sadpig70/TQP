# PRX Quantum Submission Package

**Title:** TQP: Temporal Quantum Processing for Efficient Quantum Simulation  
**Author:** Jeong-Wook Yu  
**Target Journal:** PRX Quantum  
**Submission Date:** 2025-12-27

---

## Cover Letter

Dear PRX Quantum Editors,

We are pleased to submit our manuscript entitled **"TQP: Temporal Quantum Processing for Efficient Quantum Simulation"** for consideration in PRX Quantum.

### Key Contributions

1. **Novel Framework:** We introduce Temporal Quantum Processing (TQP), a framework that extends quantum simulation beyond traditional spatial qubit representations by incorporating temporal dimensions. This enables O(M) linear scaling for time-bin encoded quantum systems.

2. **Rigorous Benchmarking:** We provide a comprehensive Rust-to-Rust comparison with Spinoza, demonstrating genuine algorithmic efficiencies (1.4-1.9× initialization speedup for N≤12, 9-17× gate operation speedup) independent of Python overhead artifacts.

3. **Hardware Validation:** We validate TQP on IBM Quantum hardware with H₂ (2-qubit, -4.2 mHa error) and LiH (4-qubit, 1.77 mHa error approaching chemical accuracy), demonstrating practical utility for molecular simulation.

4. **Physical Relevance:** We explicitly distinguish between "Computational View" (this paper) and "Physical View" (future work), providing platform-specific mappings for photonic, superconducting, and trapped ion systems.

### Why PRX Quantum?

TQP addresses a fundamental challenge in quantum simulation: efficient representation of temporally-structured quantum systems. Unlike existing simulators that treat time implicitly, TQP's explicit temporal dimension enables native encoding for time-bin photonic quantum computing—an increasingly important platform. We believe this work will be of broad interest to the PRX Quantum readership, particularly those working on:

- Photonic quantum computing
- Quantum simulation methodologies  
- Hybrid quantum-classical algorithms
- Temporal multiplexing in quantum systems

### Manuscript Highlights

| Metric | Value |
|--------|-------|
| Hardware Validation | H₂ (2-qubit), LiH (4-qubit) on IBM Quantum |
| Best Accuracy | 1.77 mHa (LiH, near chemical accuracy) |
| Speedup (Rust-to-Rust) | 1.4-1.9× init, 9-17× gate |
| Time-bin Scaling | O(M) linear |

### Statement of Originality

This manuscript has not been published elsewhere and is not under consideration by another journal. All authors have approved the manuscript and agree with its submission to PRX Quantum.

We look forward to your feedback.

Respectfully,

Jeong-Wook Yu

---

## Submission Checklist

### Files Ready

- [x] **Manuscript:** `draft_v1.md` (v5, 695 lines)
- [x] **LaTeX:** `TQP_PRX.tex` (needs sync with latest changes)
- [x] **BibTeX:** `references.bib` (22 entries)
- [x] **PDF:** `TQP_PRX.pdf` (674 KB, 5 pages) - needs rebuild

### Content Verification

| Section | Status | Notes |
|---------|:------:|-------|
| Abstract | ✅ | Spinoza + LiH included |
| §2.5 Physical Realization | ✅ | Computational vs Physical View |
| §3.4.2 Spinoza Comparison | ✅ | API Level + Fairness Note |
| §4.3.1 Error Budget | ✅ | 4-component decomposition |
| §4.3.3 LiH Validation | ✅ | Near chemical accuracy |
| §5.1.1 MPS Comparison | ✅ | Quantitative tables |
| §5.4 H_int Status | ✅ | Defined/Implemented/Disabled |

### Pre-submission Final Checks

- [x] LiH "chemical accuracy" → "approaching" ✅
- [x] Version v4 → v5 ✅
- [x] H₂ error -4.2 mHa standardized ✅
- [ ] LaTeX Recompilation (apply changes)
- [ ] Check Reference Links
- [ ] Review Figure/Table Captions

---

## Data Availability Statement

All benchmark data, IBM Quantum job results, and TQP source code are publicly available at:

- **Repository:** <https://github.com/sadpig70/TQP>
- **Benchmark Data:** `tqp-benchmark/data/`
- **IBM Job Logs:** `docs/prx/data/ibm_jobs.json`

---

## Suggested Reviewers

1. **Photonic QC Expert** - Time-bin encoding specialty
2. **Tensor Network Specialist** - MPS/DMRG comparison
3. **Quantum Chemistry Expert** - VQE and molecular simulation

---

## Expected Timeline

| Stage | Expected Date |
|-------|--------------|
| Submission | 2025-12-28 |
| First Decision | 2025-02-15 (6-8 weeks) |
| Revision | 2025-03-01 |
| Accept | 2025-04-01 |

---

**Document Version:** 1.0  
**Prepared by:** Antigravity AI Assistant  
**Date:** 2025-12-27

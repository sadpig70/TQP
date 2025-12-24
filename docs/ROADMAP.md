# TQP v3.4 Unified Upgrade Plan (Gantree)

**Version:** 1.0  
**Date:** 2025-12-24  
**Base Review:** TQP_Integrated_Review_Paper.md + TQP_Integrated_Review_Technical.md (8 AI Reviews Integrated)  
**Goal:** PRX Quantum Submission + Production-Ready Achievement

---

## Design Principles

1. **PRX Submission First** - Consider paper vs. technical dependencies
2. **Top-Down BFS** Design
3. **Atomic Nodes** - Breakdown until implementable in 15 mins by AI
4. **Separate Roots** for subtrees deeper than 5 levels

---

## Dependency Analysis

```
[Paper Tasks]            [Technical Tasks]          [Dependency]
─────────────────────────────────────────────────────────────
Fair Benchmark    ←──── Separate E2E/Kernel       (Tech→Paper)
Molecule Validation  ←──── BeH₂ IBM Execution     (Tech→Paper)
Data Availability ←──── GitHub Dataset Cleanup    (Tech→Paper)
Expand References ───── (Independent)             (Paper Only)
Sparse Memory     ───── (Independent)             (Tech Only)
GPU PoC           ───── (Independent)             (Tech Only)
```

---

## Unified Work Tree

```
TQP_v3.4_Unified_Upgrade // v3.4 Unified Upgrade (In Progress)
    Phase_0_Immediate // Immediate Fixes - Pre-PRX (1 Week) (Completed)
        P0_Paper_Abstract // Fix Paper Abstract (Completed)
            Add_Python_Overhead_Caveat // Add "includes Python overhead" (Completed)
            Add_Crossover_Mention // Mention N≥17 crossover (Completed)
            Tone_Down_Speedup // Rephrase "2-3000x" to "end-to-end" (Completed)
        P0_Tech_Benchmark_Caveat // Add Caveat to Technical Spec (Completed)
            Add_Caveat_Text // Add Section 6 Benchmark Note (Completed)
            Fix_M_Scaling // Fix "Near-linear (R²=0.98)" phrasing (Completed)
        P0_Tent_Definition // T̂_ent f(m) Definition (Completed)
            Verify_fm_Unitarity // Mathematical proof of f(m) unitarity (Completed)
            Define_fm_XOR // f(m) = m XOR (1 << (m % n)) (Completed)
            Add_N2M2_Example // Add N=2, M=2 Example (Completed)
            Prove_Unitarity // Include Unitarity Proof in Paper (Completed)
        P0_Hardware_Accuracy // Fix Hardware Accuracy Interpretation (Completed)
            Get_FCI_Reference // Cite FCI Value (NIST/Literature) (Completed)
            Change_Ref_to_FCI // Change HF→FCI Baseline (Completed)
            Add_ChemAcc_Warning // State >1.6mHa explicitly (Completed)
        P0_Data_Availability // Data Availability Statement (Completed)
            Write_DAS // Write Statement (Completed)
            Prepare_GitHub_Raw // Prepare raw_data folder (Completed)
        P0_Hint_Clarify // Clarify H_int Usage (Completed)
            Check_Usage // Check H_int usage in code (Completed)
            Move_Future_Work // Move to "future work" if unused (Completed)
        P0_Backend_Unify // Unify Backend (Completed)
            Update_Appendix // Unify to ibm_torino (Completed)
    Phase_1_PRX_Ready // PRX Preparation (3 Weeks) (In Progress)
        P1_References_Expand // Expand References (Completed)
            Add_21_References // Expand 3 → 21 references (Completed)
        P1_Fair_Benchmark // Design & Run Fair Benchmark (Completed)
            Design_Fair_Protocol // Fair Comparison Protocol (Completed)
                Option_B_Warmup_Qiskit // Measure after 10 warm-up runs (Completed)
            Separate_E2E_Kernel // Separate E2E vs Kernel Time (Completed)
                Measure_Python_Overhead // Measure Python Overhead (Completed)
            Run_Fair_Benchmark // Run Benchmark (Completed)
                Run_N14_to_N20 // 30 reps for N=14-20 (Completed)
            Update_Paper // Update Paper (Completed)
                Add_Section_3_4_1 // Add Methods Python Overhead (Completed)
                Add_Section_5_3_1 // Add Discussion Interpretation (Completed)
        P1_Crossover_Precision // Refine Crossover Analysis (Completed)
            Measure_N14_to_N20 // 30 runs each for N=14-20 (Completed)
            Report_Results // Save CSV Results (Completed)
        P1_Time_Concept // Clarify "Time as Resource" (Designing)
            Separate_Physics_DS // Physics Model vs Data Structure Separation (Designing)
            Add_Photonic_Relation // Relation to time-bin photonic QC (Designing)
        P1_Tensor_Network // Tensor Network Connection (Designing)
            Define_MPS_Eq // TQP = MPS(χ≤2^M) Equation (Designing)
        P1_PRX_Submit // PRX Final Submission Prep (In Progress)
            Prepare_SM // Prepare Supplementary Material (In Progress)
            Prepare_Rebuttal_Template // Reviewer Response Template (In Progress)
            Final_Review // Final Review (In Progress)
            Submit_Draft // Submit Draft (Designing)
    Phase_2_Extension // Tech Extension - Post-Submission (2 Months) (In Progress)
        P2_External_Dependency // External Dependency Check (Completed)
            Check_IBM_Quota // Check IBM Quota (Completed: Insufficient)
            Check_PySCF_Version // PySCF Version Compatibility (Completed)
        P2_BeH2_Validation // BeH₂ 14-qubit Validation (Completed: Simulation Only)
            Generate_BeH2_Hamiltonian // Generate 14-qubit 666 terms (Completed)
            Add_to_Paper // Add Section 4.4, SM S5 (Completed)
            Run_IBM_Hardware // Run IBM Hardware (On Hold: Quota Insufficient)
        P2_Sparse_Memory // Sparse Memory Implementation (Decomposed)
        P2_Error_Handling // Systematize Error Handling (Decomposed)
    Phase_3_Commercialization // Commercialization Prep (3 Months) (Designing)
        P3_Resource_Estimate // Resource Estimation (Designing)
            Estimate_FTE // Estimated FTE (Designing)
            Estimate_Cost // Estimated Cost (Designing)
        P3_Milestone_Checkpoint // Bi-weekly Checkpoint (Designing)
        P3_GPU_PoC // GPU PoC Implementation (Decomposed)
            GPU_Requirements // CUDA CC 7.0+, VRAM 8GB+ (Designing)
        P3_Package_Deploy // PyPI/Crates Deployment (Decomposed)
        P3_Multi_Backend // Multi-backend (On Hold)
```

---

## Decomposed Subtrees

### P2_BeH2_Validation (Decomposed)

```
P2_BeH2_Validation // BeH₂ 6-qubit Hardware Validation (Designing)
    Generate_BeH2_Hamiltonian // Generate Hamiltonian via PySCF (Designing)
        Setup_PySCF // Setup PySCF Env (Designing)
        Run_CASSCF // CASSCF(3,2) Calculation (Designing)
        Convert_JordanWigner // Jordan-Wigner Transform (Designing)
    Run_IBM_Validation // IBM Hardware Execution (Designing)
        Prepare_QASM // Generate QASM 3.0 (Designing)
        Submit_Estimator // Submit Estimator V2 (Designing)
        Collect_Results // Collect Results (Designing)
    Apply_Error_Mitigation // Apply Error Mitigation (Designing)
        Apply_ZNE // Zero Noise Extrapolation (Designing)
        Report_Mitigated_Error // Report Mitigated Error (Designing)
    Update_Paper_Results // Update Paper Results (Designing)
```

[... Other subtrees omitted for brevity, identical to Korean version ...]

---

## Schedule (Unified)

| Phase | Duration | Tasks | Completion Criteria | Dependencies |
|-------|----------|-------|---------------------|--------------|
| **Phase 0** | 1 Week | 7 Immediate Fixes | PRX Submission Minimum | None |
| **Phase 1** | 3 Weeks | PRX Prep | Paper Submission | Phase 0 |
| **Phase 2** | 2 Months | Tech Extension | N=30, BeH₂ Validation | Phase 1 |
| **Phase 3** | 3 Months | Commercialization | GPU 10-50x | Phase 2 |

### Timeline

```
Jan 2025 W1    ── Phase 0: Immediate Fixes (7 Items)
Jan 2025 W2-4  ── Phase 1: PRX Prep
Jan 2025 End   ── ★ PRX Quantum Submission
Feb-Mar 2025   ── Phase 2: Tech Extension
Apr-Jun 2025   ── Phase 3: Commercialization
Q3 2025        ── TQP v1.0 Official Launch
```

---

## Success Metrics

| Metric | Current v3.3 | Post-Phase 0 | Post-Phase 1 | Post-Phase 2 | Post-Phase 3 |
|--------|--------------|--------------|--------------|--------------|--------------|
| PRX Score | 43/100 | 55/100 | 75/100 | 80/100 | 85/100 |
| References | 3 | 3 | 20 | 20 | 25 |
| Benchmark Fairness | ❌ | Explicit | ✅ | ✅ | ✅ |
| HW Validated Molecules | 1 | 1 | 1 | 2 | 2 |
| Max Qubits | N=24 | N=24 | N=24 | N=30 | N=34 |
| GPU Accel | - | - | - | - | 10-50x |
| API Docs | None | None | None | rustdoc | SDK |

---

## Risk Management

| Risk | Impact | Mitigation Strategy |
|------|--------|---------------------|
| Fair Benchmark Degradation | PRX Rejection | Reposition TQP Strengths (Time-bin specific) |
| BeH₂ Hardware Failure | Paper Weaker | Replace with H₂ multi-bond lengths |
| Sparse Implementation Delay | Phase 2 Delay | Use external lib (nalgebra-sparse) |
| GPU Kernel Conflicts | Phase 3 Delay | Direct cuQuantum usage (Abandon 3D) |

---

## Verification Plan

| Phase | Method |
|-------|--------|
| Phase 0 | Doc Review (8 AI Re-reviews) |
| Phase 1 | `cargo test`, Rerun Benchmarks, CI Pass |
| Phase 2 | N=30 Test, BeH₂ IBM Run, `cargo doc` |
| Phase 3 | GPU Benchmark, `pip install tqp` Check |

---

## Priority Summary

```
🔴 P0 (Immediate): Abstract, f(m), Accuracy, Data Availability
🟠 P1 (3 Weeks): Fair Benchmark, Crossover, References
🟡 P2 (2 Months): Sparse, BeH₂, Error Handling, API
🟢 P3 (3 Months): GPU, Packing, Multi-backend
```

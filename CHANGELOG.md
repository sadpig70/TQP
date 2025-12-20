# Changelog

All notable changes to TQP will be documented in this file.

## [1.2.0] - 2025-12-20

### Added

- **tqp-ibm**: VQE Correlation Energy Recovery module (`vqe_correlation.rs`)
  - Double excitation ansatz QASM generation (SimpleCNOT, UCCSD Givens)
  - θ parameter sweep with configurable grid
  - Simulation mode for theoretical validation
  - Hardware sweep support (async feature)
- **tqp-ibm**: BeH₂ 6-Qubit Verification module (`beh2_verification.rs`)
  - CASSCF(3,2) based 6-qubit Hamiltonian
  - HF state QASM generation for |000011⟩
  - BeH2Verifier with energy validation
- **tqp-ibm**: BeH₂ Hamiltonian constants
  - E_HF = -15.5614 Ha, E_FCI = -15.5952 Ha
  - Correlation energy = 33.8 mHa

### Changed

- Improved error handling with `ApiErrorStructured` variant
- Added `hamiltonian()` getter to `VqeCorrelationSweep`
- Moved legacy examples to `examples/_legacy/`

### Fixed

- Removed unused imports (`HashMap`, `Deserialize`)
- Removed unnecessary `u64` casts
- Fixed `JobFailedWithId` error variant usage

### Tests

- VQE Correlation: 7 unit tests
- BeH₂ Verification: 8 unit tests
- Total: 53 library tests passing

---

## [1.1.0] - 2025-12-07

### Added

- **tqp-ibm**: Estimator V2 primitive support (REST API)
  - Direct expectation value measurement without manual basis rotation
  - Multi-observable batch measurement in single job
  - Service-CRN authentication for IBM Cloud Quantum
- **tqp-ibm**: LiH 4-qubit full Hamiltonian (including 4-body XXXX, YYYY terms)
- **tqp-ibm**: Observable collection API for Hamiltonian construction

### Hardware Validation (2025-12-07)

- **Backend**: ibm_fez (156-qubit Heron r2)
- **LiH HF state**: 1.77 mHa error ✅ (chemical accuracy achieved)
- **H₂ (2-qubit)**: Z-term measurement error < 1%
- **BeH₂ (6-qubit)**: Z-term measurement error < 0.5%

### API Changes

- `EstimatorExecutor::with_service_crn()` - Set IBM Cloud CRN
- Observable format changed from dict to string array
- Authentication header: `apikey` (was `Bearer`)

### Fixed

- LiH Hamiltonian coefficients (corrected from literature)
- 4-body term signs (XXXX: +0.0453, XXYY: -0.0453)
- HF reference energy (-7.8962 Ha, was -7.655 Ha)

## [1.0.0] - 2025-12-07

### Added

- **tqp-core**: Quantum state simulation with SIMD optimization
- **tqp-core**: VQE with UCCSD ansatz for molecular simulation
- **tqp-core**: QAOA for combinatorial optimization (MaxCut)
- **tqp-core**: Multiple optimizers (COBYLA, SPSA, Adam, QNG)
- **tqp-core**: Noise models (depolarizing, amplitude damping, phase damping)
- **tqp-core**: ZNE (Zero Noise Extrapolation) error mitigation
- **tqp-ibm**: IBM Quantum REST API integration
- **tqp-ibm**: ISA transpilation for Heron (CZ) and Eagle (ECR) processors
- **tqp-ibm**: Readout Error Mitigation (MEM) with tensor-product model
- **tqp-ibm**: Job submission, monitoring, and result parsing

### Hardware Validation

- H₂ molecule: 12.1 mHa error (98.9% accuracy)
- LiH molecule: 5.7 mHa error (99.9% accuracy)
- Bell state fidelity: 97.4% with MEM

### Performance

- 20-qubit simulation: ~15ms initialization
- VQE convergence: ~50ms for 2-qubit systems
- MEM improvement: 49-73% error reduction

## [0.1.0] - 2025-11-01

### Added

- Initial project structure
- Basic quantum state representation
- Single and two-qubit gates

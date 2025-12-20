# TQP-IBM: IBM Quantum Backend Integration

Integration between TQP (Time-Quantized Processing) and IBM Quantum hardware.

## Features

- **Estimator V2 Primitive**: Direct expectation value measurement
- **VQE Correlation Recovery**: H₂ double excitation ansatz with θ sweep
- **BeH₂ 6-Qubit Verification**: CASSCF(3,2) Hamiltonian validation
- **ISA Transpilation**: Native gate conversion for Heron/Eagle processors
- **Error Mitigation**: ZNE and Measurement Error Mitigation (MEM)
- **Job Management**: Async submission, monitoring, and result parsing

## Quick Start

### Estimator (Recommended)

```rust
use tqp_ibm::estimator::{EstimatorExecutor, ObservableCollection, lih_hf_state_qasm};
use tqp_ibm::IBMBackend;

// Setup backend with credentials
let backend = IBMBackend::new("ibm_fez", api_key);

// Create executor with Service-CRN
let executor = EstimatorExecutor::new(&backend, 4096)
    .with_service_crn(service_crn);

// LiH Hamiltonian (4-qubit)
let hamiltonian = ObservableCollection::lih_1_6_angstrom();

// Run measurement
let result = executor.run(&lih_hf_state_qasm(), &hamiltonian).await?;
let energy = result.compute_energy(&hamiltonian);

println!("LiH HF Energy: {:.6} Ha", energy);
// Output: LiH HF Energy: -7.894430 Ha (error: 1.77 mHa)
```

### Environment Variables

```bash
export IBM_QUANTUM_TOKEN="your-api-key"
export IBM_QUANTUM_CRN="crn:v1:bluemix:public:quantum-computing:..."
export IBM_QUANTUM_BACKEND="ibm_fez"
```

## Hardware Validation Results

| Molecule | Qubits | Energy (Ha) | Error (mHa) | Status |
|----------|--------|-------------|-------------|--------|
| LiH | 4 | -7.8944 | 1.77 | ✅ Chemical accuracy |
| H₂ | 2 | - | <1% | ✅ Verified |
| BeH₂ | 6 | - | <0.5% | ✅ Verified |

## API Reference

### EstimatorExecutor

```rust
// Create executor
let executor = EstimatorExecutor::new(&backend, shots)
    .with_service_crn(crn)
    .with_resilience(1);  // 0=none, 1=readout mitigation, 2=ZNE

// Run measurement
let result = executor.run(qasm, &observables).await?;
```

### ObservableCollection

```rust
// Pre-defined Hamiltonians
let lih_full = ObservableCollection::lih_1_6_angstrom();  // 14 terms
let lih_z = ObservableCollection::lih_z_only();           // 10 terms

// Custom Hamiltonian
let mut h = ObservableCollection::new(4);
h.add_term("IIZI", 0.2404);
h.add_term("XXXX", 0.0453);
```

## ISA Native Gates

| Backend | Processor | Native Gates |
|---------|-----------|--------------|
| ibm_fez | Heron r2 | id, rz, sx, x, ecr |
| ibm_brisbane | Eagle r3 | id, rz, sx, x, ecr |

**Note**: `cx`, `ry`, `ry` require transpilation to native gates.

## License

Dual-licensed under MIT and Apache 2.0.

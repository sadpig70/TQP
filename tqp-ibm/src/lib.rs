//! # TQP-IBM: IBM Quantum Backend Integration
//!
//! This crate provides integration between TQP (Time-Quantized Processing)
//! and IBM Quantum hardware through the Qiskit Runtime API.
//!
//! ## Features
//!
//! - **QASM Transpilation**: Convert TQP circuits to OpenQASM 3.0
//! - **Job Management**: Submit, monitor, and retrieve quantum jobs
//! - **Backend Selection**: Query and select optimal IBM backends
//! - **Error Handling**: Robust handling of network and hardware errors
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use tqp_ibm::{IBMBackend, QASMTranspiler, JobManager};
//!
//! // Initialize backend with API token
//! let backend = IBMBackend::new("your_api_token")?;
//!
//! // List available backends
//! let backends = backend.list_backends().await?;
//!
//! // Transpile TQP circuit to QASM
//! let qasm = QASMTranspiler::transpile(&circuit)?;
//!
//! // Submit job
//! let job = JobManager::submit(&backend, &qasm, 8192).await?;
//! ```

pub mod backend;
pub mod beh2_verification;
pub mod credentials;
pub mod error;
pub mod estimator;
pub mod jobs;
pub mod transpiler;
pub mod vqe_correlation;

// Re-exports
pub use backend::{BackendInfo, BackendStatus, IBMBackend};
pub use beh2_verification::{BeH2VerificationResult, BeH2Verifier};
pub use credentials::{Credentials, CredentialsManager};
pub use error::{IBMError, Result};
pub use estimator::{EstimatorExecutor, EstimatorResult, Observable, ObservableCollection};
pub use jobs::{Job, JobManager, JobResult, JobStatus};
pub use transpiler::{GateMapper, QASMBuilder, QASMTranspiler};
pub use vqe_correlation::{VqeCorrelationSweep, VqeSweepConfig, VqeSweepResult};

/// IBM Quantum API base URL
pub const IBM_QUANTUM_API_URL: &str = "https://api.quantum.ibm.com";

/// Default number of shots
pub const DEFAULT_SHOTS: u32 = 8192;

/// Maximum wait time for job completion (seconds)
pub const MAX_WAIT_TIME: u64 = 3600;

/// Poll interval for job status (seconds)
pub const POLL_INTERVAL: u64 = 5;

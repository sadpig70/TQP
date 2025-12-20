//! IBM Quantum backend management
//!
//! Provides functionality to:
//! - List available backends
//! - Query backend properties
//! - Select optimal backend for circuit

use crate::credentials::Credentials;
use crate::error::{IBMError, Result};
use crate::IBM_QUANTUM_API_URL;
use reqwest::Client;
use serde::{Deserialize, Serialize};

/// IBM Quantum backend
pub struct IBMBackend {
    /// HTTP client
    client: Client,

    /// Credentials
    credentials: Credentials,

    /// Selected backend name
    backend_name: Option<String>,

    /// Cached backend info
    backend_info: Option<BackendInfo>,
}

/// Backend status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum BackendStatus {
    /// Backend is online and accepting jobs
    Online,

    /// Backend is offline for maintenance
    Offline,

    /// Backend is paused
    Paused,

    /// Backend status is unknown
    #[default]
    Unknown,
}

/// Backend information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendInfo {
    /// Backend name
    pub name: String,

    /// Number of qubits
    pub n_qubits: usize,

    /// Backend status
    #[serde(default)]
    pub status: BackendStatus,

    /// Pending jobs in queue
    #[serde(default)]
    pub pending_jobs: usize,

    /// Supported gates
    #[serde(default)]
    pub basis_gates: Vec<String>,

    /// Is simulator
    #[serde(default)]
    pub simulator: bool,

    /// Maximum shots
    #[serde(default)]
    pub max_shots: u32,

    /// Maximum circuits per job
    #[serde(default)]
    pub max_circuits: usize,

    /// Description
    #[serde(default)]
    pub description: String,
}

impl Default for BackendInfo {
    fn default() -> Self {
        Self {
            name: String::new(),
            n_qubits: 0,
            status: BackendStatus::Unknown,
            pending_jobs: 0,
            basis_gates: vec![
                "cx".into(),
                "id".into(),
                "rz".into(),
                "sx".into(),
                "x".into(),
            ],
            simulator: false,
            max_shots: 8192,
            max_circuits: 100,
            description: String::new(),
        }
    }
}

impl BackendInfo {
    /// Check if backend can run a circuit with given qubit count
    pub fn can_run(&self, n_qubits: usize) -> bool {
        self.status == BackendStatus::Online && n_qubits <= self.n_qubits
    }

    /// Check if a gate is supported
    pub fn supports_gate(&self, gate: &str) -> bool {
        self.basis_gates.iter().any(|g| g == gate)
    }
}

/// Backend response from API
#[derive(Debug, Deserialize)]
struct BackendResponse {
    backends: Vec<BackendData>,
}

#[derive(Debug, Deserialize)]
struct BackendData {
    name: String,
    #[serde(default)]
    n_qubits: Option<usize>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    pending_jobs: Option<usize>,
    #[serde(default)]
    basis_gates: Option<Vec<String>>,
    #[serde(default)]
    simulator: Option<bool>,
    #[serde(default)]
    max_shots: Option<u32>,
    #[serde(default)]
    description: Option<String>,
}

impl From<BackendData> for BackendInfo {
    fn from(data: BackendData) -> Self {
        Self {
            name: data.name,
            n_qubits: data.n_qubits.unwrap_or(0),
            status: match data.status.as_deref() {
                Some("online") => BackendStatus::Online,
                Some("offline") => BackendStatus::Offline,
                Some("paused") => BackendStatus::Paused,
                _ => BackendStatus::Unknown,
            },
            pending_jobs: data.pending_jobs.unwrap_or(0),
            basis_gates: data.basis_gates.unwrap_or_default(),
            simulator: data.simulator.unwrap_or(false),
            max_shots: data.max_shots.unwrap_or(8192),
            max_circuits: 100,
            description: data.description.unwrap_or_default(),
        }
    }
}

impl IBMBackend {
    /// Create new IBM backend with credentials
    pub fn new(credentials: Credentials) -> Result<Self> {
        credentials.validate()?;

        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()?;

        Ok(Self {
            client,
            credentials,
            backend_name: None,
            backend_info: None,
        })
    }

    /// Create backend from API token
    pub fn from_token(token: impl Into<String>) -> Result<Self> {
        let credentials = Credentials::new(token);
        Self::new(credentials)
    }

    /// List available backends
    pub async fn list_backends(&self) -> Result<Vec<BackendInfo>> {
        let url = format!("{}/backends", IBM_QUANTUM_API_URL);

        let response = self
            .client
            .get(&url)
            .header("Authorization", self.credentials.auth_header())
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let text = response.text().await.unwrap_or_default();
            return Err(IBMError::ApiErrorStructured {
                code: status,
                message: text,
            });
        }

        let data: BackendResponse = response.json().await?;
        Ok(data.backends.into_iter().map(BackendInfo::from).collect())
    }

    /// Get information about a specific backend
    pub async fn get_backend(&self, name: &str) -> Result<BackendInfo> {
        let backends = self.list_backends().await?;

        backends
            .into_iter()
            .find(|b| b.name == name)
            .ok_or_else(|| IBMError::BackendNotFound(name.to_string()))
    }

    /// Select a backend by name
    pub async fn select(&mut self, name: &str) -> Result<&BackendInfo> {
        let info = self.get_backend(name).await?;

        if info.status != BackendStatus::Online {
            return Err(IBMError::BackendUnavailable(
                name.to_string(),
                format!("Status: {:?}", info.status),
            ));
        }

        self.backend_name = Some(name.to_string());
        self.backend_info = Some(info);

        Ok(self.backend_info.as_ref().unwrap())
    }

    /// Select optimal backend for given requirements
    pub async fn select_optimal(
        &mut self,
        n_qubits: usize,
        prefer_simulator: bool,
    ) -> Result<&BackendInfo> {
        let backends = self.list_backends().await?;

        // Filter backends that can run the circuit
        let mut candidates: Vec<_> = backends
            .into_iter()
            .filter(|b| b.can_run(n_qubits))
            .filter(|b| b.simulator == prefer_simulator)
            .collect();

        if candidates.is_empty() {
            // Try without simulator preference
            let backends = self.list_backends().await?;
            candidates = backends
                .into_iter()
                .filter(|b| b.can_run(n_qubits))
                .collect();
        }

        if candidates.is_empty() {
            return Err(IBMError::BackendNotFound(format!(
                "No backend available for {} qubits",
                n_qubits
            )));
        }

        // Sort by pending jobs (prefer less busy backends)
        candidates.sort_by_key(|b| b.pending_jobs);

        let best = candidates.remove(0);
        self.backend_name = Some(best.name.clone());
        self.backend_info = Some(best);

        Ok(self.backend_info.as_ref().unwrap())
    }

    /// Get selected backend name
    pub fn selected_backend(&self) -> Option<&str> {
        self.backend_name.as_deref()
    }

    /// Get selected backend info
    pub fn backend_info(&self) -> Option<&BackendInfo> {
        self.backend_info.as_ref()
    }

    /// Get credentials
    pub fn credentials(&self) -> &Credentials {
        &self.credentials
    }

    /// Get HTTP client
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// Create mock backend for testing
    #[cfg(feature = "mock")]
    pub fn mock(n_qubits: usize) -> Self {
        let credentials = Credentials::new(format!("mock_token_{}", "x".repeat(50)));
        let client = Client::new();

        let backend_info = BackendInfo {
            name: "mock_backend".to_string(),
            n_qubits,
            status: BackendStatus::Online,
            pending_jobs: 0,
            basis_gates: vec![
                "cx".into(),
                "id".into(),
                "rz".into(),
                "sx".into(),
                "x".into(),
            ],
            simulator: true,
            max_shots: 8192,
            max_circuits: 100,
            description: "Mock backend for testing".to_string(),
        };

        Self {
            client,
            credentials,
            backend_name: Some("mock_backend".to_string()),
            backend_info: Some(backend_info),
        }
    }
}

/// Common IBM Quantum backends
pub mod backends {
    /// IBM Manila (5 qubits) - Free tier
    pub const IBM_MANILA: &str = "ibm_manila";

    /// IBM Nairobi (7 qubits)
    pub const IBM_NAIROBI: &str = "ibm_nairobi";

    /// IBM Lagos (7 qubits)
    pub const IBM_LAGOS: &str = "ibm_lagos";

    /// IBM Perth (7 qubits)
    pub const IBM_PERTH: &str = "ibm_perth";

    /// IBM Brisbane (127 qubits)
    pub const IBM_BRISBANE: &str = "ibm_brisbane";

    /// IBM Sherbrooke (127 qubits)
    pub const IBM_SHERBROOKE: &str = "ibm_sherbrooke";

    /// QASM Simulator (up to 32 qubits)
    pub const QASM_SIMULATOR: &str = "ibmq_qasm_simulator";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_info_default() {
        let info = BackendInfo::default();
        assert_eq!(info.n_qubits, 0);
        assert_eq!(info.status, BackendStatus::Unknown);
        assert!(!info.basis_gates.is_empty());
    }

    #[test]
    fn test_backend_info_can_run() {
        let mut info = BackendInfo::default();
        info.n_qubits = 5;
        info.status = BackendStatus::Online;

        assert!(info.can_run(4));
        assert!(info.can_run(5));
        assert!(!info.can_run(6));
    }

    #[test]
    fn test_backend_info_can_run_offline() {
        let mut info = BackendInfo::default();
        info.n_qubits = 5;
        info.status = BackendStatus::Offline;

        assert!(!info.can_run(4));
    }

    #[test]
    fn test_backend_info_supports_gate() {
        let info = BackendInfo::default();
        assert!(info.supports_gate("cx"));
        assert!(info.supports_gate("rz"));
        assert!(!info.supports_gate("ccx"));
    }

    #[test]
    fn test_backend_status_default() {
        assert_eq!(BackendStatus::default(), BackendStatus::Unknown);
    }
}

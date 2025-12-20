use crate::state::TQPState;

/// Trait representing a Quantum Hardware Backend.
/// This allows TQP to interface with different backends (Simulator, Real QPU, etc.).
pub trait QuantumBackend {
    /// Submits a circuit/operation to the backend.
    fn submit_job(&self, job_id: &str, state: &TQPState) -> Result<String, String>;

    /// Retrieves the status of a job.
    fn get_status(&self, job_id: &str) -> String;

    /// Retrieves the result of a job (e.g., measurement counts).
    fn get_result(&self, job_id: &str) -> Result<Vec<usize>, String>;
}

/// A Mock Backend for testing and development.
/// Simulates a backend that always succeeds immediately.
pub struct MockBackend {
    pub name: String,
}

impl MockBackend {
    pub fn new(name: &str) -> Self {
        MockBackend {
            name: name.to_string(),
        }
    }
}

impl QuantumBackend for MockBackend {
    fn submit_job(&self, job_id: &str, _state: &TQPState) -> Result<String, String> {
        println!("[MockBackend: {}] Job {} submitted.", self.name, job_id);
        Ok("SUBMITTED".to_string())
    }

    fn get_status(&self, job_id: &str) -> String {
        println!(
            "[MockBackend: {}] Checking status for {}.",
            self.name, job_id
        );
        "COMPLETED".to_string()
    }

    fn get_result(&self, job_id: &str) -> Result<Vec<usize>, String> {
        println!(
            "[MockBackend: {}] Getting result for {}.",
            self.name, job_id
        );
        // Return some dummy measurement results
        Ok(vec![0, 1, 0, 1])
    }
}

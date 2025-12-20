//! IBM Quantum job management
//!
//! Provides functionality to:
//! - Submit quantum jobs
//! - Monitor job status
//! - Retrieve and parse results

use crate::backend::IBMBackend;
use crate::error::{IBMError, Result};
use crate::{IBM_QUANTUM_API_URL, MAX_WAIT_TIME, POLL_INTERVAL};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::time::sleep;

/// Job status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum JobStatus {
    /// Job is queued
    Queued,

    /// Job is running
    Running,

    /// Job completed successfully
    Completed,

    /// Job failed
    Failed,

    /// Job was cancelled
    Cancelled,

    /// Unknown status
    #[serde(other)]
    Unknown,
}

impl JobStatus {
    /// Check if job is in a terminal state
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            JobStatus::Completed | JobStatus::Failed | JobStatus::Cancelled
        )
    }

    /// Check if job succeeded
    pub fn is_success(&self) -> bool {
        matches!(self, JobStatus::Completed)
    }
}

/// Job result from IBM Quantum
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobResult {
    /// Measurement counts (bitstring → count)
    pub counts: HashMap<String, u64>,

    /// Total number of shots
    pub shots: u64,

    /// Execution time in milliseconds
    #[serde(default)]
    pub execution_time_ms: u64,

    /// Backend name
    #[serde(default)]
    pub backend: String,

    /// Job ID
    #[serde(default)]
    pub job_id: String,
}

impl JobResult {
    /// Create empty result
    pub fn new() -> Self {
        Self {
            counts: HashMap::new(),
            shots: 0,
            execution_time_ms: 0,
            backend: String::new(),
            job_id: String::new(),
        }
    }

    /// Get probability of a bitstring
    pub fn probability(&self, bitstring: &str) -> f64 {
        if self.shots == 0 {
            return 0.0;
        }
        self.counts.get(bitstring).copied().unwrap_or(0) as f64 / self.shots as f64
    }

    /// Get most likely bitstring
    pub fn most_likely(&self) -> Option<(&str, u64)> {
        self.counts
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(bs, &count)| (bs.as_str(), count))
    }

    /// Convert counts to probability distribution
    pub fn probabilities(&self) -> HashMap<String, f64> {
        if self.shots == 0 {
            return HashMap::new();
        }
        self.counts
            .iter()
            .map(|(bs, &count)| (bs.clone(), count as f64 / self.shots as f64))
            .collect()
    }

    /// Compute expectation value of Z operator on specified qubit
    pub fn expectation_z(&self, qubit: usize) -> f64 {
        let mut exp = 0.0;
        let total = self.shots as f64;

        for (bitstring, &count) in &self.counts {
            // Bitstring is typically in reverse order (q[0] is rightmost)
            let chars: Vec<char> = bitstring.chars().rev().collect();
            if qubit < chars.len() {
                let bit = chars[qubit];
                let sign = if bit == '0' { 1.0 } else { -1.0 };
                exp += sign * count as f64 / total;
            }
        }

        exp
    }

    /// Compute expectation value of ZZ operator on specified qubits
    pub fn expectation_zz(&self, qubit1: usize, qubit2: usize) -> f64 {
        let mut exp = 0.0;
        let total = self.shots as f64;

        for (bitstring, &count) in &self.counts {
            let chars: Vec<char> = bitstring.chars().rev().collect();
            if qubit1 < chars.len() && qubit2 < chars.len() {
                let bit1 = chars[qubit1];
                let bit2 = chars[qubit2];
                let sign1 = if bit1 == '0' { 1.0 } else { -1.0 };
                let sign2 = if bit2 == '0' { 1.0 } else { -1.0 };
                exp += sign1 * sign2 * count as f64 / total;
            }
        }

        exp
    }
}

impl Default for JobResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Job handle for tracking submitted jobs
#[derive(Debug, Clone)]
pub struct Job {
    /// Job ID
    pub id: String,

    /// Backend name
    pub backend: String,

    /// Current status
    pub status: JobStatus,

    /// Creation time
    pub created_at: chrono::DateTime<chrono::Utc>,

    /// Number of shots
    pub shots: u32,
}

impl Job {
    /// Create a new job handle
    pub fn new(id: String, backend: String, shots: u32) -> Self {
        Self {
            id,
            backend,
            status: JobStatus::Queued,
            created_at: chrono::Utc::now(),
            shots,
        }
    }
}

/// Job submission request
#[derive(Debug, Serialize)]
struct JobSubmitRequest {
    /// QASM circuit
    qasm: String,

    /// Number of shots
    shots: u32,

    /// Backend name
    backend: String,
}

/// Job status response
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct JobStatusResponse {
    id: String,
    status: JobStatus,
    #[serde(default)]
    backend: Option<String>,
}

/// Job result response
#[derive(Debug, Deserialize)]
struct JobResultResponse {
    #[serde(default)]
    results: Vec<CircuitResult>,
}

#[derive(Debug, Deserialize)]
struct CircuitResult {
    #[serde(default)]
    data: ResultData,
    #[serde(default)]
    shots: Option<u64>,
}

#[derive(Debug, Deserialize, Default)]
struct ResultData {
    #[serde(default)]
    counts: HashMap<String, u64>,
}

/// Job manager for submitting and monitoring jobs
pub struct JobManager;

impl JobManager {
    /// Submit a job to IBM Quantum
    pub async fn submit(backend: &IBMBackend, qasm: &str, shots: u32) -> Result<Job> {
        let backend_name = backend
            .selected_backend()
            .ok_or_else(|| IBMError::Other("No backend selected".into()))?;

        let request = JobSubmitRequest {
            qasm: qasm.to_string(),
            shots,
            backend: backend_name.to_string(),
        };

        let url = format!("{}/jobs", IBM_QUANTUM_API_URL);

        let response = backend
            .client()
            .post(&url)
            .header("Authorization", backend.credentials().auth_header())
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let text = response.text().await.unwrap_or_default();
            return Err(IBMError::JobSubmissionFailed(format!(
                "HTTP {}: {}",
                status, text
            )));
        }

        let status_resp: JobStatusResponse = response.json().await?;

        Ok(Job::new(status_resp.id, backend_name.to_string(), shots))
    }

    /// Get job status
    pub async fn get_status(backend: &IBMBackend, job_id: &str) -> Result<JobStatus> {
        let url = format!("{}/jobs/{}", IBM_QUANTUM_API_URL, job_id);

        let response = backend
            .client()
            .get(&url)
            .header("Authorization", backend.credentials().auth_header())
            .send()
            .await?;

        if response.status().as_u16() == 404 {
            return Err(IBMError::JobNotFound(job_id.to_string()));
        }

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let text = response.text().await.unwrap_or_default();
            return Err(IBMError::ApiErrorStructured {
                code: status,
                message: text,
            });
        }

        let status_resp: JobStatusResponse = response.json().await?;
        Ok(status_resp.status)
    }

    /// Wait for job completion
    pub async fn wait_for_completion(
        backend: &IBMBackend,
        job: &mut Job,
        timeout_secs: Option<u64>,
    ) -> Result<()> {
        let timeout = Duration::from_secs(timeout_secs.unwrap_or(MAX_WAIT_TIME));
        let start = Instant::now();

        loop {
            if start.elapsed() > timeout {
                return Err(IBMError::JobTimeout(job.id.clone(), timeout.as_secs()));
            }

            let status = Self::get_status(backend, &job.id).await?;
            job.status = status;

            match status {
                JobStatus::Completed => return Ok(()),
                JobStatus::Failed => {
                    return Err(IBMError::JobFailedWithId(
                        job.id.clone(),
                        "Job execution failed".into(),
                    ))
                }
                JobStatus::Cancelled => return Err(IBMError::JobCancelled(job.id.clone())),
                _ => {
                    sleep(Duration::from_secs(POLL_INTERVAL)).await;
                }
            }
        }
    }

    /// Get job result
    pub async fn get_result(backend: &IBMBackend, job: &Job) -> Result<JobResult> {
        if job.status != JobStatus::Completed {
            return Err(IBMError::Other(format!(
                "Job is not completed. Status: {:?}",
                job.status
            )));
        }

        let url = format!("{}/jobs/{}/results", IBM_QUANTUM_API_URL, job.id);

        let response = backend
            .client()
            .get(&url)
            .header("Authorization", backend.credentials().auth_header())
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

        let result_resp: JobResultResponse = response.json().await?;

        let circuit_result = result_resp
            .results
            .into_iter()
            .next()
            .ok_or_else(|| IBMError::InvalidResponse("No results in response".into()))?;

        Ok(JobResult {
            counts: circuit_result.data.counts,
            shots: circuit_result.shots.unwrap_or(job.shots as u64),
            execution_time_ms: 0,
            backend: job.backend.clone(),
            job_id: job.id.clone(),
        })
    }

    /// Submit, wait, and get result (convenience method)
    pub async fn run(
        backend: &IBMBackend,
        qasm: &str,
        shots: u32,
        timeout_secs: Option<u64>,
    ) -> Result<JobResult> {
        let mut job = Self::submit(backend, qasm, shots).await?;
        Self::wait_for_completion(backend, &mut job, timeout_secs).await?;
        Self::get_result(backend, &job).await
    }

    /// Cancel a job
    pub async fn cancel(backend: &IBMBackend, job_id: &str) -> Result<()> {
        let url = format!("{}/jobs/{}/cancel", IBM_QUANTUM_API_URL, job_id);

        let response = backend
            .client()
            .post(&url)
            .header("Authorization", backend.credentials().auth_header())
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

        Ok(())
    }
}

// ============================================================================
// Mock implementation for testing
// ============================================================================

#[cfg(feature = "mock")]
pub mod mock {
    use super::*;
    use rand::Rng;

    /// Mock job manager for testing without IBM credentials
    pub struct MockJobManager;

    impl MockJobManager {
        /// Submit a mock job
        pub fn submit(_qasm: &str, shots: u32) -> Job {
            let id = uuid::Uuid::new_v4().to_string();
            Job::new(id, "mock_backend".to_string(), shots)
        }

        /// Generate mock result
        pub fn generate_result(n_qubits: usize, shots: u64) -> JobResult {
            let mut rng = rand::thread_rng();
            let mut counts = HashMap::new();
            let mut remaining = shots;

            // Generate random bitstrings
            let n_outcomes = (1 << n_qubits).min(10); // Limit for large qubit counts

            for i in 0..n_outcomes {
                if remaining == 0 {
                    break;
                }

                let bitstring = format!("{:0width$b}", i, width = n_qubits);
                let count = if i == n_outcomes - 1 {
                    remaining
                } else {
                    rng.gen_range(0..=remaining / 2)
                };

                if count > 0 {
                    counts.insert(bitstring, count);
                    remaining -= count;
                }
            }

            JobResult {
                counts,
                shots,
                execution_time_ms: rng.gen_range(100..1000),
                backend: "mock_backend".to_string(),
                job_id: uuid::Uuid::new_v4().to_string(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_job_status_terminal() {
        assert!(JobStatus::Completed.is_terminal());
        assert!(JobStatus::Failed.is_terminal());
        assert!(JobStatus::Cancelled.is_terminal());
        assert!(!JobStatus::Queued.is_terminal());
        assert!(!JobStatus::Running.is_terminal());
    }

    #[test]
    fn test_job_result_probability() {
        let mut result = JobResult::new();
        result.counts.insert("00".to_string(), 700);
        result.counts.insert("11".to_string(), 300);
        result.shots = 1000;

        assert!((result.probability("00") - 0.7).abs() < 0.01);
        assert!((result.probability("11") - 0.3).abs() < 0.01);
        assert!((result.probability("01") - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_job_result_most_likely() {
        let mut result = JobResult::new();
        result.counts.insert("00".to_string(), 700);
        result.counts.insert("11".to_string(), 300);
        result.shots = 1000;

        let (bitstring, count) = result.most_likely().unwrap();
        assert_eq!(bitstring, "00");
        assert_eq!(count, 700);
    }

    #[test]
    fn test_expectation_z() {
        let mut result = JobResult::new();
        // All in |0⟩ state → Z expectation = +1
        result.counts.insert("0".to_string(), 1000);
        result.shots = 1000;

        assert!((result.expectation_z(0) - 1.0).abs() < 0.01);

        // All in |1⟩ state → Z expectation = -1
        result.counts.clear();
        result.counts.insert("1".to_string(), 1000);

        assert!((result.expectation_z(0) - (-1.0)).abs() < 0.01);

        // 50/50 → Z expectation = 0
        result.counts.clear();
        result.counts.insert("0".to_string(), 500);
        result.counts.insert("1".to_string(), 500);

        assert!((result.expectation_z(0) - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_expectation_zz() {
        let mut result = JobResult::new();
        // |00⟩ → ZZ = (+1)(+1) = +1
        result.counts.insert("00".to_string(), 500);
        // |11⟩ → ZZ = (-1)(-1) = +1
        result.counts.insert("11".to_string(), 500);
        result.shots = 1000;

        // Both give +1, so average should be +1
        assert!((result.expectation_zz(0, 1) - 1.0).abs() < 0.01);

        // |01⟩ → ZZ = (+1)(-1) = -1
        // |10⟩ → ZZ = (-1)(+1) = -1
        result.counts.clear();
        result.counts.insert("01".to_string(), 500);
        result.counts.insert("10".to_string(), 500);

        assert!((result.expectation_zz(0, 1) - (-1.0)).abs() < 0.01);
    }

    #[test]
    fn test_job_new() {
        let job = Job::new("test_id".to_string(), "ibm_manila".to_string(), 8192);
        assert_eq!(job.id, "test_id");
        assert_eq!(job.backend, "ibm_manila");
        assert_eq!(job.shots, 8192);
        assert_eq!(job.status, JobStatus::Queued);
    }
}

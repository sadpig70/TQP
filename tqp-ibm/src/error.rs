//! Error types for TQP-IBM integration
//!
//! Provides comprehensive error handling for:
//! - Network errors (connection, timeout)
//! - API errors (authentication, rate limiting)
//! - Hardware errors (calibration, queue)
//! - Transpilation errors (unsupported gates)

use thiserror::Error;

/// Result type alias for TQP-IBM operations
pub type Result<T> = std::result::Result<T, IBMError>;

/// Comprehensive error type for IBM Quantum operations
#[derive(Error, Debug)]
pub enum IBMError {
    // ==========================================================================
    // Credential Errors
    // ==========================================================================
    /// API token not found
    #[error("API token not found. Set IBM_QUANTUM_TOKEN environment variable.")]
    TokenNotFound,

    /// Invalid API token
    #[error("Invalid API token: {0}")]
    InvalidToken(String),

    /// Token expired
    #[error("API token has expired. Please refresh your token.")]
    TokenExpired,

    // ==========================================================================
    // Network Errors
    // ==========================================================================
    /// HTTP request failed
    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),

    /// Network error (general)
    #[error("Network error: {0}")]
    NetworkError(String),

    /// Connection timeout
    #[error("Connection timeout after {0} seconds")]
    ConnectionTimeout(u64),

    /// General timeout
    #[error("Timeout: {0}")]
    Timeout(String),

    /// Rate limit exceeded
    #[error("Rate limit exceeded. Retry after {0} seconds.")]
    RateLimitExceeded(u64),

    // ==========================================================================
    // API Errors
    // ==========================================================================
    /// API returned error response (structured)
    #[error("API error ({code}): {message}")]
    ApiErrorStructured { code: u16, message: String },

    /// API returned error response (simple string)
    #[error("API error: {0}")]
    ApiError(String),

    /// Parse error
    #[error("Parse error: {0}")]
    ParseError(String),

    /// Invalid response format
    #[error("Invalid API response: {0}")]
    InvalidResponse(String),

    /// Backend not found
    #[error("Backend '{0}' not found")]
    BackendNotFound(String),

    /// Backend unavailable
    #[error("Backend '{0}' is currently unavailable: {1}")]
    BackendUnavailable(String, String),

    // ==========================================================================
    // Job Errors
    // ==========================================================================
    /// Job submission failed
    #[error("Job submission failed: {0}")]
    JobSubmissionFailed(String),

    /// Job not found
    #[error("Job '{0}' not found")]
    JobNotFound(String),

    /// Job failed (simple)
    #[error("Job failed: {0}")]
    JobFailed(String),

    /// Job failed (with ID)
    #[error("Job '{0}' failed: {1}")]
    JobFailedWithId(String, String),

    /// Job cancelled
    #[error("Job '{0}' was cancelled")]
    JobCancelled(String),

    /// Job timeout
    #[error("Job '{0}' timed out after {1} seconds")]
    JobTimeout(String, u64),

    // ==========================================================================
    // Transpilation Errors
    // ==========================================================================
    /// Unsupported gate
    #[error("Unsupported gate: {0}")]
    UnsupportedGate(String),

    /// Invalid circuit
    #[error("Invalid circuit: {0}")]
    InvalidCircuit(String),

    /// Qubit count mismatch
    #[error("Qubit count mismatch: circuit has {circuit} qubits, backend supports {backend}")]
    QubitCountMismatch { circuit: usize, backend: usize },

    /// Circuit too deep
    #[error("Circuit depth {depth} exceeds backend limit {limit}")]
    CircuitTooDeep { depth: usize, limit: usize },

    // ==========================================================================
    // Hardware Errors
    // ==========================================================================
    /// Calibration data unavailable
    #[error("Calibration data unavailable for backend '{0}'")]
    CalibrationUnavailable(String),

    /// Hardware error during execution
    #[error("Hardware error: {0}")]
    HardwareError(String),

    // ==========================================================================
    // Other Errors
    // ==========================================================================
    /// JSON serialization/deserialization error
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Generic error
    #[error("{0}")]
    Other(String),
}

impl IBMError {
    /// Check if error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            IBMError::HttpError(_)
                | IBMError::ConnectionTimeout(_)
                | IBMError::RateLimitExceeded(_)
                | IBMError::BackendUnavailable(_, _)
        )
    }

    /// Get suggested retry delay in seconds
    pub fn retry_delay(&self) -> Option<u64> {
        match self {
            IBMError::RateLimitExceeded(delay) => Some(*delay),
            IBMError::ConnectionTimeout(_) => Some(5),
            IBMError::HttpError(_) => Some(10),
            IBMError::BackendUnavailable(_, _) => Some(60),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_retryable() {
        assert!(IBMError::ConnectionTimeout(30).is_retryable());
        assert!(IBMError::RateLimitExceeded(60).is_retryable());
        assert!(!IBMError::TokenNotFound.is_retryable());
        assert!(!IBMError::UnsupportedGate("XYZ".into()).is_retryable());
    }

    #[test]
    fn test_retry_delay() {
        assert_eq!(IBMError::RateLimitExceeded(120).retry_delay(), Some(120));
        assert_eq!(IBMError::ConnectionTimeout(30).retry_delay(), Some(5));
        assert_eq!(IBMError::TokenNotFound.retry_delay(), None);
    }

    #[test]
    fn test_error_display() {
        let err = IBMError::ApiErrorStructured {
            code: 401,
            message: "Unauthorized".into(),
        };
        assert_eq!(err.to_string(), "API error (401): Unauthorized");
    }
}

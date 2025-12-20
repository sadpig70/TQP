//! IBM Quantum credentials management
//!
//! Handles API token storage, retrieval, and validation.
//!
//! ## Token Sources (in priority order)
//! 1. Direct parameter
//! 2. Environment variable `IBM_QUANTUM_TOKEN`
//! 3. `.env` file
//! 4. Config file `~/.qiskit/qiskit-ibm.json`

use crate::error::{IBMError, Result};
use serde::{Deserialize, Serialize};
use std::env;
use std::path::PathBuf;

/// IBM Quantum credentials
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Credentials {
    /// API token
    token: String,

    /// Instance (hub/group/project)
    #[serde(default)]
    instance: Option<String>,

    /// Channel (ibm_quantum or ibm_cloud)
    #[serde(default = "default_channel")]
    channel: String,
}

fn default_channel() -> String {
    "ibm_quantum".to_string()
}

impl Credentials {
    /// Create credentials with token
    pub fn new(token: impl Into<String>) -> Self {
        Self {
            token: token.into(),
            instance: None,
            channel: default_channel(),
        }
    }

    /// Create credentials with token and instance
    pub fn with_instance(token: impl Into<String>, instance: impl Into<String>) -> Self {
        Self {
            token: token.into(),
            instance: Some(instance.into()),
            channel: default_channel(),
        }
    }

    /// Get API token
    pub fn token(&self) -> &str {
        &self.token
    }

    /// Get instance
    pub fn instance(&self) -> Option<&str> {
        self.instance.as_deref()
    }

    /// Get channel
    pub fn channel(&self) -> &str {
        &self.channel
    }

    /// Set channel
    pub fn set_channel(&mut self, channel: impl Into<String>) {
        self.channel = channel.into();
    }

    /// Validate token format (basic check)
    pub fn validate(&self) -> Result<()> {
        if self.token.is_empty() {
            return Err(IBMError::InvalidToken("Token is empty".into()));
        }

        // IBM tokens are typically long alphanumeric strings
        if self.token.len() < 32 {
            return Err(IBMError::InvalidToken("Token too short".into()));
        }

        Ok(())
    }

    /// Create authorization header value
    pub fn auth_header(&self) -> String {
        format!("Bearer {}", self.token)
    }
}

/// Credentials manager for loading and storing credentials
pub struct CredentialsManager;

impl CredentialsManager {
    /// Load credentials from available sources
    ///
    /// Priority:
    /// 1. Environment variable `IBM_QUANTUM_TOKEN`
    /// 2. `.env` file
    /// 3. Qiskit config file
    pub fn load() -> Result<Credentials> {
        // Try loading .env file (ignore errors if not found)
        let _ = dotenvy::dotenv();

        // Try environment variable
        if let Ok(token) = env::var("IBM_QUANTUM_TOKEN") {
            let instance = env::var("IBM_QUANTUM_INSTANCE").ok();
            let creds = if let Some(inst) = instance {
                Credentials::with_instance(token, inst)
            } else {
                Credentials::new(token)
            };
            creds.validate()?;
            return Ok(creds);
        }

        // Try Qiskit config file
        if let Some(creds) = Self::load_from_qiskit_config()? {
            return Ok(creds);
        }

        Err(IBMError::TokenNotFound)
    }

    /// Load from specific token
    pub fn from_token(token: impl Into<String>) -> Result<Credentials> {
        let creds = Credentials::new(token);
        creds.validate()?;
        Ok(creds)
    }

    /// Load from Qiskit config file
    fn load_from_qiskit_config() -> Result<Option<Credentials>> {
        let config_path = Self::qiskit_config_path()?;

        if !config_path.exists() {
            return Ok(None);
        }

        let content = std::fs::read_to_string(&config_path)?;
        let config: serde_json::Value = serde_json::from_str(&content)?;

        // Try to extract token from various possible locations
        if let Some(token) = config.get("token").and_then(|v| v.as_str()) {
            let creds = Credentials::new(token);
            creds.validate()?;
            return Ok(Some(creds));
        }

        // Try default_provider format
        if let Some(provider) = config.get("default_provider") {
            if let Some(token) = provider.get("token").and_then(|v| v.as_str()) {
                let instance = provider.get("instance").and_then(|v| v.as_str());
                let creds = if let Some(inst) = instance {
                    Credentials::with_instance(token, inst)
                } else {
                    Credentials::new(token)
                };
                creds.validate()?;
                return Ok(Some(creds));
            }
        }

        Ok(None)
    }

    /// Get Qiskit config file path
    fn qiskit_config_path() -> Result<PathBuf> {
        let home = env::var("HOME")
            .or_else(|_| env::var("USERPROFILE"))
            .map_err(|_| IBMError::Other("Could not determine home directory".into()))?;

        Ok(PathBuf::from(home).join(".qiskit").join("qiskit-ibm.json"))
    }

    /// Save credentials to environment (for current session)
    pub fn save_to_env(creds: &Credentials) {
        env::set_var("IBM_QUANTUM_TOKEN", creds.token());
        if let Some(instance) = creds.instance() {
            env::set_var("IBM_QUANTUM_INSTANCE", instance);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_credentials_new() {
        let creds = Credentials::new("test_token_1234567890123456789012345678901234567890");
        assert_eq!(
            creds.token(),
            "test_token_1234567890123456789012345678901234567890"
        );
        assert!(creds.instance().is_none());
        assert_eq!(creds.channel(), "ibm_quantum");
    }

    #[test]
    fn test_credentials_with_instance() {
        let creds = Credentials::with_instance(
            "test_token_1234567890123456789012345678901234567890",
            "ibm-q/open/main",
        );
        assert_eq!(creds.instance(), Some("ibm-q/open/main"));
    }

    #[test]
    fn test_validate_empty_token() {
        let creds = Credentials::new("");
        assert!(creds.validate().is_err());
    }

    #[test]
    fn test_validate_short_token() {
        let creds = Credentials::new("short");
        assert!(creds.validate().is_err());
    }

    #[test]
    fn test_validate_valid_token() {
        let creds = Credentials::new("a".repeat(64));
        assert!(creds.validate().is_ok());
    }

    #[test]
    fn test_auth_header() {
        let creds = Credentials::new("my_token_12345678901234567890123456");
        assert!(creds.auth_header().starts_with("Bearer "));
    }
}

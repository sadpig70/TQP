//! IBM Quantum Hardware Validation
//!
//! Tests real hardware connectivity and circuit execution.

use tqp_ibm::backend::IBMBackend;
use tqp_ibm::credentials::Credentials;
use tqp_ibm::jobs::{DynamicalDecouplingOptions, JobManager, RuntimeOptions};
use tqp_ibm::transpiler::{Circuit, Gate, GateType, ISATranspiler};

const API_KEY: &str = "eaZg3euoMUGpZVcWxXVsU55MGddnbwuR74uDPU8-F48W";
const SERVICE_CRN: &str = "crn:v1:bluemix:public:quantum-computing:us-east:a/81a3ca8cfbdd4b9b97f558485923bb5e:d1243ad8-1843-490c-ae6c-0df4297d55fc::";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== IBM Quantum Hardware Validation ===\n");

    // Step 1: Create credentials with CRN
    println!("1. Creating credentials...");
    let mut creds = Credentials::with_crn(API_KEY, SERVICE_CRN);
    creds.set_channel("ibm_cloud");
    println!("   ✓ Token: {}...", &API_KEY[..20]);
    println!("   ✓ CRN set");

    // Step 2: Create backend
    println!("\n2. Creating backend connection...");
    let mut backend = IBMBackend::new(creds)?;
    println!("   ✓ Backend created");

    // Step 3: List available backends
    println!("\n3. Querying available backends...");
    match backend.list_backends().await {
        Ok(backends) => {
            println!("   ✓ Found {} backends:", backends.len());
            for info in &backends {
                println!(
                    "     - {} ({} qubits, {:?})",
                    info.name, info.n_qubits, info.status
                );
            }

            // Step 4: Select first backend
            if let Some(first) = backends.first() {
                println!("\n4. Selecting backend: {}...", first.name);
                backend.select(&first.name).await?;
                println!("   ✓ Backend selected");

                // Step 5: Create test circuit (Bell state)
                println!("\n5. Creating Bell state circuit...");
                let mut circuit = Circuit::new(2);
                circuit.add(Gate::single(GateType::H, 0));
                circuit.add(Gate::two(GateType::CNOT, 0, 1));

                // Convert to ISA gates for IBM hardware
                let qasm = ISATranspiler::to_isa_qasm(&circuit, &[])?;
                println!("   ✓ Circuit created and converted to ISA gates");
                println!("   QASM preview (ISA):");
                for line in qasm.lines().take(12) {
                    println!("     {}", line);
                }

                // Step 6: Submit job
                println!("\n6. Submitting job...");

                // Enable dynamical decoupling for error suppression
                let options = RuntimeOptions {
                    transpilation: None,
                    dynamical_decoupling: Some(DynamicalDecouplingOptions {
                        enable: true,
                        sequence_type: Some("XY4".to_string()),
                    }),
                    twirling: None,
                };

                match JobManager::submit_sampler(&backend, &qasm, 4096, Some(options)).await {
                    Ok(job) => {
                        println!("   ✓ Job submitted: {}", job.id);
                        println!("   Waiting for completion (this may take a few minutes)...");

                        // Wait and get result
                        let mut job = job;
                        match JobManager::wait_for_completion(&backend, &mut job, Some(600)).await {
                            Ok(()) => {
                                println!("   ✓ Job completed!");

                                match JobManager::get_result(&backend, &job).await {
                                    Ok(result) => {
                                        println!("   Shots: {}", result.shots);
                                        println!("   Results:");

                                        let mut counts: Vec<_> = result.counts.iter().collect();
                                        counts.sort_by(|a, b| b.1.cmp(a.1));

                                        for (state, count) in counts.iter().take(4) {
                                            let prob =
                                                (**count as f64) / (result.shots as f64) * 100.0;
                                            println!("     |{}⟩: {} ({:.1}%)", state, count, prob);
                                        }

                                        // Validate Bell state
                                        let prob_00 = result.counts.get("00").copied().unwrap_or(0)
                                            as f64
                                            / result.shots as f64;
                                        let prob_11 = result.counts.get("11").copied().unwrap_or(0)
                                            as f64
                                            / result.shots as f64;
                                        let bell_fidelity = prob_00 + prob_11;

                                        println!(
                                            "\n   Bell state fidelity: {:.1}%",
                                            bell_fidelity * 100.0
                                        );

                                        if bell_fidelity > 0.8 {
                                            println!("   ✓ Hardware validation PASSED");
                                        } else {
                                            println!("   ⚠ Hardware shows significant noise");
                                        }
                                    }
                                    Err(e) => println!("   ⚠ Failed to get result: {}", e),
                                }
                            }
                            Err(e) => println!("   ⚠ Job failed: {}", e),
                        }
                    }
                    Err(e) => {
                        println!("   ⚠ Job submission failed: {}", e);
                    }
                }
            }
        }
        Err(e) => {
            println!("   ⚠ Failed to list backends: {}", e);
        }
    }

    println!("\n=== Validation Complete ===");
    Ok(())
}

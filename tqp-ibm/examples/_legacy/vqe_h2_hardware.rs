//! VQE H₂ Hardware Validation
//!
//! Executes Variational Quantum Eigensolver for H₂ molecule on IBM Quantum hardware.
//! Uses 2-qubit symmetry-reduced Hamiltonian for minimal circuit depth.

use std::f64::consts::PI;
use tqp_ibm::backend::IBMBackend;
use tqp_ibm::credentials::Credentials;
use tqp_ibm::error::Result;
use tqp_ibm::jobs::JobManager;
use tqp_ibm::transpiler::{Circuit, Gate, GateType, ISATranspiler, ProcessorType};

// IBM Quantum credentials
const API_KEY: &str = "eaZg3euoMUGpZVcWxXVsU55MGddnbwuR74uDPU8-F48W";
const SERVICE_CRN: &str = "crn:v1:bluemix:public:quantum-computing:us-east:a/81a3ca8cfbdd4b9b97f558485923bb5e:d1243ad8-1843-490c-ae6c-0df4297d55fc::";

// H₂ Hamiltonian coefficients at equilibrium (0.7414 Å)
// H = g0*I + g1*Z0 + g2*Z1 + g3*Z0Z1 + g4*(X0X1 + Y0Y1)
const G0: f64 = -0.7384; // Identity coefficient
const G1: f64 = 0.1322; // Z0 coefficient
const G2: f64 = -0.1322; // Z1 coefficient
const G3: f64 = 0.1480; // Z0Z1 coefficient
const G4: f64 = 0.1435; // X0X1 + Y0Y1 coefficient

// Theoretical exact ground state energy
const EXACT_ENERGY: f64 = -1.1372; // Hartree (for 2-qubit model)

/// Build UCC ansatz circuit for H₂
///
/// |ψ(θ)⟩ = e^{iθ(Y0X1 - X0Y1)/2} |01⟩
///
/// Simplified to single parameter rotation
fn build_h2_ansatz(theta: f64) -> Circuit {
    let mut circuit = Circuit::new(2);

    // Initial state |01⟩ (one electron in each orbital)
    circuit.add(Gate::single(GateType::X, 0));

    // UCC single excitation: exp(iθ(Y0X1 - X0Y1)/2)
    // Decomposed into native gates
    circuit.add(Gate::single(GateType::H, 0));
    circuit.add(Gate::single(GateType::H, 1));
    circuit.add(Gate::two(GateType::CNOT, 0, 1));
    circuit.add(Gate::single(GateType::Rz(theta), 1));
    circuit.add(Gate::two(GateType::CNOT, 0, 1));
    circuit.add(Gate::single(GateType::H, 0));
    circuit.add(Gate::single(GateType::H, 1));

    circuit
}

/// Build circuit to measure Z0
fn build_z0_circuit(theta: f64) -> Circuit {
    build_h2_ansatz(theta)
    // Measurement in Z basis is default
}

/// Build circuit to measure Z1
fn build_z1_circuit(theta: f64) -> Circuit {
    build_h2_ansatz(theta)
    // Measurement in Z basis is default
}

/// Build circuit to measure Z0Z1
fn build_z0z1_circuit(theta: f64) -> Circuit {
    build_h2_ansatz(theta)
    // Measure both qubits, parity gives ZZ
}

/// Build circuit to measure X0X1
fn build_x0x1_circuit(theta: f64) -> Circuit {
    let mut circuit = build_h2_ansatz(theta);
    // Change to X basis
    circuit.add(Gate::single(GateType::H, 0));
    circuit.add(Gate::single(GateType::H, 1));
    circuit
}

/// Build circuit to measure Y0Y1
fn build_y0y1_circuit(theta: f64) -> Circuit {
    let mut circuit = build_h2_ansatz(theta);
    // Change to Y basis: S†H
    circuit.add(Gate::single(GateType::Sdg, 0));
    circuit.add(Gate::single(GateType::H, 0));
    circuit.add(Gate::single(GateType::Sdg, 1));
    circuit.add(Gate::single(GateType::H, 1));
    circuit
}

/// Calculate expectation value from measurement results
fn expectation_from_counts(counts: &std::collections::HashMap<String, u64>, shots: u64) -> f64 {
    let mut exp = 0.0;
    for (bitstring, &count) in counts {
        // Count number of 1s (parity)
        let parity = bitstring.chars().filter(|&c| c == '1').count();
        let sign = if parity % 2 == 0 { 1.0 } else { -1.0 };
        exp += sign * (count as f64) / (shots as f64);
    }
    exp
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║        VQE H₂ Hardware Validation - IBM Quantum           ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    // Setup credentials and backend
    println!("1. Setting up IBM Quantum connection...");
    let mut creds = Credentials::with_crn(API_KEY, SERVICE_CRN);
    creds.set_channel("ibm_cloud");

    let mut backend = IBMBackend::new(creds)?;

    // List and select backend
    let backends = backend.list_backends().await?;
    println!("   Found {} backends", backends.len());

    // Select ibm_fez (Heron processor with CZ native)
    let target_backend = "ibm_fez";
    backend.select(target_backend).await?;
    println!("   ✓ Selected: {}", target_backend);

    // Optimal theta from classical optimization (pre-computed)
    // For H₂ at equilibrium, optimal θ ≈ 0.11
    let theta = 0.11;
    let shots = 4096;

    println!("\n2. VQE Parameters:");
    println!("   θ = {:.4} rad", theta);
    println!("   Shots = {}", shots);
    println!("   Target: E_exact = {:.4} Ha", EXACT_ENERGY);

    // Transpiler for Heron processor
    let transpiler = ISATranspiler::new(ProcessorType::Heron);

    println!("\n3. Submitting measurement circuits...");

    // Submit all measurement circuits
    let circuits = vec![
        ("Z0", build_z0_circuit(theta)),
        ("Z1", build_z1_circuit(theta)),
        ("Z0Z1", build_z0z1_circuit(theta)),
        ("X0X1", build_x0x1_circuit(theta)),
        ("Y0Y1", build_y0y1_circuit(theta)),
    ];

    let mut results = Vec::new();

    for (name, circuit) in circuits {
        // Transpile to ISA gates
        let qasm = transpiler.to_isa_qasm(&circuit, &[])?;

        println!("   Submitting {} measurement...", name);

        // Submit job
        let mut job = JobManager::submit_sampler(&backend, &qasm, shots as u32, None).await?;
        println!("   Job ID: {}", job.id);

        // Wait for completion
        JobManager::wait_for_completion(&backend, &mut job, Some(300)).await?;

        // Get result
        let result = JobManager::get_result(&backend, &job).await?;
        let exp = expectation_from_counts(&result.counts, result.shots);

        println!("   ✓ {} = {:.4}", name, exp);
        results.push((name, exp));
    }

    // Calculate total energy
    println!("\n4. Computing VQE Energy:");

    let exp_z0 = results
        .iter()
        .find(|(n, _)| *n == "Z0")
        .map(|(_, e)| *e)
        .unwrap_or(0.0);
    let exp_z1 = results
        .iter()
        .find(|(n, _)| *n == "Z1")
        .map(|(_, e)| *e)
        .unwrap_or(0.0);
    let exp_z0z1 = results
        .iter()
        .find(|(n, _)| *n == "Z0Z1")
        .map(|(_, e)| *e)
        .unwrap_or(0.0);
    let exp_x0x1 = results
        .iter()
        .find(|(n, _)| *n == "X0X1")
        .map(|(_, e)| *e)
        .unwrap_or(0.0);
    let exp_y0y1 = results
        .iter()
        .find(|(n, _)| *n == "Y0Y1")
        .map(|(_, e)| *e)
        .unwrap_or(0.0);

    // H = g0*I + g1*Z0 + g2*Z1 + g3*Z0Z1 + g4*(X0X1 + Y0Y1)
    let energy = G0 + G1 * exp_z0 + G2 * exp_z1 + G3 * exp_z0z1 + G4 * (exp_x0x1 + exp_y0y1);

    println!("   ⟨Z0⟩    = {:.4}", exp_z0);
    println!("   ⟨Z1⟩    = {:.4}", exp_z1);
    println!("   ⟨Z0Z1⟩  = {:.4}", exp_z0z1);
    println!("   ⟨X0X1⟩  = {:.4}", exp_x0x1);
    println!("   ⟨Y0Y1⟩  = {:.4}", exp_y0y1);

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║                      RESULTS                               ║");
    println!("╠════════════════════════════════════════════════════════════╣");
    println!(
        "║  VQE Energy:    {:.6} Ha                              ║",
        energy
    );
    println!(
        "║  Exact Energy:  {:.6} Ha                              ║",
        EXACT_ENERGY
    );
    println!(
        "║  Error:         {:.6} Ha                              ║",
        (energy - EXACT_ENERGY).abs()
    );
    println!(
        "║  Accuracy:      {:.2}%                                    ║",
        (1.0 - (energy - EXACT_ENERGY).abs() / EXACT_ENERGY.abs()) * 100.0
    );
    println!("╚════════════════════════════════════════════════════════════╝");

    // Validation
    let error = (energy - EXACT_ENERGY).abs();
    if error < 0.1 {
        println!("\n✅ VQE Hardware Validation PASSED (error < 0.1 Ha)");
    } else if error < 0.2 {
        println!("\n⚠️ VQE Hardware Validation MARGINAL (0.1 < error < 0.2 Ha)");
    } else {
        println!("\n❌ VQE Hardware Validation FAILED (error > 0.2 Ha)");
    }

    Ok(())
}

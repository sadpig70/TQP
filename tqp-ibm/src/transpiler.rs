//! QASM Transpiler: TQP Circuit → OpenQASM 3.0
//!
//! Converts TQP variational circuits to OpenQASM 3.0 format
//! for execution on IBM Quantum hardware.
//!
//! ## Supported Gates
//! - Single-qubit: H, X, Y, Z, Rx, Ry, Rz, S, T, SX
//! - Two-qubit: CNOT, CZ, SWAP
//! - Parameterized: All rotation gates with symbolic parameters

use crate::error::{IBMError, Result};

/// Gate types supported by the transpiler
#[derive(Debug, Clone, PartialEq)]
pub enum GateType {
    // Identity
    I,

    // Single-qubit Pauli gates
    X,
    Y,
    Z,

    // Hadamard
    H,

    // Phase gates
    S,
    Sdg,
    T,
    Tdg,

    // Rotation gates (angle in radians)
    Rx(f64),
    Ry(f64),
    Rz(f64),

    // IBM native gates
    SX,

    // Two-qubit gates
    CNOT,
    CZ,
    SWAP,

    // Parameterized rotation (parameter index)
    RxParam(usize),
    RyParam(usize),
    RzParam(usize),

    // Barrier
    Barrier,
}

/// A gate in the circuit
#[derive(Debug, Clone)]
pub struct Gate {
    /// Gate type
    pub gate_type: GateType,

    /// Target qubit(s)
    pub qubits: Vec<usize>,

    /// Optional label
    pub label: Option<String>,
}

impl Gate {
    /// Create a single-qubit gate
    pub fn single(gate_type: GateType, qubit: usize) -> Self {
        Self {
            gate_type,
            qubits: vec![qubit],
            label: None,
        }
    }

    /// Create a two-qubit gate
    pub fn two(gate_type: GateType, control: usize, target: usize) -> Self {
        Self {
            gate_type,
            qubits: vec![control, target],
            label: None,
        }
    }

    /// Create a barrier on specified qubits
    pub fn barrier(qubits: Vec<usize>) -> Self {
        Self {
            gate_type: GateType::Barrier,
            qubits,
            label: None,
        }
    }
}

/// Circuit representation for transpilation
#[derive(Debug, Clone)]
pub struct Circuit {
    /// Number of qubits
    pub n_qubits: usize,

    /// Gates in the circuit
    pub gates: Vec<Gate>,

    /// Number of parameters
    pub n_params: usize,

    /// Parameter names
    pub param_names: Vec<String>,
}

impl Circuit {
    /// Create a new circuit
    pub fn new(n_qubits: usize) -> Self {
        Self {
            n_qubits,
            gates: Vec::new(),
            n_params: 0,
            param_names: Vec::new(),
        }
    }

    /// Add a gate
    pub fn add(&mut self, gate: Gate) {
        self.gates.push(gate);
    }

    /// Add a parameterized rotation and return parameter index
    pub fn add_param_rx(&mut self, qubit: usize, name: Option<String>) -> usize {
        let idx = self.n_params;
        self.n_params += 1;
        self.param_names
            .push(name.unwrap_or_else(|| format!("theta_{}", idx)));
        self.gates.push(Gate::single(GateType::RxParam(idx), qubit));
        idx
    }

    /// Add a parameterized Ry rotation
    pub fn add_param_ry(&mut self, qubit: usize, name: Option<String>) -> usize {
        let idx = self.n_params;
        self.n_params += 1;
        self.param_names
            .push(name.unwrap_or_else(|| format!("theta_{}", idx)));
        self.gates.push(Gate::single(GateType::RyParam(idx), qubit));
        idx
    }

    /// Add a parameterized Rz rotation
    pub fn add_param_rz(&mut self, qubit: usize, name: Option<String>) -> usize {
        let idx = self.n_params;
        self.n_params += 1;
        self.param_names
            .push(name.unwrap_or_else(|| format!("theta_{}", idx)));
        self.gates.push(Gate::single(GateType::RzParam(idx), qubit));
        idx
    }

    /// Get circuit depth (number of layers)
    pub fn depth(&self) -> usize {
        if self.gates.is_empty() {
            return 0;
        }

        // Track when each qubit is next available
        let mut qubit_depth = vec![0usize; self.n_qubits];

        for gate in &self.gates {
            if gate.gate_type == GateType::Barrier {
                continue;
            }

            let max_depth = gate
                .qubits
                .iter()
                .map(|&q| qubit_depth[q])
                .max()
                .unwrap_or(0);
            let new_depth = max_depth + 1;

            for &q in &gate.qubits {
                qubit_depth[q] = new_depth;
            }
        }

        qubit_depth.into_iter().max().unwrap_or(0)
    }
}

/// Gate mapper for QASM output
pub struct GateMapper;

impl GateMapper {
    /// Map gate type to QASM instruction
    pub fn to_qasm(gate: &Gate, params: Option<&[f64]>) -> Result<String> {
        let qubits = &gate.qubits;

        match &gate.gate_type {
            // Identity
            GateType::I => Ok(format!("id q[{}];", qubits[0])),

            // Pauli gates
            GateType::X => Ok(format!("x q[{}];", qubits[0])),
            GateType::Y => Ok(format!("y q[{}];", qubits[0])),
            GateType::Z => Ok(format!("z q[{}];", qubits[0])),

            // Hadamard
            GateType::H => Ok(format!("h q[{}];", qubits[0])),

            // Phase gates
            GateType::S => Ok(format!("s q[{}];", qubits[0])),
            GateType::Sdg => Ok(format!("sdg q[{}];", qubits[0])),
            GateType::T => Ok(format!("t q[{}];", qubits[0])),
            GateType::Tdg => Ok(format!("tdg q[{}];", qubits[0])),

            // Rotation gates with fixed angle
            GateType::Rx(angle) => Ok(format!("rx({}) q[{}];", angle, qubits[0])),
            GateType::Ry(angle) => Ok(format!("ry({}) q[{}];", angle, qubits[0])),
            GateType::Rz(angle) => Ok(format!("rz({}) q[{}];", angle, qubits[0])),

            // IBM native sqrt(X)
            GateType::SX => Ok(format!("sx q[{}];", qubits[0])),

            // Two-qubit gates
            GateType::CNOT => Ok(format!("cx q[{}], q[{}];", qubits[0], qubits[1])),
            GateType::CZ => Ok(format!("cz q[{}], q[{}];", qubits[0], qubits[1])),
            GateType::SWAP => Ok(format!("swap q[{}], q[{}];", qubits[0], qubits[1])),

            // Parameterized gates
            GateType::RxParam(idx) => {
                if let Some(p) = params {
                    Ok(format!("rx({}) q[{}];", p[*idx], qubits[0]))
                } else {
                    Ok(format!("rx(theta[{}]) q[{}];", idx, qubits[0]))
                }
            }
            GateType::RyParam(idx) => {
                if let Some(p) = params {
                    Ok(format!("ry({}) q[{}];", p[*idx], qubits[0]))
                } else {
                    Ok(format!("ry(theta[{}]) q[{}];", idx, qubits[0]))
                }
            }
            GateType::RzParam(idx) => {
                if let Some(p) = params {
                    Ok(format!("rz({}) q[{}];", p[*idx], qubits[0]))
                } else {
                    Ok(format!("rz(theta[{}]) q[{}];", idx, qubits[0]))
                }
            }

            // Barrier
            GateType::Barrier => {
                if qubits.is_empty() {
                    Ok("barrier q;".to_string())
                } else {
                    let qlist: Vec<String> = qubits.iter().map(|q| format!("q[{}]", q)).collect();
                    Ok(format!("barrier {};", qlist.join(", ")))
                }
            }
        }
    }
}

/// QASM Builder for constructing QASM output
pub struct QASMBuilder {
    /// QASM version
    version: String,

    /// Include statements
    includes: Vec<String>,

    /// Number of qubits
    n_qubits: usize,

    /// Number of classical bits
    n_cbits: usize,

    /// Gate instructions
    instructions: Vec<String>,

    /// Add measurement at the end
    add_measurement: bool,
}

impl QASMBuilder {
    /// Create new QASM builder
    pub fn new(n_qubits: usize) -> Self {
        Self {
            version: "OPENQASM 3.0;".to_string(),
            includes: vec!["include \"stdgates.inc\";".to_string()],
            n_qubits,
            n_cbits: n_qubits,
            instructions: Vec::new(),
            add_measurement: true,
        }
    }

    /// Set whether to add measurement
    pub fn with_measurement(mut self, add: bool) -> Self {
        self.add_measurement = add;
        self
    }

    /// Add an instruction
    pub fn add_instruction(&mut self, instruction: String) {
        self.instructions.push(instruction);
    }

    /// Build QASM string
    pub fn build(self) -> String {
        let mut lines = Vec::new();

        // Version
        lines.push(self.version);
        lines.push(String::new());

        // Includes
        for inc in self.includes {
            lines.push(inc);
        }
        lines.push(String::new());

        // Declarations
        lines.push(format!("qubit[{}] q;", self.n_qubits));
        lines.push(format!("bit[{}] c;", self.n_cbits));
        lines.push(String::new());

        // Instructions
        for inst in self.instructions {
            lines.push(inst);
        }

        // Measurement
        if self.add_measurement {
            lines.push(String::new());
            lines.push("c = measure q;".to_string());
        }

        lines.join("\n")
    }
}

/// Main transpiler
pub struct QASMTranspiler;

impl QASMTranspiler {
    /// Transpile circuit to QASM with bound parameters
    pub fn transpile(circuit: &Circuit, params: &[f64]) -> Result<String> {
        Self::validate_circuit(circuit)?;
        Self::validate_params(circuit, params)?;

        let mut builder = QASMBuilder::new(circuit.n_qubits);

        for gate in &circuit.gates {
            let instruction = GateMapper::to_qasm(gate, Some(params))?;
            builder.add_instruction(instruction);
        }

        Ok(builder.build())
    }

    /// Transpile circuit to QASM with symbolic parameters
    pub fn transpile_symbolic(circuit: &Circuit) -> Result<String> {
        Self::validate_circuit(circuit)?;

        let mut builder = QASMBuilder::new(circuit.n_qubits);

        // Add parameter declarations if any
        if circuit.n_params > 0 {
            builder.add_instruction(format!("input float[64] theta[{}];", circuit.n_params));
            builder.add_instruction(String::new());
        }

        for gate in &circuit.gates {
            let instruction = GateMapper::to_qasm(gate, None)?;
            builder.add_instruction(instruction);
        }

        Ok(builder.build())
    }

    /// Transpile circuit without measurement
    pub fn transpile_no_measure(circuit: &Circuit, params: &[f64]) -> Result<String> {
        Self::validate_circuit(circuit)?;
        Self::validate_params(circuit, params)?;

        let mut builder = QASMBuilder::new(circuit.n_qubits).with_measurement(false);

        for gate in &circuit.gates {
            let instruction = GateMapper::to_qasm(gate, Some(params))?;
            builder.add_instruction(instruction);
        }

        Ok(builder.build())
    }

    /// Validate circuit
    fn validate_circuit(circuit: &Circuit) -> Result<()> {
        if circuit.n_qubits == 0 {
            return Err(IBMError::InvalidCircuit("Circuit has no qubits".into()));
        }

        // Validate qubit indices
        for gate in &circuit.gates {
            for &qubit in &gate.qubits {
                if qubit >= circuit.n_qubits {
                    return Err(IBMError::InvalidCircuit(format!(
                        "Qubit index {} out of range (circuit has {} qubits)",
                        qubit, circuit.n_qubits
                    )));
                }
            }
        }

        Ok(())
    }

    /// Validate parameters
    fn validate_params(circuit: &Circuit, params: &[f64]) -> Result<()> {
        if params.len() != circuit.n_params {
            return Err(IBMError::InvalidCircuit(format!(
                "Parameter count mismatch: circuit has {} parameters, got {}",
                circuit.n_params,
                params.len()
            )));
        }

        // Check for NaN/Inf
        for (i, &p) in params.iter().enumerate() {
            if !p.is_finite() {
                return Err(IBMError::InvalidCircuit(format!(
                    "Parameter {} is not finite: {}",
                    i, p
                )));
            }
        }

        Ok(())
    }
}

// ============================================================================
// Builder patterns for common circuits
// ============================================================================

/// Helper to build common VQE/QAOA circuits
pub struct CircuitBuilder;

impl CircuitBuilder {
    /// Build a simple hardware-efficient ansatz
    pub fn hardware_efficient(n_qubits: usize, depth: usize) -> Circuit {
        let mut circuit = Circuit::new(n_qubits);

        for layer in 0..depth {
            // Single-qubit rotation layer
            for q in 0..n_qubits {
                circuit.add_param_ry(q, Some(format!("ry_{}_{}", layer, q)));
                circuit.add_param_rz(q, Some(format!("rz_{}_{}", layer, q)));
            }

            // Entangling layer (linear connectivity)
            for q in 0..(n_qubits - 1) {
                circuit.add(Gate::two(GateType::CNOT, q, q + 1));
            }
        }

        // Final rotation layer
        for q in 0..n_qubits {
            circuit.add_param_ry(q, Some(format!("ry_{}_{}", depth, q)));
            circuit.add_param_rz(q, Some(format!("rz_{}_{}", depth, q)));
        }

        circuit
    }

    /// Build QAOA circuit for MaxCut
    pub fn qaoa_maxcut(n_qubits: usize, edges: &[(usize, usize)], p: usize) -> Circuit {
        let mut circuit = Circuit::new(n_qubits);

        // Initial superposition
        for q in 0..n_qubits {
            circuit.add(Gate::single(GateType::H, q));
        }

        for layer in 0..p {
            // Cost unitary (gamma)
            for &(i, j) in edges {
                circuit.add(Gate::two(GateType::CNOT, i, j));
                circuit.add_param_rz(j, Some(format!("gamma_{}_{}", layer, j)));
                circuit.add(Gate::two(GateType::CNOT, i, j));
            }

            // Mixer unitary (beta)
            for q in 0..n_qubits {
                circuit.add_param_rx(q, Some(format!("beta_{}_{}", layer, q)));
            }
        }

        circuit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_circuit() {
        let mut circuit = Circuit::new(2);
        circuit.add(Gate::single(GateType::H, 0));
        circuit.add(Gate::two(GateType::CNOT, 0, 1));

        let qasm = QASMTranspiler::transpile(&circuit, &[]).unwrap();

        assert!(qasm.contains("OPENQASM 3.0"));
        assert!(qasm.contains("qubit[2] q;"));
        assert!(qasm.contains("h q[0];"));
        assert!(qasm.contains("cx q[0], q[1];"));
        assert!(qasm.contains("c = measure q;"));
    }

    #[test]
    fn test_parameterized_circuit() {
        let mut circuit = Circuit::new(2);
        circuit.add(Gate::single(GateType::H, 0));
        circuit.add_param_ry(0, None);
        circuit.add_param_rz(1, None);
        circuit.add(Gate::two(GateType::CNOT, 0, 1));

        let params = vec![0.5, 1.0];
        let qasm = QASMTranspiler::transpile(&circuit, &params).unwrap();

        assert!(qasm.contains("ry(0.5) q[0];"));
        assert!(qasm.contains("rz(1) q[1];"));
    }

    #[test]
    fn test_symbolic_parameters() {
        let mut circuit = Circuit::new(2);
        circuit.add_param_ry(0, None);
        circuit.add_param_rz(1, None);

        let qasm = QASMTranspiler::transpile_symbolic(&circuit).unwrap();

        assert!(qasm.contains("input float[64] theta[2];"));
        assert!(qasm.contains("ry(theta[0]) q[0];"));
        assert!(qasm.contains("rz(theta[1]) q[1];"));
    }

    #[test]
    fn test_circuit_depth() {
        let mut circuit = Circuit::new(3);
        circuit.add(Gate::single(GateType::H, 0));
        circuit.add(Gate::single(GateType::H, 1));
        circuit.add(Gate::single(GateType::H, 2));
        circuit.add(Gate::two(GateType::CNOT, 0, 1));
        circuit.add(Gate::two(GateType::CNOT, 1, 2));

        assert_eq!(circuit.depth(), 3);
    }

    #[test]
    fn test_hardware_efficient_ansatz() {
        let circuit = CircuitBuilder::hardware_efficient(4, 2);

        assert_eq!(circuit.n_qubits, 4);
        assert_eq!(circuit.n_params, 4 * 2 * 3); // (depth+1) * n_qubits * 2
        assert!(circuit.depth() > 0);
    }

    #[test]
    fn test_qaoa_circuit() {
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let circuit = CircuitBuilder::qaoa_maxcut(3, &edges, 2);

        assert_eq!(circuit.n_qubits, 3);
        // Each layer: 3 gamma params + 3 beta params = 6
        // 2 layers = 12 params
        assert_eq!(circuit.n_params, 12);
    }

    #[test]
    fn test_invalid_qubit_index() {
        let mut circuit = Circuit::new(2);
        circuit.add(Gate::single(GateType::H, 5)); // Out of range

        let result = QASMTranspiler::transpile(&circuit, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_param_count_mismatch() {
        let mut circuit = Circuit::new(2);
        circuit.add_param_ry(0, None);
        circuit.add_param_rz(1, None);

        let result = QASMTranspiler::transpile(&circuit, &[0.5]); // Missing one param
        assert!(result.is_err());
    }

    #[test]
    fn test_all_gate_types() {
        let mut circuit = Circuit::new(3);

        // Single-qubit gates
        circuit.add(Gate::single(GateType::I, 0));
        circuit.add(Gate::single(GateType::X, 0));
        circuit.add(Gate::single(GateType::Y, 0));
        circuit.add(Gate::single(GateType::Z, 0));
        circuit.add(Gate::single(GateType::H, 0));
        circuit.add(Gate::single(GateType::S, 0));
        circuit.add(Gate::single(GateType::Sdg, 0));
        circuit.add(Gate::single(GateType::T, 0));
        circuit.add(Gate::single(GateType::Tdg, 0));
        circuit.add(Gate::single(GateType::SX, 0));
        circuit.add(Gate::single(GateType::Rx(0.5), 0));
        circuit.add(Gate::single(GateType::Ry(0.5), 0));
        circuit.add(Gate::single(GateType::Rz(0.5), 0));

        // Two-qubit gates
        circuit.add(Gate::two(GateType::CNOT, 0, 1));
        circuit.add(Gate::two(GateType::CZ, 1, 2));
        circuit.add(Gate::two(GateType::SWAP, 0, 2));

        // Barrier
        circuit.add(Gate::barrier(vec![0, 1, 2]));

        let qasm = QASMTranspiler::transpile(&circuit, &[]).unwrap();

        assert!(qasm.contains("id q[0];"));
        assert!(qasm.contains("x q[0];"));
        assert!(qasm.contains("y q[0];"));
        assert!(qasm.contains("z q[0];"));
        assert!(qasm.contains("h q[0];"));
        assert!(qasm.contains("s q[0];"));
        assert!(qasm.contains("sdg q[0];"));
        assert!(qasm.contains("t q[0];"));
        assert!(qasm.contains("tdg q[0];"));
        assert!(qasm.contains("sx q[0];"));
        assert!(qasm.contains("rx(0.5) q[0];"));
        assert!(qasm.contains("ry(0.5) q[0];"));
        assert!(qasm.contains("rz(0.5) q[0];"));
        assert!(qasm.contains("cx q[0], q[1];"));
        assert!(qasm.contains("cz q[1], q[2];"));
        assert!(qasm.contains("swap q[0], q[2];"));
        assert!(qasm.contains("barrier"));
    }
}

use tqp_core::memory::{MemoryManager, MemoryPolicy};
use tqp_core::scheduler::{AdaptiveScheduler, Operation, ScheduleConstraints, ScheduleRequest};
use tqp_core::state::{TQPDimensions, TQPState};

#[test]
fn test_adaptive_scheduler_basic() {
    let dims = TQPDimensions::new(2, 5, 1);
    let scheduler = AdaptiveScheduler::new(dims);

    let op1 = Operation::Gate {
        name: "H".to_string(),
        qubits: vec![0],
        params: vec![],
    };
    let op2 = Operation::Gate {
        name: "X".to_string(),
        qubits: vec![1],
        params: vec![],
    };
    // CNOT depends on 0 and 1. Should be scheduled after op1 and op2.
    let op3 = Operation::Gate {
        name: "CNOT".to_string(),
        qubits: vec![0, 1],
        params: vec![],
    };

    let req = ScheduleRequest {
        circuit_id: "test_circuit".to_string(),
        operations: vec![op1, op2, op3],
        constraints: ScheduleConstraints {
            max_depth: 10,
            max_crosstalk: 0.1,
        },
    };

    let result = scheduler.optimize_schedule(req);

    assert_eq!(result.scheduled_ops.len(), 3);

    // Check timing
    // op1 and op2 can be at time 0
    // op3 must be at time 1 (since it uses q0 and q1 which are used at time 0)

    let scheduled_op1 = &result.scheduled_ops[0];
    let scheduled_op2 = &result.scheduled_ops[1];
    let scheduled_op3 = &result.scheduled_ops[2];

    assert_eq!(scheduled_op1.time_bin, 0);
    assert_eq!(scheduled_op2.time_bin, 0);
    assert_eq!(scheduled_op3.time_bin, 1);
}

#[test]
fn test_memory_manager_policy() {
    let mut manager = MemoryManager::new();
    let state = TQPState::new(2, 2, 1); // Small state

    let policy = manager.decide_policy(&state);
    assert_eq!(policy, MemoryPolicy::Dense);

    // Test optimization (should not change for small state)
    let mut state_mut = TQPState::new(2, 2, 1);
    manager.optimize_storage(&mut state_mut);
    assert_eq!(manager.current_policy, MemoryPolicy::Dense);
}

pub mod autodiff; // Automatic differentiation (parameter-shift rule)
pub mod chunk_strategy; // TQP 3D structure parallel chunking
pub mod dispatcher; // Unified dispatcher for automatic backend selection
pub mod gradient_cache; // Gradient caching for VQE optimization
pub mod gradient_tape; // PyTorch-style gradient tape
pub mod hardware;
pub mod inplace; // In-place optimization (double buffer, view mutation)
pub mod layers;
pub mod memory;
pub mod memory_policy; // PPR-based memory management
pub mod noise;
pub mod ops;
pub mod ops_simd; // AVX-512 optimized operations
pub mod optimizer; // VQE/QAOA optimizers (SGD, Adam, SPSA)
pub mod parallel_sparse; // Rayon-based parallel operations
pub mod pulse;
pub mod scheduler;
pub mod simd_avx2; // AVX2 SIMD optimized operations
pub mod sparse; // Sparse state vector representation
pub mod sparse_ops; // Sparse quantum operations
pub mod state;

pub use autodiff::{
    compute_expectation, compute_expectation_and_gradient, compute_gradient, rx, ry, rz, u3,
    verify_gradient, GateType, Hamiltonian, ParameterizedGate, PauliObservable, VariationalCircuit,
    GRADIENT_EPSILON, PARAMETER_SHIFT,
};
pub use chunk_strategy::{
    AtomicStateBuffer, BarrierSync, BinParallel, ChunkResult, ChunkStrategy, ChunkStrategyKind,
    LayerParallel, SpatialParallel, StateChunk, DEFAULT_SPATIAL_CHUNK_SIZE,
};
pub use dispatcher::{
    dispatch_circuit, dispatch_gate_1q, dispatch_gate_2q, select_backend, Backend, CircuitGate,
    DispatchStats, DispatcherConfig, ExecutionBackend, QuantumState, StateBackend,
    PARALLEL_THRESHOLD, SIMD_MIN_DIM, SPARSE_THRESHOLD,
};
pub use inplace::{
    apply_gate_sparse_pooled, apply_gates_dense_batched, BufferPool, BufferedSparseState,
    DoubleBuffer,
};
pub use memory_policy::{ManagedState, MemoryPolicy, PolicyAction, PolicyConfig, UnifiedState};
pub use ops_simd::apply_spatial_gate_optimized;
pub use parallel_sparse::{
    configure_thread_pool, execute_circuit, get_thread_count, parallel_apply_gate_2q_sparse,
    parallel_apply_gate_sparse, parallel_compute_stats, parallel_execute_circuits,
    parallel_expval_z, parallel_fidelity, parallel_inner_product, parallel_normalize,
    parallel_total_probability, GateOp, ParallelStats,
};
pub use simd_avx2::{
    apply_gate_1q_simd, apply_gate_2q_simd, apply_gates_batch_simd, has_avx2, has_avx512,
    total_probability_simd, SimdLevel,
};
pub use sparse::SparseStateVector;
pub use sparse_ops::{
    apply_cnot_sparse, apply_cz_sparse, apply_gate_2q_sparse, apply_gate_sparse, apply_swap_sparse,
    expval_z_sparse, fidelity_sparse, gates, inner_product_sparse, measure_qubit_sparse,
    measure_sparse,
};
pub use state::TQPState;

// Sprint 2 Week 3: AutoDiff subsystem
pub use gradient_cache::{
    BatchGradientCache, CacheEntry, CacheKey, CacheStats, CachedGradientComputer, GradientCache,
    DEFAULT_CACHE_CAPACITY, DEFAULT_QUANTIZATION_BITS, DEFAULT_TOLERANCE,
};
pub use gradient_tape::{
    shared_tape, CachedValues, GradientAccumulator, GradientTape, OperationType, SharedTape,
    TapeContext, TapeEntry, TapeSummary, DEFAULT_ACCUMULATE, VJP_SHIFT,
};
pub use optimizer::{
    minimize, minimize_spsa, AdaGradState, AdamState, LRSchedule, MomentumState,
    OptimizationResult, Optimizer, OptimizerConfig, OptimizerType, RMSpropState, SPSAState,
    DEFAULT_BETA1, DEFAULT_BETA2, DEFAULT_LEARNING_RATE, DEFAULT_MOMENTUM,
};

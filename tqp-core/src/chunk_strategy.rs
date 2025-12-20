//! Chunk Strategy for TQP Parallel Processing
//!
//! Provides parallel processing strategies based on TQP's 3D structure:
//! - H_total = H_layer(L) ⊗ H_time(M) ⊗ H_spatial(2^N)
//!
//! # Index Mapping
//! ```text
//! global_idx = layer * (M * 2^N) + bin * 2^N + spatial
//! ```
//!
//! # Strategies
//! 1. **LayerParallel**: Process each layer independently (L chunks)
//! 2. **BinParallel**: Process each time-bin independently (M chunks)
//! 3. **SpatialParallel**: Chunk spatial indices (configurable chunk size)
//!
//! # Use Cases
//! - LayerParallel: Best when L is large and layers are independent
//! - BinParallel: Best when M is large and operations are bin-local
//! - SpatialParallel: Best for large spatial dimension (2^N)

use ndarray::Array1;
use num_complex::Complex64;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::state::{TQPDimensions, TQPState};

// =============================================================================
// Chunk Types
// =============================================================================

/// Represents a chunk of the state vector for parallel processing
#[derive(Debug, Clone)]
pub struct StateChunk {
    /// Start index in global state vector
    pub start: usize,
    /// End index (exclusive)
    pub end: usize,
    /// Chunk identifier (layer/bin/spatial chunk index)
    pub chunk_id: usize,
    /// Amplitudes in this chunk
    pub data: Vec<Complex64>,
}

impl StateChunk {
    /// Create a new chunk from a slice of the state vector
    pub fn from_state(state: &TQPState, start: usize, end: usize, chunk_id: usize) -> Self {
        let data: Vec<Complex64> = state
            .state_vector
            .slice(ndarray::s![start..end])
            .iter()
            .copied()
            .collect();

        StateChunk {
            start,
            end,
            chunk_id,
            data,
        }
    }

    /// Number of elements in this chunk
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// Check if chunk is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Result of processing a chunk
#[derive(Debug)]
pub struct ChunkResult {
    /// Chunk identifier
    pub chunk_id: usize,
    /// Start index in global state
    pub start: usize,
    /// Processed amplitudes
    pub data: Vec<Complex64>,
}

// =============================================================================
// Layer Parallel Strategy
// =============================================================================

/// Layer-parallel processing strategy
///
/// Splits the state vector by layers (L chunks).
/// Each layer contains M * 2^N elements.
///
/// Best for:
/// - Large L (many layers)
/// - Layer-independent operations
/// - DeepLogic circuits
pub struct LayerParallel {
    dims: TQPDimensions,
}

impl LayerParallel {
    pub fn new(dims: TQPDimensions) -> Self {
        LayerParallel { dims }
    }

    /// Split state into L layer chunks
    pub fn split(&self, state: &TQPState) -> Vec<StateChunk> {
        let layer_size = self.dims.num_time_bins * self.dims.spatial_dim();

        (0..self.dims.num_layers)
            .map(|layer| {
                let start = layer * layer_size;
                let end = start + layer_size;
                StateChunk::from_state(state, start, end, layer)
            })
            .collect()
    }

    /// Split state by layer indices (returns index ranges)
    pub fn split_by_layer(&self) -> Vec<(usize, usize, usize)> {
        let layer_size = self.dims.num_time_bins * self.dims.spatial_dim();

        (0..self.dims.num_layers)
            .map(|layer| {
                let start = layer * layer_size;
                let end = start + layer_size;
                (layer, start, end)
            })
            .collect()
    }

    /// Process chunks in parallel with a user-defined function
    pub fn process_parallel<F>(&self, state: &TQPState, f: F) -> Vec<ChunkResult>
    where
        F: Fn(&StateChunk) -> Vec<Complex64> + Sync + Send,
    {
        let chunks = self.split(state);

        chunks
            .into_par_iter()
            .map(|chunk| {
                let processed = f(&chunk);
                ChunkResult {
                    chunk_id: chunk.chunk_id,
                    start: chunk.start,
                    data: processed,
                }
            })
            .collect()
    }

    /// Join chunk results back into a state vector
    pub fn join_layer_results(&self, results: Vec<ChunkResult>) -> Array1<Complex64> {
        let total_dim = self.dims.total_dim();
        let mut output = Array1::<Complex64>::zeros(total_dim);

        // Sort by chunk_id to ensure correct ordering
        let mut sorted_results = results;
        sorted_results.sort_by_key(|r| r.chunk_id);

        for result in sorted_results {
            for (i, &amp) in result.data.iter().enumerate() {
                output[result.start + i] = amp;
            }
        }

        output
    }
}

// =============================================================================
// Bin Parallel Strategy
// =============================================================================

/// Time-bin parallel processing strategy
///
/// Splits the state vector by time bins (M chunks per layer).
/// Total chunks = L * M, each containing 2^N elements.
///
/// Best for:
/// - Large M (many time bins)
/// - Bin-local operations (FastMux processing)
pub struct BinParallel {
    dims: TQPDimensions,
}

impl BinParallel {
    pub fn new(dims: TQPDimensions) -> Self {
        BinParallel { dims }
    }

    /// Split state into M bins per layer (total L*M chunks)
    pub fn split(&self, state: &TQPState) -> Vec<StateChunk> {
        let spatial_dim = self.dims.spatial_dim();
        let mut chunks = Vec::with_capacity(self.dims.num_layers * self.dims.num_time_bins);

        for layer in 0..self.dims.num_layers {
            for bin in 0..self.dims.num_time_bins {
                let start = layer * (self.dims.num_time_bins * spatial_dim) + bin * spatial_dim;
                let end = start + spatial_dim;
                let chunk_id = layer * self.dims.num_time_bins + bin;
                chunks.push(StateChunk::from_state(state, start, end, chunk_id));
            }
        }

        chunks
    }

    /// Split by bin indices (returns layer, bin, start, end)
    pub fn split_by_bin(&self) -> Vec<(usize, usize, usize, usize)> {
        let spatial_dim = self.dims.spatial_dim();
        let mut ranges = Vec::with_capacity(self.dims.num_layers * self.dims.num_time_bins);

        for layer in 0..self.dims.num_layers {
            for bin in 0..self.dims.num_time_bins {
                let start = layer * (self.dims.num_time_bins * spatial_dim) + bin * spatial_dim;
                let end = start + spatial_dim;
                ranges.push((layer, bin, start, end));
            }
        }

        ranges
    }

    /// Process bin chunks in parallel
    pub fn process_parallel<F>(&self, state: &TQPState, f: F) -> Vec<ChunkResult>
    where
        F: Fn(&StateChunk, usize, usize) -> Vec<Complex64> + Sync + Send,
    {
        let _spatial_dim = self.dims.spatial_dim();
        let num_bins = self.dims.num_time_bins;
        let chunks = self.split(state);

        chunks
            .into_par_iter()
            .map(|chunk| {
                let layer = chunk.chunk_id / num_bins;
                let bin = chunk.chunk_id % num_bins;
                let processed = f(&chunk, layer, bin);
                ChunkResult {
                    chunk_id: chunk.chunk_id,
                    start: chunk.start,
                    data: processed,
                }
            })
            .collect()
    }

    /// Join bin results
    pub fn join_bin_results(&self, results: Vec<ChunkResult>) -> Array1<Complex64> {
        let total_dim = self.dims.total_dim();
        let mut output = Array1::<Complex64>::zeros(total_dim);

        for result in results {
            for (i, &amp) in result.data.iter().enumerate() {
                output[result.start + i] = amp;
            }
        }

        output
    }
}

// =============================================================================
// Spatial Parallel Strategy
// =============================================================================

/// Default spatial chunk size (tuned for cache efficiency)
pub const DEFAULT_SPATIAL_CHUNK_SIZE: usize = 1024;

/// Spatial parallel processing strategy
///
/// Chunks the spatial dimension (2^N) into configurable sizes.
/// Uses atomic operations for safe concurrent updates.
///
/// Best for:
/// - Large N (many qubits)
/// - Operations that span multiple qubits
/// - Fine-grained parallelism
pub struct SpatialParallel {
    dims: TQPDimensions,
    chunk_size: usize,
}

impl SpatialParallel {
    pub fn new(dims: TQPDimensions) -> Self {
        SpatialParallel {
            dims,
            chunk_size: DEFAULT_SPATIAL_CHUNK_SIZE,
        }
    }

    pub fn with_chunk_size(dims: TQPDimensions, chunk_size: usize) -> Self {
        SpatialParallel { dims, chunk_size }
    }

    /// Split spatial dimension into chunks
    pub fn split_by_spatial(&self) -> Vec<(usize, usize)> {
        let spatial_dim = self.dims.spatial_dim();
        let mut ranges = Vec::new();

        let mut start = 0;
        while start < spatial_dim {
            let end = (start + self.chunk_size).min(spatial_dim);
            ranges.push((start, end));
            start = end;
        }

        ranges
    }

    /// Get number of spatial chunks
    pub fn num_chunks(&self) -> usize {
        self.dims.spatial_dim().div_ceil(self.chunk_size)
    }

    /// Process spatial chunks in parallel for a single layer/bin
    pub fn process_layer_bin_parallel<F>(
        &self,
        state: &TQPState,
        layer: usize,
        bin: usize,
        f: F,
    ) -> Vec<Complex64>
    where
        F: Fn(usize, Complex64) -> Complex64 + Sync + Send,
    {
        let spatial_dim = self.dims.spatial_dim();
        let base_idx = layer * (self.dims.num_time_bins * spatial_dim) + bin * spatial_dim;

        let chunks = self.split_by_spatial();

        // Process chunks in parallel
        let results: Vec<Vec<Complex64>> = chunks
            .par_iter()
            .map(|&(start, end)| {
                (start..end)
                    .map(|spatial_idx| {
                        let global_idx = base_idx + spatial_idx;
                        let amp = state.state_vector[global_idx];
                        f(spatial_idx, amp)
                    })
                    .collect()
            })
            .collect();

        // Flatten results
        results.into_iter().flatten().collect()
    }

    /// Atomic update structure for concurrent writes
    pub fn create_atomic_buffer(&self) -> AtomicStateBuffer {
        AtomicStateBuffer::new(self.dims.total_dim())
    }
}

/// Thread-safe buffer for atomic state updates
pub struct AtomicStateBuffer {
    /// Real parts stored as u64 (bit representation)
    real: Vec<AtomicU64>,
    /// Imaginary parts stored as u64 (bit representation)
    imag: Vec<AtomicU64>,
}

impl AtomicStateBuffer {
    pub fn new(size: usize) -> Self {
        let real: Vec<AtomicU64> = (0..size)
            .map(|_| AtomicU64::new(0.0_f64.to_bits()))
            .collect();
        let imag: Vec<AtomicU64> = (0..size)
            .map(|_| AtomicU64::new(0.0_f64.to_bits()))
            .collect();

        AtomicStateBuffer { real, imag }
    }

    /// Atomically add to amplitude at index
    pub fn atomic_add(&self, index: usize, value: Complex64) {
        // Atomic add for real part
        loop {
            let old_bits = self.real[index].load(Ordering::Relaxed);
            let old_val = f64::from_bits(old_bits);
            let new_val = old_val + value.re;
            let new_bits = new_val.to_bits();

            if self.real[index]
                .compare_exchange_weak(old_bits, new_bits, Ordering::SeqCst, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }

        // Atomic add for imaginary part
        loop {
            let old_bits = self.imag[index].load(Ordering::Relaxed);
            let old_val = f64::from_bits(old_bits);
            let new_val = old_val + value.im;
            let new_bits = new_val.to_bits();

            if self.imag[index]
                .compare_exchange_weak(old_bits, new_bits, Ordering::SeqCst, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }
    }

    /// Set amplitude at index (non-atomic, for initialization)
    pub fn set(&self, index: usize, value: Complex64) {
        self.real[index].store(value.re.to_bits(), Ordering::Relaxed);
        self.imag[index].store(value.im.to_bits(), Ordering::Relaxed);
    }

    /// Get amplitude at index
    pub fn get(&self, index: usize) -> Complex64 {
        let re = f64::from_bits(self.real[index].load(Ordering::Relaxed));
        let im = f64::from_bits(self.imag[index].load(Ordering::Relaxed));
        Complex64::new(re, im)
    }

    /// Convert to Array1
    pub fn to_array(&self) -> Array1<Complex64> {
        let size = self.real.len();
        let mut arr = Array1::<Complex64>::zeros(size);

        for i in 0..size {
            arr[i] = self.get(i);
        }

        arr
    }

    /// Convert to TQPState
    pub fn to_state(&self, dims: TQPDimensions) -> TQPState {
        TQPState {
            dims,
            state_vector: self.to_array(),
        }
    }
}

// =============================================================================
// Barrier Synchronization
// =============================================================================

use std::sync::{Arc, Barrier};

/// Barrier-synchronized parallel processing
pub struct BarrierSync {
    barrier: Arc<Barrier>,
    num_threads: usize,
}

impl BarrierSync {
    pub fn new(num_threads: usize) -> Self {
        BarrierSync {
            barrier: Arc::new(Barrier::new(num_threads)),
            num_threads,
        }
    }

    /// Get a clone of the barrier for a thread
    pub fn get_barrier(&self) -> Arc<Barrier> {
        Arc::clone(&self.barrier)
    }

    /// Execute with barrier synchronization
    pub fn execute_with_barrier<F, T>(&self, f: F) -> Vec<T>
    where
        F: Fn(usize, Arc<Barrier>) -> T + Sync + Send,
        T: Send,
    {
        (0..self.num_threads)
            .into_par_iter()
            .map(|thread_id| {
                let barrier = self.get_barrier();
                f(thread_id, barrier)
            })
            .collect()
    }
}

// =============================================================================
// Unified Chunk Strategy
// =============================================================================

/// Strategy selection for automatic chunk selection
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChunkStrategyKind {
    /// Layer-based parallelism
    Layer,
    /// Time-bin based parallelism
    Bin,
    /// Spatial chunk parallelism
    Spatial,
    /// Automatic selection based on dimensions
    Auto,
}

/// Unified chunk strategy that selects the best approach
pub struct ChunkStrategy {
    dims: TQPDimensions,
    kind: ChunkStrategyKind,
}

impl ChunkStrategy {
    pub fn new(dims: TQPDimensions, kind: ChunkStrategyKind) -> Self {
        ChunkStrategy { dims, kind }
    }

    /// Create with automatic strategy selection
    pub fn auto(dims: TQPDimensions) -> Self {
        ChunkStrategy {
            dims,
            kind: ChunkStrategyKind::Auto,
        }
    }

    /// Select the best strategy based on dimensions
    pub fn select_strategy(&self) -> ChunkStrategyKind {
        if self.kind != ChunkStrategyKind::Auto {
            return self.kind;
        }

        let l = self.dims.num_layers;
        let m = self.dims.num_time_bins;
        let spatial = self.dims.spatial_dim();

        // Heuristics for strategy selection:
        // - If L >= 4: Layer parallelism (good chunk granularity)
        // - If M >= 8: Bin parallelism (many independent bins)
        // - Otherwise: Spatial parallelism (default for large 2^N)

        if l >= 4 && l >= m {
            ChunkStrategyKind::Layer
        } else if m >= 8 && m > l {
            ChunkStrategyKind::Bin
        } else if spatial >= 1024 {
            ChunkStrategyKind::Spatial
        } else {
            // Default to bin for small states
            ChunkStrategyKind::Bin
        }
    }

    /// Get recommended number of chunks
    pub fn recommended_chunks(&self) -> usize {
        match self.select_strategy() {
            ChunkStrategyKind::Layer => self.dims.num_layers,
            ChunkStrategyKind::Bin => self.dims.num_layers * self.dims.num_time_bins,
            ChunkStrategyKind::Spatial => {
                let spatial = SpatialParallel::new(self.dims);
                spatial.num_chunks() * self.dims.num_layers * self.dims.num_time_bins
            }
            ChunkStrategyKind::Auto => self.recommended_chunks(), // Recursive with resolved strategy
        }
    }

    /// Process state with selected strategy
    pub fn process<F>(&self, state: &TQPState, f: F) -> TQPState
    where
        F: Fn(&[Complex64]) -> Vec<Complex64> + Sync + Send + Clone,
    {
        match self.select_strategy() {
            ChunkStrategyKind::Layer => {
                let layer_par = LayerParallel::new(self.dims);
                let results = layer_par.process_parallel(state, |chunk| f(&chunk.data));
                let output = layer_par.join_layer_results(results);
                TQPState {
                    dims: self.dims,
                    state_vector: output,
                }
            }
            ChunkStrategyKind::Bin => {
                let bin_par = BinParallel::new(self.dims);
                let results = bin_par.process_parallel(state, |chunk, _layer, _bin| f(&chunk.data));
                let output = bin_par.join_bin_results(results);
                TQPState {
                    dims: self.dims,
                    state_vector: output,
                }
            }
            ChunkStrategyKind::Spatial => {
                // For spatial, we process each layer/bin combination
                let mut output = Array1::<Complex64>::zeros(self.dims.total_dim());
                let spatial_dim = self.dims.spatial_dim();

                let ranges: Vec<(usize, usize, usize)> = (0..self.dims.num_layers)
                    .flat_map(|layer| {
                        (0..self.dims.num_time_bins).map(move |bin| {
                            let base =
                                layer * (self.dims.num_time_bins * spatial_dim) + bin * spatial_dim;
                            (layer, bin, base)
                        })
                    })
                    .collect();

                let results: Vec<(usize, Vec<Complex64>)> = ranges
                    .par_iter()
                    .map(|&(_layer, _bin, base)| {
                        let chunk: Vec<Complex64> = (0..spatial_dim)
                            .map(|i| state.state_vector[base + i])
                            .collect();
                        (base, f(&chunk))
                    })
                    .collect();

                for (base, data) in results {
                    for (i, amp) in data.into_iter().enumerate() {
                        output[base + i] = amp;
                    }
                }

                TQPState {
                    dims: self.dims,
                    state_vector: output,
                }
            }
            ChunkStrategyKind::Auto => unreachable!("Auto should be resolved"),
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    fn create_test_state(n: usize, m: usize, l: usize) -> TQPState {
        let mut state = TQPState::new(n, m, l);
        // Initialize with sequential values for testing
        for i in 0..state.dimension() {
            state.state_vector[i] = Complex64::new(i as f64, 0.0);
        }
        state.normalize();
        state
    }

    #[test]
    fn test_layer_parallel_split() {
        let state = create_test_state(3, 2, 4); // 8 spatial, 2 bins, 4 layers
        let layer_par = LayerParallel::new(state.dims);

        let chunks = layer_par.split(&state);

        assert_eq!(chunks.len(), 4, "Should have 4 layer chunks");

        // Each chunk should have M * 2^N = 2 * 8 = 16 elements
        for chunk in &chunks {
            assert_eq!(chunk.len(), 16, "Each layer chunk should have 16 elements");
        }

        // Verify non-overlapping coverage
        let total_elements: usize = chunks.iter().map(|c| c.len()).sum();
        assert_eq!(total_elements, state.dimension());
    }

    #[test]
    fn test_layer_parallel_process() {
        let state = create_test_state(3, 2, 4);
        let layer_par = LayerParallel::new(state.dims);

        // Simple identity processing
        let results = layer_par.process_parallel(&state, |chunk| chunk.data.clone());

        let output = layer_par.join_layer_results(results);

        // Should reconstruct original state
        for i in 0..state.dimension() {
            let diff = (state.state_vector[i] - output[i]).norm();
            assert!(diff < EPSILON, "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_bin_parallel_split() {
        let state = create_test_state(3, 4, 2); // 8 spatial, 4 bins, 2 layers
        let bin_par = BinParallel::new(state.dims);

        let chunks = bin_par.split(&state);

        assert_eq!(chunks.len(), 8, "Should have L*M = 2*4 = 8 bin chunks");

        // Each chunk should have 2^N = 8 elements
        for chunk in &chunks {
            assert_eq!(chunk.len(), 8, "Each bin chunk should have 8 elements");
        }
    }

    #[test]
    fn test_bin_parallel_process() {
        let state = create_test_state(3, 4, 2);
        let bin_par = BinParallel::new(state.dims);

        // Double all amplitudes
        let results = bin_par.process_parallel(&state, |chunk, _layer, _bin| {
            chunk.data.iter().map(|&c| c * 2.0).collect()
        });

        let output = bin_par.join_bin_results(results);

        for i in 0..state.dimension() {
            let expected = state.state_vector[i] * 2.0;
            let diff = (expected - output[i]).norm();
            assert!(diff < EPSILON, "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_spatial_parallel_split() {
        let dims = TQPDimensions::new(10, 1, 1); // 1024 spatial
        let spatial_par = SpatialParallel::with_chunk_size(dims, 256);

        let ranges = spatial_par.split_by_spatial();

        assert_eq!(ranges.len(), 4, "Should have 1024/256 = 4 chunks");

        // Verify coverage
        assert_eq!(ranges[0], (0, 256));
        assert_eq!(ranges[1], (256, 512));
        assert_eq!(ranges[2], (512, 768));
        assert_eq!(ranges[3], (768, 1024));
    }

    #[test]
    fn test_spatial_parallel_process() {
        let state = create_test_state(8, 1, 1); // 256 spatial
        let spatial_par = SpatialParallel::with_chunk_size(state.dims, 64);

        // Negate all amplitudes
        let result = spatial_par.process_layer_bin_parallel(&state, 0, 0, |_idx, amp| -amp);

        assert_eq!(result.len(), 256);

        for i in 0..256 {
            let expected = -state.state_vector[i];
            let diff = (expected - result[i]).norm();
            assert!(diff < EPSILON, "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_atomic_buffer() {
        let buffer = AtomicStateBuffer::new(100);

        // Set initial value
        buffer.set(50, Complex64::new(1.0, 2.0));

        // Atomic add
        buffer.atomic_add(50, Complex64::new(0.5, -0.5));

        let result = buffer.get(50);
        assert!((result.re - 1.5).abs() < EPSILON);
        assert!((result.im - 1.5).abs() < EPSILON);
    }

    #[test]
    fn test_atomic_buffer_parallel() {
        let buffer = AtomicStateBuffer::new(1000);

        // Parallel atomic adds to same index
        (0..1000).into_par_iter().for_each(|_| {
            buffer.atomic_add(500, Complex64::new(0.001, 0.001));
        });

        let result = buffer.get(500);
        assert!(
            (result.re - 1.0).abs() < 0.01,
            "Real part should sum to ~1.0"
        );
        assert!(
            (result.im - 1.0).abs() < 0.01,
            "Imag part should sum to ~1.0"
        );
    }

    #[test]
    fn test_barrier_sync() {
        let sync = BarrierSync::new(4);

        let results: Vec<usize> = sync.execute_with_barrier(|thread_id, barrier| {
            // Phase 1: compute
            let val = thread_id * 10;

            // Synchronize
            barrier.wait();

            // Phase 2: return
            val
        });

        assert_eq!(results.len(), 4);
        let sum: usize = results.iter().sum();
        assert_eq!(sum, 0 + 10 + 20 + 30);
    }

    #[test]
    fn test_chunk_strategy_auto_selection() {
        // Large L -> Layer
        let dims1 = TQPDimensions::new(5, 2, 8);
        let strategy1 = ChunkStrategy::auto(dims1);
        assert_eq!(strategy1.select_strategy(), ChunkStrategyKind::Layer);

        // Large M -> Bin
        let dims2 = TQPDimensions::new(5, 16, 2);
        let strategy2 = ChunkStrategy::auto(dims2);
        assert_eq!(strategy2.select_strategy(), ChunkStrategyKind::Bin);

        // Large N -> Spatial
        let dims3 = TQPDimensions::new(12, 2, 2);
        let strategy3 = ChunkStrategy::auto(dims3);
        assert_eq!(strategy3.select_strategy(), ChunkStrategyKind::Spatial);
    }

    #[test]
    fn test_chunk_strategy_unified_process() {
        let state = create_test_state(4, 4, 2);

        // Test with layer strategy
        let layer_strategy = ChunkStrategy::new(state.dims, ChunkStrategyKind::Layer);
        let result =
            layer_strategy.process(&state, |chunk| chunk.iter().map(|&c| c * 2.0).collect());

        for i in 0..state.dimension() {
            let expected = state.state_vector[i] * 2.0;
            let diff = (expected - result.state_vector[i]).norm();
            assert!(diff < EPSILON, "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_performance_comparison() {
        use std::time::Instant;

        let state = create_test_state(10, 4, 4); // 1024 * 4 * 4 = 16384 elements

        println!("\n=== Chunk Strategy Performance ===");
        println!("State size: {} elements", state.dimension());

        // Layer parallel
        let layer_par = LayerParallel::new(state.dims);
        let start = Instant::now();
        let _ = layer_par.process_parallel(&state, |chunk| chunk.data.clone());
        let layer_time = start.elapsed();
        println!("Layer parallel (copy): {:?}", layer_time);

        // Bin parallel
        let bin_par = BinParallel::new(state.dims);
        let start = Instant::now();
        let _ = bin_par.process_parallel(&state, |chunk, _, _| chunk.data.clone());
        let bin_time = start.elapsed();
        println!("Bin parallel (copy): {:?}", bin_time);

        // Sequential baseline
        let start = Instant::now();
        let _: Vec<Complex64> = state.state_vector.iter().map(|&c| c).collect();
        let seq_time = start.elapsed();
        println!("Sequential (copy): {:?}", seq_time);
    }

    /// Test with compute-intensive workload
    #[test]
    fn test_compute_intensive_performance() {
        use std::time::Instant;

        let state = create_test_state(12, 4, 4); // 4096 * 4 * 4 = 65536 elements

        println!("\n=== Compute-Intensive Performance ===");
        println!("State size: {} elements", state.dimension());

        // Simulate expensive computation (multiple trig operations)
        let expensive_op = |chunk: &[Complex64]| -> Vec<Complex64> {
            chunk
                .iter()
                .map(|&c| {
                    // Simulate complex computation
                    let r = c.norm();
                    let theta = c.arg();
                    let new_r = (r * 10.0).sin().powi(2);
                    let new_theta = theta + std::f64::consts::PI / 4.0;
                    Complex64::from_polar(new_r, new_theta)
                })
                .collect()
        };

        // Sequential
        let start = Instant::now();
        let _: Vec<Complex64> = expensive_op(&state.state_vector.to_vec());
        let seq_time = start.elapsed();

        // Layer parallel
        let layer_par = LayerParallel::new(state.dims);
        let start = Instant::now();
        let _ = layer_par.process_parallel(&state, |chunk| expensive_op(&chunk.data));
        let layer_time = start.elapsed();

        // Bin parallel
        let bin_par = BinParallel::new(state.dims);
        let start = Instant::now();
        let _ = bin_par.process_parallel(&state, |chunk, _, _| expensive_op(&chunk.data));
        let bin_time = start.elapsed();

        println!("Sequential: {:?}", seq_time);
        println!(
            "Layer parallel: {:?} (speedup: {:.2}x)",
            layer_time,
            seq_time.as_secs_f64() / layer_time.as_secs_f64()
        );
        println!(
            "Bin parallel: {:?} (speedup: {:.2}x)",
            bin_time,
            seq_time.as_secs_f64() / bin_time.as_secs_f64()
        );
    }
}

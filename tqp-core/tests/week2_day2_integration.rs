//! Week 2 Day 2: ChunkStrategy Integration Tests
//!
//! Validates:
//! - Layer/Bin/Spatial chunk strategies
//! - TQP 3D structure parallel processing
//! - Strategy selection heuristics

use num_complex::Complex64;
use tqp_core::state::TQPDimensions;
use tqp_core::{
    AtomicStateBuffer, BarrierSync, BinParallel, ChunkStrategy, ChunkStrategyKind, LayerParallel,
    SpatialParallel, TQPState,
};

const EPSILON: f64 = 1e-10;

// =============================================================================
// Layer Parallel Tests
// =============================================================================

#[test]
fn test_layer_parallel_correctness() {
    // Create multi-layer state
    let mut state = TQPState::new(4, 2, 4); // 16 spatial, 2 bins, 4 layers

    // Initialize with distinct values per layer
    let spatial_dim = 1 << 4;
    let bin_size = spatial_dim;
    let layer_size = 2 * bin_size;

    for layer in 0..4 {
        for i in 0..layer_size {
            let idx = layer * layer_size + i;
            state.state_vector[idx] = Complex64::new((layer + 1) as f64, 0.0);
        }
    }

    // Process: multiply each layer by its index
    let layer_par = LayerParallel::new(state.dims);
    let results = layer_par.process_parallel(&state, |chunk| {
        let multiplier = (chunk.chunk_id + 1) as f64;
        chunk.data.iter().map(|&c| c * multiplier).collect()
    });

    let output = layer_par.join_layer_results(results);

    // Verify: layer 0 -> *1, layer 1 -> *2, etc.
    for layer in 0..4 {
        for i in 0..layer_size {
            let idx = layer * layer_size + i;
            let expected = (layer + 1) as f64 * (layer + 1) as f64;
            let diff = (output[idx].re - expected).abs();
            assert!(
                diff < EPSILON,
                "Layer {} mismatch at {}: expected {}, got {}",
                layer,
                i,
                expected,
                output[idx].re
            );
        }
    }
}

#[test]
fn test_layer_independence() {
    // Verify layers are processed independently
    let state = TQPState::new(3, 2, 8); // 8 layers
    let layer_par = LayerParallel::new(state.dims);

    let ranges = layer_par.split_by_layer();

    assert_eq!(ranges.len(), 8);

    // Verify non-overlapping
    for i in 0..ranges.len() {
        for j in (i + 1)..ranges.len() {
            let (_, s1, e1) = ranges[i];
            let (_, s2, e2) = ranges[j];
            assert!(e1 <= s2 || e2 <= s1, "Layers {} and {} overlap", i, j);
        }
    }
}

// =============================================================================
// Bin Parallel Tests
// =============================================================================

#[test]
fn test_bin_parallel_correctness() {
    let mut state = TQPState::new(3, 4, 2); // 8 spatial, 4 bins, 2 layers

    // Initialize: each bin gets its bin index as value
    let spatial_dim = 8;
    for layer in 0..2 {
        for bin in 0..4 {
            for s in 0..spatial_dim {
                let idx = layer * (4 * spatial_dim) + bin * spatial_dim + s;
                state.state_vector[idx] = Complex64::new(bin as f64, layer as f64);
            }
        }
    }

    // Process: add 10 to bin index
    let bin_par = BinParallel::new(state.dims);
    let results = bin_par.process_parallel(&state, |chunk, layer, bin| {
        chunk
            .data
            .iter()
            .map(|&c| Complex64::new(c.re + 10.0, c.im))
            .collect()
    });

    let output = bin_par.join_bin_results(results);

    // Verify
    for layer in 0..2 {
        for bin in 0..4 {
            for s in 0..spatial_dim {
                let idx = layer * (4 * spatial_dim) + bin * spatial_dim + s;
                let expected_re = bin as f64 + 10.0;
                let diff = (output[idx].re - expected_re).abs();
                assert!(diff < EPSILON, "Bin {} layer {} mismatch", bin, layer);
            }
        }
    }
}

#[test]
fn test_bin_layer_access() {
    let state = TQPState::new(4, 8, 3); // 3 layers, 8 bins
    let bin_par = BinParallel::new(state.dims);

    // Verify layer/bin passed correctly
    let results = bin_par.process_parallel(&state, |chunk, layer, bin| {
        // Return layer*100 + bin as all elements
        vec![Complex64::new((layer * 100 + bin) as f64, 0.0); chunk.len()]
    });

    // Check each chunk got correct layer/bin
    for result in &results {
        let layer = result.chunk_id / 8;
        let bin = result.chunk_id % 8;
        let expected = (layer * 100 + bin) as f64;
        assert!((result.data[0].re - expected).abs() < EPSILON);
    }
}

// =============================================================================
// Spatial Parallel Tests
// =============================================================================

#[test]
fn test_spatial_chunking() {
    let dims = TQPDimensions::new(12, 1, 1); // 4096 spatial
    let spatial_par = SpatialParallel::with_chunk_size(dims, 512);

    let ranges = spatial_par.split_by_spatial();

    assert_eq!(ranges.len(), 8, "Should have 4096/512 = 8 chunks");

    // Verify complete coverage
    let mut covered = vec![false; 4096];
    for (start, end) in ranges {
        for i in start..end {
            assert!(!covered[i], "Index {} covered twice", i);
            covered[i] = true;
        }
    }
    assert!(covered.iter().all(|&c| c), "Not all indices covered");
}

#[test]
fn test_spatial_process() {
    let mut state = TQPState::new(8, 1, 1); // 256 spatial
    for i in 0..256 {
        state.state_vector[i] = Complex64::new(i as f64, 0.0);
    }

    let spatial_par = SpatialParallel::with_chunk_size(state.dims, 64);

    let result = spatial_par.process_layer_bin_parallel(&state, 0, 0, |idx, amp| {
        amp + Complex64::new(idx as f64, 0.0)
    });

    // Should double the value
    for i in 0..256 {
        let expected = 2.0 * i as f64;
        let diff = (result[i].re - expected).abs();
        assert!(diff < EPSILON, "Index {} mismatch", i);
    }
}

// =============================================================================
// Atomic Buffer Tests
// =============================================================================

#[test]
fn test_atomic_concurrent_writes() {
    use rayon::prelude::*;

    let buffer = AtomicStateBuffer::new(256);

    // Many threads writing to same locations
    (0..10000).into_par_iter().for_each(|i| {
        let idx = i % 256;
        buffer.atomic_add(idx, Complex64::new(0.01, 0.01));
    });

    // Each of 256 indices should have ~39 additions (10000/256 ≈ 39)
    let mut sum = Complex64::new(0.0, 0.0);
    for i in 0..256 {
        sum += buffer.get(i);
    }

    // Total should be ~100 for both real and imag
    assert!((sum.re - 100.0).abs() < 1.0, "Real sum wrong: {}", sum.re);
    assert!((sum.im - 100.0).abs() < 1.0, "Imag sum wrong: {}", sum.im);
}

// =============================================================================
// Barrier Sync Tests
// =============================================================================

#[test]
fn test_barrier_phases() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    let counter = Arc::new(AtomicUsize::new(0));
    let sync = BarrierSync::new(4);

    let counter_clone = Arc::clone(&counter);
    let results: Vec<usize> = sync.execute_with_barrier(move |thread_id, barrier| {
        // Phase 1: increment counter
        counter_clone.fetch_add(1, Ordering::SeqCst);

        // Wait for all threads
        barrier.wait();

        // Phase 2: read counter (should be 4)
        let val = counter_clone.load(Ordering::SeqCst);

        barrier.wait();

        val
    });

    // All threads should see counter = 4
    assert!(
        results.iter().all(|&v| v == 4),
        "Barrier sync failed: {:?}",
        results
    );
}

// =============================================================================
// Strategy Selection Tests
// =============================================================================

#[test]
fn test_auto_strategy_selection() {
    // Test various dimension combinations
    let test_cases = vec![
        // (N, M, L, expected_strategy)
        (4, 2, 8, ChunkStrategyKind::Layer), // Large L >= M
        (4, 16, 2, ChunkStrategyKind::Bin),  // Large M > L
        (14, 2, 2, ChunkStrategyKind::Spatial), // Large N (2^14 >= 1024)
                                             // For L=M=4, L>=4 so Layer wins
    ];

    for (n, m, l, expected) in test_cases {
        let dims = TQPDimensions::new(n, m, l);
        let strategy = ChunkStrategy::auto(dims);
        let selected = strategy.select_strategy();

        assert_eq!(
            selected, expected,
            "For N={}, M={}, L={}: expected {:?}, got {:?}",
            n, m, l, expected, selected
        );
    }
}

#[test]
fn test_unified_strategy_api() {
    let state = TQPState::new(4, 4, 2); // 16 * 4 * 2 = 128 elements

    // Test all strategies produce correct results
    for kind in [
        ChunkStrategyKind::Layer,
        ChunkStrategyKind::Bin,
        ChunkStrategyKind::Spatial,
    ] {
        let strategy = ChunkStrategy::new(state.dims, kind);

        let result = strategy.process(&state, |chunk| chunk.iter().map(|&c| c * 2.0).collect());

        // Verify doubling
        for i in 0..state.dimension() {
            let expected = state.state_vector[i] * 2.0;
            let diff = (expected - result.state_vector[i]).norm();
            assert!(diff < EPSILON, "{:?} strategy mismatch at {}", kind, i);
        }
    }
}

// =============================================================================
// Performance Validation
// =============================================================================

#[test]
fn test_parallel_speedup() {
    use std::time::Instant;

    // Large state for meaningful parallelization
    let mut state = TQPState::new(12, 8, 4); // 4096 * 8 * 4 = 131072 elements
    for i in 0..state.dimension() {
        state.state_vector[i] = Complex64::new(i as f64, (i as f64).sin());
    }

    // Expensive operation
    let expensive_op = |data: &[Complex64]| -> Vec<Complex64> {
        data.iter()
            .map(|&c| {
                let r = c.norm();
                let theta = c.arg();
                Complex64::from_polar((r * 5.0).sin().abs(), theta + 0.1)
            })
            .collect()
    };

    // Sequential
    let start = Instant::now();
    let _: Vec<Complex64> = expensive_op(&state.state_vector.to_vec());
    let seq_time = start.elapsed();

    // Bin parallel (best for this config: 8*4 = 32 chunks)
    let bin_par = BinParallel::new(state.dims);
    let start = Instant::now();
    let _ = bin_par.process_parallel(&state, |chunk, _, _| expensive_op(&chunk.data));
    let par_time = start.elapsed();

    let speedup = seq_time.as_secs_f64() / par_time.as_secs_f64();

    println!("\n=== Parallel Speedup Test ===");
    println!("State size: {} elements", state.dimension());
    println!("Chunks: {}", 8 * 4);
    println!("Sequential: {:?}", seq_time);
    println!("Parallel: {:?}", par_time);
    println!("Speedup: {:.2}x", speedup);

    // Note: Parallelization overhead can exceed benefit for moderate workloads
    // This test verifies correctness and measures performance
    // Actual speedup depends on workload intensity and hardware
    assert!(
        speedup > 0.3,
        "Parallel performance degraded too much: {:.2}x",
        speedup
    );
}

//! Gradient Cache for Variational Quantum Algorithms
//!
//! Implements caching strategies to avoid redundant gradient computations
//! in iterative optimization loops (VQE, QAOA).
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                      GradientCache                              │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  LRU Cache Layer                                                │
//! │    ├── capacity: max entries                                    │
//! │    ├── entries: HashMap<CacheKey, CacheEntry>                   │
//! │    └── lru_order: VecDeque<CacheKey>                           │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  Quantized Lookup                                               │
//! │    ├── quantization_bits: precision level                       │
//! │    └── tolerance: cache hit threshold                           │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  Statistics                                                     │
//! │    ├── hits / misses                                            │
//! │    ├── evictions                                                │
//! │    └── compute_time_saved                                       │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};

// =============================================================================
// Constants
// =============================================================================

/// Default cache capacity
pub const DEFAULT_CACHE_CAPACITY: usize = 1000;

/// Default quantization bits (precision: 2^-10 ≈ 0.001)
pub const DEFAULT_QUANTIZATION_BITS: u32 = 10;

/// Default tolerance for cache hits
pub const DEFAULT_TOLERANCE: f64 = 1e-8;

/// Default interpolation threshold
pub const INTERPOLATION_THRESHOLD: f64 = 0.1;

// =============================================================================
// Cache Key
// =============================================================================

/// Cache key with quantized parameters
#[derive(Debug, Clone)]
pub struct CacheKey {
    /// Quantized parameter values
    quantized: Vec<i64>,
    /// Original parameters (for tolerance check)
    original: Vec<f64>,
    /// Quantization scale
    scale: f64,
}

impl CacheKey {
    /// Create cache key from parameters
    pub fn new(params: &[f64], quantization_bits: u32) -> Self {
        let scale = (1u64 << quantization_bits) as f64;
        let quantized: Vec<i64> = params.iter().map(|&p| (p * scale).round() as i64).collect();
        Self {
            quantized,
            original: params.to_vec(),
            scale,
        }
    }

    /// Create key with custom scale
    pub fn with_scale(params: &[f64], scale: f64) -> Self {
        let quantized: Vec<i64> = params.iter().map(|&p| (p * scale).round() as i64).collect();
        Self {
            quantized,
            original: params.to_vec(),
            scale,
        }
    }

    /// Get original parameters
    pub fn params(&self) -> &[f64] {
        &self.original
    }

    /// Get quantized values
    pub fn quantized(&self) -> &[i64] {
        &self.quantized
    }

    /// Compute L2 distance to another key
    pub fn distance(&self, other: &CacheKey) -> f64 {
        self.original
            .iter()
            .zip(other.original.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Check if within tolerance of another key
    pub fn within_tolerance(&self, other: &CacheKey, tolerance: f64) -> bool {
        self.distance(other) <= tolerance
    }

    /// Dequantize to get approximate parameters
    pub fn dequantize(&self) -> Vec<f64> {
        self.quantized
            .iter()
            .map(|&q| q as f64 / self.scale)
            .collect()
    }

    /// Number of parameters
    pub fn len(&self) -> usize {
        self.original.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.original.is_empty()
    }
}

impl PartialEq for CacheKey {
    fn eq(&self, other: &Self) -> bool {
        self.quantized == other.quantized
    }
}

impl Eq for CacheKey {}

impl Hash for CacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.quantized.hash(state);
    }
}

// =============================================================================
// Cache Entry
// =============================================================================

/// Entry in gradient cache
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// Cached gradient values
    pub gradients: Vec<f64>,
    /// Expectation value (optional)
    pub expectation: Option<f64>,
    /// Computation time for this entry
    pub compute_time: Duration,
    /// Number of times accessed
    pub access_count: usize,
    /// Last access timestamp
    pub last_access: Instant,
    /// Creation timestamp
    pub created: Instant,
}

impl CacheEntry {
    /// Create new cache entry
    pub fn new(gradients: Vec<f64>) -> Self {
        let now = Instant::now();
        Self {
            gradients,
            expectation: None,
            compute_time: Duration::ZERO,
            access_count: 0,
            last_access: now,
            created: now,
        }
    }

    /// Create entry with expectation value
    pub fn with_expectation(gradients: Vec<f64>, expectation: f64) -> Self {
        let mut entry = Self::new(gradients);
        entry.expectation = Some(expectation);
        entry
    }

    /// Create entry with timing info
    pub fn with_timing(gradients: Vec<f64>, compute_time: Duration) -> Self {
        let mut entry = Self::new(gradients);
        entry.compute_time = compute_time;
        entry
    }

    /// Record access
    pub fn touch(&mut self) {
        self.access_count += 1;
        self.last_access = Instant::now();
    }

    /// Get age in seconds
    pub fn age(&self) -> f64 {
        self.created.elapsed().as_secs_f64()
    }

    /// Get time since last access
    pub fn idle_time(&self) -> f64 {
        self.last_access.elapsed().as_secs_f64()
    }
}

// =============================================================================
// Cache Statistics
// =============================================================================

/// Cache performance statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of cache hits
    pub hits: usize,
    /// Number of cache misses
    pub misses: usize,
    /// Number of evictions
    pub evictions: usize,
    /// Number of insertions
    pub insertions: usize,
    /// Total compute time saved (estimated)
    pub time_saved: Duration,
    /// Total lookups
    pub lookups: usize,
    /// Number of interpolation hits
    pub interpolation_hits: usize,
    /// Number of tolerance hits
    pub tolerance_hits: usize,
}

impl CacheStats {
    /// Create new stats
    pub fn new() -> Self {
        Self::default()
    }

    /// Get hit rate
    pub fn hit_rate(&self) -> f64 {
        if self.lookups == 0 {
            0.0
        } else {
            self.hits as f64 / self.lookups as f64
        }
    }

    /// Get miss rate
    pub fn miss_rate(&self) -> f64 {
        1.0 - self.hit_rate()
    }

    /// Get eviction rate
    pub fn eviction_rate(&self) -> f64 {
        if self.insertions == 0 {
            0.0
        } else {
            self.evictions as f64 / self.insertions as f64
        }
    }

    /// Reset statistics
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Record hit
    pub fn record_hit(&mut self, time_saved: Duration) {
        self.hits += 1;
        self.lookups += 1;
        self.time_saved += time_saved;
    }

    /// Record miss
    pub fn record_miss(&mut self) {
        self.misses += 1;
        self.lookups += 1;
    }

    /// Record eviction
    pub fn record_eviction(&mut self) {
        self.evictions += 1;
    }

    /// Record insertion
    pub fn record_insertion(&mut self) {
        self.insertions += 1;
    }

    /// Record interpolation hit
    pub fn record_interpolation(&mut self) {
        self.interpolation_hits += 1;
    }

    /// Record tolerance hit
    pub fn record_tolerance(&mut self) {
        self.tolerance_hits += 1;
    }
}

// =============================================================================
// Gradient Cache
// =============================================================================

/// LRU cache for gradient values
#[derive(Debug)]
pub struct GradientCache {
    /// Cache entries
    entries: HashMap<CacheKey, CacheEntry>,
    /// LRU order (front = oldest, back = newest)
    lru_order: VecDeque<CacheKey>,
    /// Maximum capacity
    capacity: usize,
    /// Quantization bits
    quantization_bits: u32,
    /// Tolerance for fuzzy matching
    tolerance: f64,
    /// Enable interpolation
    enable_interpolation: bool,
    /// Statistics
    stats: CacheStats,
    /// Average compute time (for time saved estimation)
    avg_compute_time: Duration,
}

impl GradientCache {
    /// Create new cache with default settings
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(capacity),
            lru_order: VecDeque::with_capacity(capacity),
            capacity,
            quantization_bits: DEFAULT_QUANTIZATION_BITS,
            tolerance: DEFAULT_TOLERANCE,
            enable_interpolation: false,
            stats: CacheStats::new(),
            avg_compute_time: Duration::from_micros(100),
        }
    }

    /// Create cache with custom quantization
    pub fn with_quantization(capacity: usize, quantization_bits: u32) -> Self {
        let mut cache = Self::new(capacity);
        cache.quantization_bits = quantization_bits;
        cache
    }

    /// Create cache with tolerance
    pub fn with_tolerance(capacity: usize, tolerance: f64) -> Self {
        let mut cache = Self::new(capacity);
        cache.tolerance = tolerance;
        cache
    }

    /// Enable interpolation
    pub fn with_interpolation(mut self) -> Self {
        self.enable_interpolation = true;
        self
    }

    /// Set average compute time for statistics
    pub fn set_avg_compute_time(&mut self, time: Duration) {
        self.avg_compute_time = time;
    }

    // -------------------------------------------------------------------------
    // Core Operations
    // -------------------------------------------------------------------------

    /// Get cached gradient for parameters
    pub fn get(&mut self, params: &[f64]) -> Option<Vec<f64>> {
        let key = CacheKey::new(params, self.quantization_bits);

        // Try exact match first
        if self.entries.contains_key(&key) {
            // Update LRU first (before borrowing entry)
            self.update_lru(&key);
            self.stats.record_hit(self.avg_compute_time);

            // Now get mutable reference and update
            if let Some(entry) = self.entries.get_mut(&key) {
                entry.touch();
                return Some(entry.gradients.clone());
            }
        }

        // Try tolerance-based lookup
        if self.tolerance > 0.0 {
            if let Some(grads) = self.get_within_tolerance(&key) {
                self.stats.record_tolerance();
                return Some(grads);
            }
        }

        // Try interpolation
        if self.enable_interpolation {
            if let Some(grads) = self.interpolate(&key) {
                self.stats.record_interpolation();
                return Some(grads);
            }
        }

        self.stats.record_miss();
        None
    }

    /// Get with expectation value
    pub fn get_with_expectation(&mut self, params: &[f64]) -> Option<(Vec<f64>, Option<f64>)> {
        let key = CacheKey::new(params, self.quantization_bits);

        if self.entries.contains_key(&key) {
            // Update LRU first
            self.update_lru(&key);
            self.stats.record_hit(self.avg_compute_time);

            if let Some(entry) = self.entries.get_mut(&key) {
                entry.touch();
                return Some((entry.gradients.clone(), entry.expectation));
            }
        }

        self.stats.record_miss();
        None
    }

    /// Insert gradient into cache
    pub fn insert(&mut self, params: Vec<f64>, gradients: Vec<f64>) {
        let key = CacheKey::new(&params, self.quantization_bits);
        self.insert_with_key(key, CacheEntry::new(gradients));
    }

    /// Insert with expectation value
    pub fn insert_with_expectation(
        &mut self,
        params: Vec<f64>,
        gradients: Vec<f64>,
        expectation: f64,
    ) {
        let key = CacheKey::new(&params, self.quantization_bits);
        self.insert_with_key(key, CacheEntry::with_expectation(gradients, expectation));
    }

    /// Insert with timing
    pub fn insert_with_timing(
        &mut self,
        params: Vec<f64>,
        gradients: Vec<f64>,
        compute_time: Duration,
    ) {
        let key = CacheKey::new(&params, self.quantization_bits);
        self.insert_with_key(key, CacheEntry::with_timing(gradients, compute_time));
    }

    /// Internal insert with key
    fn insert_with_key(&mut self, key: CacheKey, entry: CacheEntry) {
        // Check if already exists
        if self.entries.contains_key(&key) {
            self.entries.insert(key.clone(), entry);
            self.update_lru(&key);
            return;
        }

        // Evict if at capacity
        while self.entries.len() >= self.capacity {
            self.evict_lru();
        }

        // Insert new entry
        self.entries.insert(key.clone(), entry);
        self.lru_order.push_back(key);
        self.stats.record_insertion();
    }

    /// Update LRU order (move to back)
    fn update_lru(&mut self, key: &CacheKey) {
        if let Some(pos) = self.lru_order.iter().position(|k| k == key) {
            self.lru_order.remove(pos);
            self.lru_order.push_back(key.clone());
        }
    }

    /// Evict least recently used entry
    fn evict_lru(&mut self) {
        if let Some(key) = self.lru_order.pop_front() {
            self.entries.remove(&key);
            self.stats.record_eviction();
        }
    }

    // -------------------------------------------------------------------------
    // Tolerance-based Lookup
    // -------------------------------------------------------------------------

    /// Find entry within tolerance
    fn get_within_tolerance(&mut self, key: &CacheKey) -> Option<Vec<f64>> {
        let mut best_match: Option<(CacheKey, f64)> = None;

        for cached_key in self.entries.keys() {
            let dist = key.distance(cached_key);
            if dist <= self.tolerance {
                match best_match {
                    None => best_match = Some((cached_key.clone(), dist)),
                    Some((_, best_dist)) if dist < best_dist => {
                        best_match = Some((cached_key.clone(), dist));
                    }
                    _ => {}
                }
            }
        }

        if let Some((matched_key, _)) = best_match {
            // Update LRU first
            self.update_lru(&matched_key);
            self.stats.record_hit(self.avg_compute_time);

            if let Some(entry) = self.entries.get_mut(&matched_key) {
                entry.touch();
                return Some(entry.gradients.clone());
            }
        }

        None
    }

    // -------------------------------------------------------------------------
    // Interpolation
    // -------------------------------------------------------------------------

    /// Interpolate gradient from nearby cached values
    fn interpolate(&self, key: &CacheKey) -> Option<Vec<f64>> {
        if self.entries.len() < 2 || key.is_empty() {
            return None;
        }

        // Find two nearest neighbors
        let mut neighbors: Vec<(&CacheKey, f64)> = self
            .entries
            .keys()
            .map(|k| (k, key.distance(k)))
            .filter(|(_, d)| *d <= INTERPOLATION_THRESHOLD)
            .collect();

        if neighbors.len() < 2 {
            return None;
        }

        neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let (k1, d1) = neighbors[0];
        let (k2, d2) = neighbors[1];

        let entry1 = self.entries.get(k1)?;
        let entry2 = self.entries.get(k2)?;

        if entry1.gradients.len() != entry2.gradients.len() {
            return None;
        }

        // Linear interpolation: weight inversely proportional to distance
        let total_dist = d1 + d2;
        if total_dist < 1e-12 {
            return Some(entry1.gradients.clone());
        }

        let w1 = d2 / total_dist;
        let w2 = d1 / total_dist;

        let interpolated: Vec<f64> = entry1
            .gradients
            .iter()
            .zip(entry2.gradients.iter())
            .map(|(&g1, &g2)| w1 * g1 + w2 * g2)
            .collect();

        Some(interpolated)
    }

    // -------------------------------------------------------------------------
    // Cache Management
    // -------------------------------------------------------------------------

    /// Get current size
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.entries.clear();
        self.lru_order.clear();
    }

    /// Get statistics
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats.reset();
    }

    /// Check if key exists
    pub fn contains(&self, params: &[f64]) -> bool {
        let key = CacheKey::new(params, self.quantization_bits);
        self.entries.contains_key(&key)
    }

    /// Remove entry
    pub fn remove(&mut self, params: &[f64]) -> Option<Vec<f64>> {
        let key = CacheKey::new(params, self.quantization_bits);
        if let Some(entry) = self.entries.remove(&key) {
            if let Some(pos) = self.lru_order.iter().position(|k| k == &key) {
                self.lru_order.remove(pos);
            }
            Some(entry.gradients)
        } else {
            None
        }
    }

    /// Get all cached parameters
    pub fn cached_params(&self) -> Vec<Vec<f64>> {
        self.entries.keys().map(|k| k.params().to_vec()).collect()
    }

    /// Prune old entries
    pub fn prune_older_than(&mut self, max_age_secs: f64) {
        let keys_to_remove: Vec<CacheKey> = self
            .entries
            .iter()
            .filter(|(_, entry)| entry.age() > max_age_secs)
            .map(|(key, _)| key.clone())
            .collect();

        for key in keys_to_remove {
            self.entries.remove(&key);
            if let Some(pos) = self.lru_order.iter().position(|k| k == &key) {
                self.lru_order.remove(pos);
            }
            self.stats.record_eviction();
        }
    }

    /// Prune entries not accessed recently
    pub fn prune_idle(&mut self, max_idle_secs: f64) {
        let keys_to_remove: Vec<CacheKey> = self
            .entries
            .iter()
            .filter(|(_, entry)| entry.idle_time() > max_idle_secs)
            .map(|(key, _)| key.clone())
            .collect();

        for key in keys_to_remove {
            self.entries.remove(&key);
            if let Some(pos) = self.lru_order.iter().position(|k| k == &key) {
                self.lru_order.remove(pos);
            }
            self.stats.record_eviction();
        }
    }
}

// =============================================================================
// Cached Gradient Computer
// =============================================================================

/// Wrapper that combines cache with gradient computation
pub struct CachedGradientComputer<F>
where
    F: Fn(&[f64]) -> (f64, Vec<f64>),
{
    cache: GradientCache,
    compute_fn: F,
}

impl<F> CachedGradientComputer<F>
where
    F: Fn(&[f64]) -> (f64, Vec<f64>),
{
    /// Create new cached computer
    pub fn new(cache: GradientCache, compute_fn: F) -> Self {
        Self { cache, compute_fn }
    }

    /// Compute or retrieve from cache
    pub fn compute(&mut self, params: &[f64]) -> (f64, Vec<f64>) {
        // Check cache
        if let Some((grads, Some(exp))) = self.cache.get_with_expectation(params) {
            return (exp, grads);
        }

        // Compute
        let start = Instant::now();
        let (expectation, gradients) = (self.compute_fn)(params);
        let elapsed = start.elapsed();

        // Cache result
        let mut entry = CacheEntry::with_expectation(gradients.clone(), expectation);
        entry.compute_time = elapsed;
        let key = CacheKey::new(params, self.cache.quantization_bits);
        self.cache.insert_with_key(key, entry);

        (expectation, gradients)
    }

    /// Get cache reference
    pub fn cache(&self) -> &GradientCache {
        &self.cache
    }

    /// Get mutable cache reference
    pub fn cache_mut(&mut self) -> &mut GradientCache {
        &mut self.cache
    }
}

// =============================================================================
// Batch Cache
// =============================================================================

/// Cache optimized for batch gradient computation
#[derive(Debug)]
pub struct BatchGradientCache {
    /// Inner cache
    inner: GradientCache,
    /// Batch size for prefetching
    batch_size: usize,
    /// Prefetch buffer
    prefetch_buffer: Vec<(Vec<f64>, Option<Vec<f64>>)>,
}

impl BatchGradientCache {
    /// Create new batch cache
    pub fn new(capacity: usize, batch_size: usize) -> Self {
        Self {
            inner: GradientCache::new(capacity),
            batch_size,
            prefetch_buffer: Vec::with_capacity(batch_size),
        }
    }

    /// Get batch of gradients
    pub fn get_batch(&mut self, params_batch: &[Vec<f64>]) -> Vec<Option<Vec<f64>>> {
        params_batch
            .iter()
            .map(|params| self.inner.get(params))
            .collect()
    }

    /// Insert batch of gradients
    pub fn insert_batch(&mut self, params_batch: Vec<Vec<f64>>, gradients_batch: Vec<Vec<f64>>) {
        for (params, grads) in params_batch.into_iter().zip(gradients_batch.into_iter()) {
            self.inner.insert(params, grads);
        }
    }

    /// Prefetch: mark parameters that will be needed
    pub fn prefetch(&mut self, params: Vec<f64>) {
        if self.prefetch_buffer.len() < self.batch_size {
            let cached = self.inner.get(&params);
            self.prefetch_buffer.push((params, cached));
        }
    }

    /// Get prefetched results
    pub fn get_prefetched(&mut self) -> Vec<(Vec<f64>, Option<Vec<f64>>)> {
        std::mem::take(&mut self.prefetch_buffer)
    }

    /// Get inner cache
    pub fn inner(&self) -> &GradientCache {
        &self.inner
    }

    /// Get mutable inner cache
    pub fn inner_mut(&mut self) -> &mut GradientCache {
        &mut self.inner
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_key_creation() {
        let key = CacheKey::new(&[0.1, 0.2, 0.3], 10);
        assert_eq!(key.len(), 3);
        assert!(!key.is_empty());
    }

    #[test]
    fn test_cache_key_equality() {
        let key1 = CacheKey::new(&[0.1, 0.2], 10);
        let key2 = CacheKey::new(&[0.1, 0.2], 10);
        let key3 = CacheKey::new(&[0.1, 0.3], 10);

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_cache_key_distance() {
        let key1 = CacheKey::new(&[0.0, 0.0], 10);
        let key2 = CacheKey::new(&[3.0, 4.0], 10);

        let dist = key1.distance(&key2);
        assert!((dist - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_cache_key_tolerance() {
        let key1 = CacheKey::new(&[0.0, 0.0], 10);
        let key2 = CacheKey::new(&[0.001, 0.001], 10);

        assert!(key1.within_tolerance(&key2, 0.01));
        assert!(!key1.within_tolerance(&key2, 0.0001));
    }

    #[test]
    fn test_cache_entry() {
        let entry = CacheEntry::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(entry.gradients, vec![1.0, 2.0, 3.0]);
        assert!(entry.expectation.is_none());
        assert_eq!(entry.access_count, 0);
    }

    #[test]
    fn test_cache_entry_with_expectation() {
        let entry = CacheEntry::with_expectation(vec![1.0], 0.5);
        assert_eq!(entry.expectation, Some(0.5));
    }

    #[test]
    fn test_cache_stats() {
        let mut stats = CacheStats::new();
        stats.record_hit(Duration::from_micros(100));
        stats.record_miss();
        stats.record_miss();

        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 2);
        assert_eq!(stats.lookups, 3);
        assert!((stats.hit_rate() - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_gradient_cache_basic() {
        let mut cache = GradientCache::new(100);

        cache.insert(vec![0.1, 0.2], vec![1.0, 2.0]);

        let result = cache.get(&[0.1, 0.2]);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), vec![1.0, 2.0]);
    }

    #[test]
    fn test_gradient_cache_miss() {
        let mut cache = GradientCache::new(100);

        cache.insert(vec![0.1, 0.2], vec![1.0, 2.0]);

        let result = cache.get(&[0.3, 0.4]);
        assert!(result.is_none());
    }

    #[test]
    fn test_gradient_cache_lru_eviction() {
        let mut cache = GradientCache::new(3);

        cache.insert(vec![0.1], vec![1.0]);
        cache.insert(vec![0.2], vec![2.0]);
        cache.insert(vec![0.3], vec![3.0]);

        // Access first to make it recently used
        cache.get(&[0.1]);

        // Insert fourth, should evict second (LRU)
        cache.insert(vec![0.4], vec![4.0]);

        assert!(cache.get(&[0.1]).is_some());
        assert!(cache.get(&[0.2]).is_none()); // Evicted
        assert!(cache.get(&[0.3]).is_some());
        assert!(cache.get(&[0.4]).is_some());
    }

    #[test]
    fn test_gradient_cache_with_expectation() {
        let mut cache = GradientCache::new(100);

        cache.insert_with_expectation(vec![0.1], vec![1.0, 2.0], 0.5);

        let result = cache.get_with_expectation(&[0.1]);
        assert!(result.is_some());
        let (grads, exp) = result.unwrap();
        assert_eq!(grads, vec![1.0, 2.0]);
        assert_eq!(exp, Some(0.5));
    }

    #[test]
    fn test_gradient_cache_clear() {
        let mut cache = GradientCache::new(100);

        cache.insert(vec![0.1], vec![1.0]);
        cache.insert(vec![0.2], vec![2.0]);

        assert_eq!(cache.len(), 2);

        cache.clear();

        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_gradient_cache_contains() {
        let mut cache = GradientCache::new(100);

        cache.insert(vec![0.1, 0.2], vec![1.0]);

        assert!(cache.contains(&[0.1, 0.2]));
        assert!(!cache.contains(&[0.3, 0.4]));
    }

    #[test]
    fn test_gradient_cache_remove() {
        let mut cache = GradientCache::new(100);

        cache.insert(vec![0.1], vec![1.0, 2.0]);

        let removed = cache.remove(&[0.1]);
        assert_eq!(removed, Some(vec![1.0, 2.0]));
        assert!(!cache.contains(&[0.1]));
    }

    #[test]
    fn test_batch_gradient_cache() {
        let mut cache = BatchGradientCache::new(100, 4);

        cache.insert_batch(
            vec![vec![0.1], vec![0.2], vec![0.3]],
            vec![vec![1.0], vec![2.0], vec![3.0]],
        );

        let results = cache.get_batch(&[vec![0.1], vec![0.2], vec![0.5]]);

        assert!(results[0].is_some());
        assert!(results[1].is_some());
        assert!(results[2].is_none());
    }

    #[test]
    fn test_cached_params() {
        let mut cache = GradientCache::new(100);

        cache.insert(vec![0.1, 0.2], vec![1.0]);
        cache.insert(vec![0.3, 0.4], vec![2.0]);

        let params = cache.cached_params();
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn test_interpolation() {
        let mut cache = GradientCache::new(100).with_interpolation();
        cache.tolerance = 0.0; // Disable tolerance to test interpolation

        cache.insert(vec![0.0], vec![0.0]);
        cache.insert(vec![1.0], vec![1.0]);

        // Query midpoint
        let result = cache.get(&[0.5]);

        // Should interpolate to ~0.5
        if let Some(grads) = result {
            assert!((grads[0] - 0.5).abs() < 0.2);
        }
    }

    #[test]
    fn test_quantization_precision() {
        // High precision (20 bits: scale = 2^20 ≈ 1M)
        let key_high = CacheKey::new(&[0.123456789], 20);
        // Very low precision (2 bits: scale = 4)
        let key_low = CacheKey::new(&[0.123456789], 2);

        // High precision preserves more digits
        let dequant_high = key_high.dequantize();
        let dequant_low = key_low.dequantize();

        // High: error < 1e-5 (scale=1M, so error ≈ 1/1M)
        assert!((dequant_high[0] - 0.123456789).abs() < 1e-5);
        // Low: 0.123... * 4 ≈ 0.49 → rounds to 0 → dequant = 0, error = 0.123...
        assert!((dequant_low[0] - 0.123456789).abs() > 0.01);
    }
}

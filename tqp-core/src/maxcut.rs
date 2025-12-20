//! MaxCut Problem Definition
//!
//! Sprint 3 Week 6 Day 1: Graph-based combinatorial optimization
//!
//! MaxCut is a fundamental NP-hard problem: given a graph G=(V,E),
//! find a partition of vertices into two sets that maximizes the
//! number of edges between the sets.
//!
//! # Theory
//!
//! For a graph with adjacency matrix W, the MaxCut objective is:
//! C(z) = (1/2) Σ_{ij} W_{ij} (1 - z_i z_j)
//!
//! where z_i ∈ {-1, +1} indicates which partition vertex i belongs to.
//!
//! In quantum form (z_i → Z_i Pauli operator):
//! C = (1/2) Σ_{ij} W_{ij} (I - Z_i Z_j)
//!
//! # Example
//!
//! ```ignore
//! use tqp_core::maxcut::{Graph, MaxCutProblem};
//!
//! // Create a triangle graph
//! let graph = Graph::triangle();
//! let problem = MaxCutProblem::new(graph);
//!
//! println!("Optimal cut: {}", problem.optimal_cut_value());
//! ```

use std::collections::HashSet;
use crate::autodiff::{Hamiltonian, PauliObservable};

// =============================================================================
// Graph Representation
// =============================================================================

/// Edge in a graph
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Edge {
    /// First vertex
    pub u: usize,
    /// Second vertex
    pub v: usize,
    /// Edge weight (default 1.0)
    pub weight: f64,
}

impl Edge {
    /// Create unweighted edge
    pub fn new(u: usize, v: usize) -> Self {
        Self { u, v, weight: 1.0 }
    }

    /// Create weighted edge
    pub fn weighted(u: usize, v: usize, weight: f64) -> Self {
        Self { u, v, weight }
    }
}

/// Undirected graph for MaxCut
#[derive(Debug, Clone)]
pub struct Graph {
    /// Number of vertices
    n_vertices: usize,
    /// List of edges
    edges: Vec<Edge>,
    /// Adjacency matrix (optional, computed on demand)
    adjacency: Option<Vec<Vec<f64>>>,
}

impl Graph {
    /// Create empty graph with n vertices
    pub fn new(n_vertices: usize) -> Self {
        Self {
            n_vertices,
            edges: Vec::new(),
            adjacency: None,
        }
    }

    /// Create graph from edge list
    pub fn from_edges(n_vertices: usize, edges: Vec<Edge>) -> Self {
        Self {
            n_vertices,
            edges,
            adjacency: None,
        }
    }

    /// Add an unweighted edge
    pub fn add_edge(&mut self, u: usize, v: usize) {
        self.edges.push(Edge::new(u, v));
        self.adjacency = None; // Invalidate cache
    }

    /// Add a weighted edge
    pub fn add_weighted_edge(&mut self, u: usize, v: usize, weight: f64) {
        self.edges.push(Edge::weighted(u, v, weight));
        self.adjacency = None;
    }

    /// Get number of vertices
    pub fn n_vertices(&self) -> usize {
        self.n_vertices
    }

    /// Get number of edges
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Get edges
    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    /// Get total weight of all edges
    pub fn total_weight(&self) -> f64 {
        self.edges.iter().map(|e| e.weight).sum()
    }

    /// Compute adjacency matrix
    pub fn adjacency_matrix(&mut self) -> &Vec<Vec<f64>> {
        if self.adjacency.is_none() {
            let mut adj = vec![vec![0.0; self.n_vertices]; self.n_vertices];
            for edge in &self.edges {
                adj[edge.u][edge.v] = edge.weight;
                adj[edge.v][edge.u] = edge.weight;
            }
            self.adjacency = Some(adj);
        }
        self.adjacency.as_ref().unwrap()
    }

    /// Get degree of a vertex
    pub fn degree(&self, v: usize) -> usize {
        self.edges.iter()
            .filter(|e| e.u == v || e.v == v)
            .count()
    }

    /// Check if graph is connected (simple BFS)
    pub fn is_connected(&self) -> bool {
        if self.n_vertices == 0 {
            return true;
        }

        let mut visited = vec![false; self.n_vertices];
        let mut queue = vec![0usize];
        visited[0] = true;
        let mut count = 1;

        while let Some(v) = queue.pop() {
            for edge in &self.edges {
                let neighbor = if edge.u == v {
                    Some(edge.v)
                } else if edge.v == v {
                    Some(edge.u)
                } else {
                    None
                };

                if let Some(n) = neighbor {
                    if !visited[n] {
                        visited[n] = true;
                        queue.push(n);
                        count += 1;
                    }
                }
            }
        }

        count == self.n_vertices
    }

    // =========================================================================
    // Standard Graph Constructors
    // =========================================================================

    /// Triangle graph (3 vertices, 3 edges)
    pub fn triangle() -> Self {
        Self::from_edges(3, vec![
            Edge::new(0, 1),
            Edge::new(1, 2),
            Edge::new(2, 0),
        ])
    }

    /// Square graph (4 vertices, 4 edges)
    pub fn square() -> Self {
        Self::from_edges(4, vec![
            Edge::new(0, 1),
            Edge::new(1, 2),
            Edge::new(2, 3),
            Edge::new(3, 0),
        ])
    }

    /// Complete graph K_n
    pub fn complete(n: usize) -> Self {
        let mut edges = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                edges.push(Edge::new(i, j));
            }
        }
        Self::from_edges(n, edges)
    }

    /// Path graph P_n (n vertices, n-1 edges)
    pub fn path(n: usize) -> Self {
        let edges: Vec<Edge> = (0..n - 1)
            .map(|i| Edge::new(i, i + 1))
            .collect();
        Self::from_edges(n, edges)
    }

    /// Cycle graph C_n (n vertices, n edges)
    pub fn cycle(n: usize) -> Self {
        let mut edges: Vec<Edge> = (0..n - 1)
            .map(|i| Edge::new(i, i + 1))
            .collect();
        edges.push(Edge::new(n - 1, 0));
        Self::from_edges(n, edges)
    }

    /// Star graph S_n (center connected to n-1 leaves)
    pub fn star(n: usize) -> Self {
        let edges: Vec<Edge> = (1..n)
            .map(|i| Edge::new(0, i))
            .collect();
        Self::from_edges(n, edges)
    }

    /// Bipartite complete graph K_{m,n}
    pub fn bipartite_complete(m: usize, n: usize) -> Self {
        let mut edges = Vec::new();
        for i in 0..m {
            for j in 0..n {
                edges.push(Edge::new(i, m + j));
            }
        }
        Self::from_edges(m + n, edges)
    }

    /// Random graph (Erdős–Rényi G(n, p))
    pub fn random(n: usize, p: f64, seed: u64) -> Self {
        let mut edges = Vec::new();
        let mut rng_state = seed;

        for i in 0..n {
            for j in (i + 1)..n {
                rng_state = rng_state.wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let r = (rng_state >> 33) as f64 / (1u64 << 31) as f64;

                if r < p {
                    edges.push(Edge::new(i, j));
                }
            }
        }

        Self::from_edges(n, edges)
    }

    /// Random regular graph (approximately d-regular)
    pub fn random_regular(n: usize, d: usize, seed: u64) -> Self {
        // Simple approximation: add edges randomly until average degree ≈ d
        let target_edges = n * d / 2;
        let mut edges = Vec::new();
        let mut rng_state = seed;

        let mut attempts = 0;
        while edges.len() < target_edges && attempts < target_edges * 10 {
            rng_state = rng_state.wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let i = ((rng_state >> 33) as usize) % n;

            rng_state = rng_state.wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let j = ((rng_state >> 33) as usize) % n;

            if i != j && !edges.iter().any(|e: &Edge| 
                (e.u == i && e.v == j) || (e.u == j && e.v == i)
            ) {
                edges.push(Edge::new(i.min(j), i.max(j)));
            }
            attempts += 1;
        }

        Self::from_edges(n, edges)
    }
}

// =============================================================================
// MaxCut Problem
// =============================================================================

/// MaxCut problem instance
#[derive(Debug, Clone)]
pub struct MaxCutProblem {
    /// Underlying graph
    graph: Graph,
    /// Cost Hamiltonian for QAOA
    cost_hamiltonian: Hamiltonian,
    /// Optimal solution (computed on demand)
    optimal: Option<(Vec<bool>, f64)>,
}

impl MaxCutProblem {
    /// Create MaxCut problem from graph
    pub fn new(graph: Graph) -> Self {
        let cost_hamiltonian = Self::build_cost_hamiltonian(&graph);

        Self {
            graph,
            cost_hamiltonian,
            optimal: None,
        }
    }

    /// Build cost Hamiltonian: C = (1/2) Σ_{ij} W_{ij} (I - Z_i Z_j)
    fn build_cost_hamiltonian(graph: &Graph) -> Hamiltonian {
        let mut hamiltonian = Hamiltonian::new();

        for edge in graph.edges() {
            // (1/2) W_{ij} (I - Z_i Z_j)
            // = (1/2) W_{ij} I - (1/2) W_{ij} Z_i Z_j
            // We encode the constant term separately
            // For the Hamiltonian, we use -0.5 * W * ZZ
            let coeff = -0.5 * edge.weight;
            hamiltonian.add_term_weighted(
                PauliObservable::zz(edge.u, edge.v),
                coeff,
            );
        }

        hamiltonian
    }

    /// Get the graph
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Get number of qubits needed
    pub fn n_qubits(&self) -> usize {
        self.graph.n_vertices()
    }

    /// Get the cost Hamiltonian
    pub fn cost_hamiltonian(&self) -> &Hamiltonian {
        &self.cost_hamiltonian
    }

    /// Get constant offset in the Hamiltonian
    /// C = offset - Σ H_terms
    pub fn constant_offset(&self) -> f64 {
        0.5 * self.graph.total_weight()
    }

    /// Evaluate cut value for a given bitstring
    /// bitstring[i] = true means vertex i is in set S
    pub fn evaluate_cut(&self, bitstring: &[bool]) -> f64 {
        let mut cut_value = 0.0;

        for edge in self.graph.edges() {
            // Edge is cut if vertices are in different sets
            if bitstring[edge.u] != bitstring[edge.v] {
                cut_value += edge.weight;
            }
        }

        cut_value
    }

    /// Evaluate cut from integer assignment (0/1)
    pub fn evaluate_cut_int(&self, assignment: &[u8]) -> f64 {
        let bitstring: Vec<bool> = assignment.iter().map(|&b| b != 0).collect();
        self.evaluate_cut(&bitstring)
    }

    /// Compute optimal cut by brute force (exponential, for small graphs only)
    pub fn compute_optimal(&mut self) -> (Vec<bool>, f64) {
        if let Some(ref opt) = self.optimal {
            return opt.clone();
        }

        let n = self.graph.n_vertices();
        assert!(n <= 20, "Brute force MaxCut only for n <= 20");

        let mut best_cut = 0.0;
        let mut best_assignment = vec![false; n];

        // Try all 2^n assignments
        for bits in 0..(1u32 << n) {
            let assignment: Vec<bool> = (0..n)
                .map(|i| (bits >> i) & 1 == 1)
                .collect();

            let cut = self.evaluate_cut(&assignment);
            if cut > best_cut {
                best_cut = cut;
                best_assignment = assignment;
            }
        }

        self.optimal = Some((best_assignment.clone(), best_cut));
        (best_assignment, best_cut)
    }

    /// Get optimal cut value (computes if not cached)
    pub fn optimal_cut_value(&mut self) -> f64 {
        self.compute_optimal().1
    }

    /// Get approximation ratio for a given cut
    pub fn approximation_ratio(&mut self, cut_value: f64) -> f64 {
        let optimal = self.optimal_cut_value();
        if optimal > 0.0 {
            cut_value / optimal
        } else {
            1.0
        }
    }

    /// Sample a random cut
    pub fn random_cut(&self, seed: u64) -> (Vec<bool>, f64) {
        let n = self.graph.n_vertices();
        let mut assignment = Vec::with_capacity(n);
        let mut rng_state = seed;

        for _ in 0..n {
            rng_state = rng_state.wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            assignment.push((rng_state >> 63) == 1);
        }

        let cut = self.evaluate_cut(&assignment);
        (assignment, cut)
    }

    /// Greedy cut approximation
    pub fn greedy_cut(&self) -> (Vec<bool>, f64) {
        let n = self.graph.n_vertices();
        let mut assignment = vec![false; n];
        let mut set_s_weight = vec![0.0; n]; // Weight to S
        let mut set_t_weight = vec![0.0; n]; // Weight to T (complement)

        // Assign vertices one by one to maximize cut
        for v in 0..n {
            // Count weight of edges to vertices already in S vs T
            for edge in self.graph.edges() {
                let other = if edge.u == v {
                    Some(edge.v)
                } else if edge.v == v {
                    Some(edge.u)
                } else {
                    None
                };

                if let Some(u) = other {
                    if u < v {
                        if assignment[u] {
                            set_s_weight[v] += edge.weight;
                        } else {
                            set_t_weight[v] += edge.weight;
                        }
                    }
                }
            }

            // Put v in the set that maximizes cut
            assignment[v] = set_t_weight[v] > set_s_weight[v];
        }

        let cut = self.evaluate_cut(&assignment);
        (assignment, cut)
    }

    /// Print problem summary
    pub fn print_summary(&mut self) {
        println!("=== MaxCut Problem ===");
        println!("Vertices: {}", self.graph.n_vertices());
        println!("Edges: {}", self.graph.n_edges());
        println!("Total weight: {:.2}", self.graph.total_weight());

        if self.graph.n_vertices() <= 20 {
            let (_, opt) = self.compute_optimal();
            println!("Optimal cut: {:.2}", opt);
        }

        let (_, greedy) = self.greedy_cut();
        println!("Greedy cut: {:.2}", greedy);
    }
}

// =============================================================================
// Common Problem Instances
// =============================================================================

/// Create standard MaxCut test instances
pub mod instances {
    use super::*;

    /// Triangle graph - optimal cut = 2
    pub fn triangle() -> MaxCutProblem {
        MaxCutProblem::new(Graph::triangle())
    }

    /// Square graph - optimal cut = 4
    pub fn square() -> MaxCutProblem {
        MaxCutProblem::new(Graph::square())
    }

    /// 4-vertex complete graph K4 - optimal cut = 4
    pub fn k4() -> MaxCutProblem {
        MaxCutProblem::new(Graph::complete(4))
    }

    /// 5-vertex complete graph K5 - optimal cut = 6
    pub fn k5() -> MaxCutProblem {
        MaxCutProblem::new(Graph::complete(5))
    }

    /// 3-regular graph on 6 vertices
    pub fn regular_6_3() -> MaxCutProblem {
        // Petersen-like structure
        let edges = vec![
            Edge::new(0, 1), Edge::new(1, 2), Edge::new(2, 0),
            Edge::new(3, 4), Edge::new(4, 5), Edge::new(5, 3),
            Edge::new(0, 3), Edge::new(1, 4), Edge::new(2, 5),
        ];
        MaxCutProblem::new(Graph::from_edges(6, edges))
    }

    /// Random graph for testing
    pub fn random_sparse(n: usize, seed: u64) -> MaxCutProblem {
        MaxCutProblem::new(Graph::random(n, 0.3, seed))
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_creation() {
        let e = Edge::new(0, 1);
        assert_eq!(e.u, 0);
        assert_eq!(e.v, 1);
        assert_eq!(e.weight, 1.0);

        let ew = Edge::weighted(0, 1, 2.5);
        assert_eq!(ew.weight, 2.5);
    }

    #[test]
    fn test_graph_creation() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1);
        g.add_edge(1, 2);

        assert_eq!(g.n_vertices(), 4);
        assert_eq!(g.n_edges(), 2);
    }

    #[test]
    fn test_graph_triangle() {
        let g = Graph::triangle();
        assert_eq!(g.n_vertices(), 3);
        assert_eq!(g.n_edges(), 3);
    }

    #[test]
    fn test_graph_complete() {
        let g = Graph::complete(5);
        assert_eq!(g.n_vertices(), 5);
        assert_eq!(g.n_edges(), 10); // C(5,2) = 10
    }

    #[test]
    fn test_graph_cycle() {
        let g = Graph::cycle(6);
        assert_eq!(g.n_vertices(), 6);
        assert_eq!(g.n_edges(), 6);
    }

    #[test]
    fn test_graph_connected() {
        assert!(Graph::triangle().is_connected());
        assert!(Graph::complete(5).is_connected());
        assert!(Graph::path(4).is_connected());

        // Disconnected graph
        let g = Graph::from_edges(4, vec![Edge::new(0, 1)]);
        assert!(!g.is_connected());
    }

    #[test]
    fn test_adjacency_matrix() {
        let mut g = Graph::triangle();
        let adj = g.adjacency_matrix();

        assert_eq!(adj.len(), 3);
        assert_eq!(adj[0][1], 1.0);
        assert_eq!(adj[1][0], 1.0);
        assert_eq!(adj[0][0], 0.0);
    }

    #[test]
    fn test_maxcut_triangle() {
        let mut problem = instances::triangle();

        // Triangle: optimal cut is 2 (any edge pair)
        let opt = problem.optimal_cut_value();
        assert_eq!(opt, 2.0);
    }

    #[test]
    fn test_maxcut_square() {
        let mut problem = instances::square();

        // Square: optimal cut is 4 (diagonal partition)
        let opt = problem.optimal_cut_value();
        assert_eq!(opt, 4.0);
    }

    #[test]
    fn test_maxcut_k4() {
        let mut problem = instances::k4();

        // K4: optimal cut is 4 (2 vs 2 partition)
        let opt = problem.optimal_cut_value();
        assert_eq!(opt, 4.0);
    }

    #[test]
    fn test_evaluate_cut() {
        let problem = instances::triangle();

        // All in same set: cut = 0
        assert_eq!(problem.evaluate_cut(&[false, false, false]), 0.0);
        assert_eq!(problem.evaluate_cut(&[true, true, true]), 0.0);

        // One vertex separate: cut = 2
        assert_eq!(problem.evaluate_cut(&[true, false, false]), 2.0);
        assert_eq!(problem.evaluate_cut(&[false, true, false]), 2.0);
    }

    #[test]
    fn test_cost_hamiltonian() {
        let problem = instances::triangle();
        let ham = problem.cost_hamiltonian();

        assert_eq!(ham.terms().len(), 3); // 3 ZZ terms
    }

    #[test]
    fn test_constant_offset() {
        let problem = instances::triangle();
        let offset = problem.constant_offset();

        // Triangle: 3 edges, total weight 3, offset = 1.5
        assert_eq!(offset, 1.5);
    }

    #[test]
    fn test_greedy_cut() {
        let problem = instances::square();
        let (_, cut) = problem.greedy_cut();

        // Greedy should find a reasonable cut
        assert!(cut >= 2.0);
    }

    #[test]
    fn test_random_cut() {
        let problem = instances::triangle();
        let (assignment, cut) = problem.random_cut(42);

        assert_eq!(assignment.len(), 3);
        assert!(cut >= 0.0);
        assert!(cut <= 3.0);
    }

    #[test]
    fn test_approximation_ratio() {
        let mut problem = instances::triangle();
        let ratio = problem.approximation_ratio(2.0);

        assert_eq!(ratio, 1.0); // Optimal
    }

    #[test]
    fn test_weighted_edges() {
        let mut g = Graph::new(3);
        g.add_weighted_edge(0, 1, 2.0);
        g.add_weighted_edge(1, 2, 3.0);
        g.add_weighted_edge(2, 0, 1.0);

        let mut problem = MaxCutProblem::new(g);

        // Optimal: cut edges with weights 2+3=5 or 2+1=3 or 3+1=4
        let opt = problem.optimal_cut_value();
        assert_eq!(opt, 5.0);
    }

    #[test]
    fn test_random_graph() {
        let g = Graph::random(10, 0.5, 12345);

        assert_eq!(g.n_vertices(), 10);
        assert!(g.n_edges() > 0);
    }

    #[test]
    fn test_star_graph() {
        let g = Graph::star(5);

        assert_eq!(g.n_vertices(), 5);
        assert_eq!(g.n_edges(), 4);
        assert_eq!(g.degree(0), 4); // Center
    }

    #[test]
    fn test_bipartite_complete() {
        let g = Graph::bipartite_complete(3, 3);

        assert_eq!(g.n_vertices(), 6);
        assert_eq!(g.n_edges(), 9);
    }
}

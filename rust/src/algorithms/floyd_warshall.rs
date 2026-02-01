//! Floyd-Warshall all-pairs shortest paths algorithm.
//!
//! Time complexity: O(n³)
//! Space complexity: O(n²)

use crate::callback::ProgressCallback;
use crate::types::{AlgorithmResult, Progress, Status};

/// Edge representation: (from, to, weight)
pub type Edge = (usize, usize, f64);

/// Result of Floyd-Warshall: distance matrix and predecessor matrix.
#[derive(Debug, Clone)]
pub struct FloydWarshallResult {
    /// Distance matrix: dist[i][j] = shortest distance from i to j
    pub distances: Vec<Vec<f64>>,
    /// Predecessor matrix: pred[i][j] = predecessor of j on shortest path from i
    /// None if no path exists
    pub predecessors: Vec<Vec<Option<usize>>>,
    /// Whether a negative cycle was detected
    pub has_negative_cycle: bool,
    /// Number of iterations performed
    pub iterations: usize,
}

/// Compute all-pairs shortest paths using Floyd-Warshall algorithm.
///
/// # Arguments
/// * `n_nodes` - Number of nodes in the graph (0 to n_nodes-1)
/// * `edges` - List of (from, to, weight) tuples
/// * `callback` - Optional progress callback
///
/// # Returns
/// FloydWarshallResult with distance and predecessor matrices
pub fn floyd_warshall(
    n_nodes: usize,
    edges: &[Edge],
    callback: &mut ProgressCallback,
) -> FloydWarshallResult {
    let n = n_nodes;

    // Initialize distance matrix with infinity
    let mut dist: Vec<Vec<f64>> = vec![vec![f64::INFINITY; n]; n];
    let mut pred: Vec<Vec<Option<usize>>> = vec![vec![None; n]; n];

    // Distance to self is 0
    for i in 0..n {
        dist[i][i] = 0.0;
    }

    // Add edges
    for &(u, v, w) in edges {
        if u < n && v < n && w < dist[u][v] {
            dist[u][v] = w;
            pred[u][v] = Some(u);
        }
    }

    let mut iterations = 0;
    let mut has_negative_cycle = false;

    // Main Floyd-Warshall loop
    for k in 0..n {
        // Report progress every n iterations (once per k)
        let progress = Progress::new(k, 0.0).with_evaluations(iterations);
        if callback.report(&progress).unwrap_or(false) {
            // Early termination requested
            return FloydWarshallResult {
                distances: dist,
                predecessors: pred,
                has_negative_cycle: false,
                iterations,
            };
        }

        for i in 0..n {
            for j in 0..n {
                iterations += 1;
                let via_k = dist[i][k] + dist[k][j];
                if via_k < dist[i][j] {
                    dist[i][j] = via_k;
                    pred[i][j] = pred[k][j];
                }
            }
        }
    }

    // Check for negative cycles (negative values on diagonal)
    for i in 0..n {
        if dist[i][i] < 0.0 {
            has_negative_cycle = true;
            break;
        }
    }

    FloydWarshallResult {
        distances: dist,
        predecessors: pred,
        has_negative_cycle,
        iterations,
    }
}

/// Reconstruct path from i to j using predecessor matrix.
pub fn reconstruct_path(
    pred: &[Vec<Option<usize>>],
    dist: &[Vec<f64>],
    from: usize,
    to: usize,
) -> Option<Vec<usize>> {
    // No path if distance is infinity
    if dist[from][to].is_infinite() {
        return None;
    }

    // Same node
    if from == to {
        return Some(vec![from]);
    }

    // Reconstruct backwards
    let mut path = vec![to];
    let mut current = to;

    while current != from {
        match pred[from][current] {
            Some(p) => {
                path.push(p);
                current = p;
            }
            None => return None, // No path
        }

        // Safety check for cycles in reconstruction
        if path.len() > pred.len() {
            return None;
        }
    }

    path.reverse();
    Some(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_graph() {
        let edges = vec![(0, 1, 1.0), (1, 2, 2.0), (0, 2, 5.0)];
        let result = floyd_warshall(3, &edges, &mut ProgressCallback::none());

        assert!((result.distances[0][2] - 3.0).abs() < 1e-9);
        assert!(!result.has_negative_cycle);
    }

    #[test]
    fn test_no_path() {
        let edges = vec![(0, 1, 1.0)];
        let result = floyd_warshall(3, &edges, &mut ProgressCallback::none());

        assert!(result.distances[1][0].is_infinite());
        assert!(result.distances[2][0].is_infinite());
    }

    #[test]
    fn test_negative_cycle() {
        let edges = vec![(0, 1, 1.0), (1, 2, -1.0), (2, 0, -1.0)];
        let result = floyd_warshall(3, &edges, &mut ProgressCallback::none());

        assert!(result.has_negative_cycle);
    }

    #[test]
    fn test_path_reconstruction() {
        let edges = vec![(0, 1, 1.0), (1, 2, 2.0), (0, 2, 5.0)];
        let result = floyd_warshall(3, &edges, &mut ProgressCallback::none());

        let path = reconstruct_path(&result.predecessors, &result.distances, 0, 2);
        assert_eq!(path, Some(vec![0, 1, 2]));
    }
}

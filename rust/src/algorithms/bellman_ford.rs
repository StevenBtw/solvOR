//! Bellman-Ford algorithm for shortest paths with negative weights.

/// Result of Bellman-Ford algorithm.
pub struct BellmanFordResult {
    /// Distances from source to each node (f64::INFINITY if unreachable).
    pub distances: Vec<f64>,
    /// Parent node for path reconstruction (-1 if no parent).
    pub predecessors: Vec<i64>,
    /// Whether a negative cycle was detected.
    pub has_negative_cycle: bool,
    /// Number of relaxation iterations.
    pub iterations: usize,
}

/// Runs Bellman-Ford algorithm.
///
/// # Arguments
///
/// * `n_nodes` - Number of nodes (0 to n_nodes-1)
/// * `edges` - List of (from, to, weight) tuples
/// * `source` - Starting node
///
/// # Returns
///
/// Distances, predecessors, and negative cycle flag.
pub fn bellman_ford(
    n_nodes: usize,
    edges: &[(usize, usize, f64)],
    source: usize,
) -> BellmanFordResult {
    let mut distances = vec![f64::INFINITY; n_nodes];
    let mut predecessors = vec![-1i64; n_nodes];
    let mut iterations = 0;

    distances[source] = 0.0;

    // Relax edges n-1 times
    for _ in 0..n_nodes.saturating_sub(1) {
        let mut updated = false;
        for &(u, v, w) in edges {
            iterations += 1;
            if distances[u] != f64::INFINITY && distances[u] + w < distances[v] {
                distances[v] = distances[u] + w;
                predecessors[v] = u as i64;
                updated = true;
            }
        }
        if !updated {
            break; // Early termination
        }
    }

    // Check for negative cycles
    let mut has_negative_cycle = false;
    for &(u, v, w) in edges {
        iterations += 1;
        if distances[u] != f64::INFINITY && distances[u] + w < distances[v] {
            has_negative_cycle = true;
            break;
        }
    }

    BellmanFordResult {
        distances,
        predecessors,
        has_negative_cycle,
        iterations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_graph() {
        // 0 -> 1 (weight 4)
        // 0 -> 2 (weight 5)
        // 1 -> 2 (weight -3)
        let edges = vec![(0, 1, 4.0), (0, 2, 5.0), (1, 2, -3.0)];
        let result = bellman_ford(3, &edges, 0);

        assert!(!result.has_negative_cycle);
        assert_eq!(result.distances[0], 0.0);
        assert_eq!(result.distances[1], 4.0);
        assert_eq!(result.distances[2], 1.0); // 0 -> 1 -> 2 = 4 + (-3) = 1
    }

    #[test]
    fn test_negative_cycle() {
        // Create a negative cycle: 0 -> 1 -> 2 -> 0 with total weight -1
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, -3.0)];
        let result = bellman_ford(3, &edges, 0);

        assert!(result.has_negative_cycle);
    }
}

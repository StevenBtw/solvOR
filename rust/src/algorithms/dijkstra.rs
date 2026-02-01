//! Dijkstra's algorithm for shortest paths with non-negative weights.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Min-heap wrapper for Dijkstra (BinaryHeap is max-heap by default).
#[derive(Copy, Clone, PartialEq)]
struct State {
    cost: f64,
    node: usize,
}

impl Eq for State {}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        // Flip ordering for min-heap
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Result of Dijkstra's algorithm.
pub struct DijkstraResult {
    /// Distance from source to each node (f64::INFINITY if unreachable).
    pub distances: Vec<f64>,
    /// Parent node for path reconstruction (-1 if no parent).
    pub predecessors: Vec<i64>,
    /// Number of nodes processed.
    pub iterations: usize,
    /// Whether target was reached (if specified).
    pub target_reached: bool,
    /// Path to target (if target specified and reached).
    pub path: Vec<usize>,
    /// Distance to target.
    pub target_distance: f64,
}

/// Build adjacency list from edge list.
fn build_adjacency(n_nodes: usize, edges: &[(usize, usize, f64)]) -> Vec<Vec<(usize, f64)>> {
    let mut adj = vec![Vec::new(); n_nodes];
    for &(u, v, w) in edges {
        if u < n_nodes && v < n_nodes {
            adj[u].push((v, w));
        }
    }
    adj
}

/// Runs Dijkstra's algorithm from a source node.
///
/// # Arguments
///
/// * `n_nodes` - Number of nodes (0 to n_nodes-1)
/// * `edges` - List of (from, to, weight) tuples
/// * `source` - Starting node
/// * `target` - Optional target node (early termination)
///
/// # Returns
///
/// Distances, predecessors, and path information.
pub fn dijkstra(
    n_nodes: usize,
    edges: &[(usize, usize, f64)],
    source: usize,
    target: Option<usize>,
) -> DijkstraResult {
    let adj = build_adjacency(n_nodes, edges);

    let mut distances = vec![f64::INFINITY; n_nodes];
    let mut predecessors = vec![-1i64; n_nodes];
    let mut visited = vec![false; n_nodes];
    let mut heap = BinaryHeap::new();
    let mut iterations = 0;

    distances[source] = 0.0;
    heap.push(State {
        cost: 0.0,
        node: source,
    });

    while let Some(State { cost, node }) = heap.pop() {
        if visited[node] {
            continue;
        }

        visited[node] = true;
        iterations += 1;

        // Early termination if target reached
        if let Some(t) = target {
            if node == t {
                let path = reconstruct_path(&predecessors, source, t);
                return DijkstraResult {
                    distances,
                    predecessors,
                    iterations,
                    target_reached: true,
                    path,
                    target_distance: cost,
                };
            }
        }

        // Skip if we found a better path already
        if cost > distances[node] {
            continue;
        }

        // Explore neighbors
        for &(neighbor, weight) in &adj[node] {
            if visited[neighbor] {
                continue;
            }

            let new_dist = cost + weight;
            if new_dist < distances[neighbor] {
                distances[neighbor] = new_dist;
                predecessors[neighbor] = node as i64;
                heap.push(State {
                    cost: new_dist,
                    node: neighbor,
                });
            }
        }
    }

    // Build result
    let (target_reached, path, target_distance) = if let Some(t) = target {
        if distances[t].is_finite() {
            (true, reconstruct_path(&predecessors, source, t), distances[t])
        } else {
            (false, vec![], f64::INFINITY)
        }
    } else {
        (false, vec![], f64::INFINITY)
    };

    DijkstraResult {
        distances,
        predecessors,
        iterations,
        target_reached,
        path,
        target_distance,
    }
}

/// Reconstruct path from predecessors.
fn reconstruct_path(predecessors: &[i64], source: usize, target: usize) -> Vec<usize> {
    let mut path = Vec::new();
    let mut current = target;

    while current != source {
        path.push(current);
        let pred = predecessors[current];
        if pred < 0 {
            return vec![]; // No path
        }
        current = pred as usize;
    }
    path.push(source);
    path.reverse();
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_path() {
        let edges = vec![(0, 1, 1.0), (1, 2, 2.0), (0, 2, 5.0)];
        let result = dijkstra(3, &edges, 0, Some(2));

        assert!(result.target_reached);
        assert_eq!(result.target_distance, 3.0); // 0->1->2 = 1+2 = 3
        assert_eq!(result.path, vec![0, 1, 2]);
    }

    #[test]
    fn test_no_path() {
        let edges = vec![(0, 1, 1.0)];
        let result = dijkstra(3, &edges, 0, Some(2));

        assert!(!result.target_reached);
    }
}

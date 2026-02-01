//! Breadth-first and depth-first search algorithms.

use std::collections::VecDeque;

/// Result of BFS/DFS.
pub struct SearchResult {
    /// Path from source to target (empty if not found).
    pub path: Vec<usize>,
    /// Whether target was reached.
    pub target_reached: bool,
    /// Nodes visited in order.
    pub visited_order: Vec<usize>,
    /// Number of iterations.
    pub iterations: usize,
}

/// Build adjacency list from edge list (unweighted).
fn build_adjacency(n_nodes: usize, edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let mut adj = vec![Vec::new(); n_nodes];
    for &(u, v) in edges {
        if u < n_nodes && v < n_nodes {
            adj[u].push(v);
        }
    }
    adj
}

/// Breadth-first search from source to target.
///
/// # Arguments
///
/// * `n_nodes` - Number of nodes (0 to n_nodes-1)
/// * `edges` - List of (from, to) tuples
/// * `source` - Starting node
/// * `target` - Optional target node
///
/// # Returns
///
/// Path and traversal information.
pub fn bfs(
    n_nodes: usize,
    edges: &[(usize, usize)],
    source: usize,
    target: Option<usize>,
) -> SearchResult {
    let adj = build_adjacency(n_nodes, edges);

    let mut visited = vec![false; n_nodes];
    let mut predecessors = vec![-1i64; n_nodes];
    let mut queue = VecDeque::new();
    let mut visited_order = Vec::new();
    let mut iterations = 0;

    visited[source] = true;
    queue.push_back(source);

    while let Some(node) = queue.pop_front() {
        iterations += 1;
        visited_order.push(node);

        // Check if target reached
        if let Some(t) = target {
            if node == t {
                let path = reconstruct_path(&predecessors, source, t);
                return SearchResult {
                    path,
                    target_reached: true,
                    visited_order,
                    iterations,
                };
            }
        }

        // Explore neighbors
        for &neighbor in &adj[node] {
            if !visited[neighbor] {
                visited[neighbor] = true;
                predecessors[neighbor] = node as i64;
                queue.push_back(neighbor);
            }
        }
    }

    // Target not found or no target specified
    let (path, target_reached) = if let Some(t) = target {
        if visited[t] {
            (reconstruct_path(&predecessors, source, t), true)
        } else {
            (vec![], false)
        }
    } else {
        (vec![], false)
    };

    SearchResult {
        path,
        target_reached,
        visited_order,
        iterations,
    }
}

/// Depth-first search from source to target.
pub fn dfs(
    n_nodes: usize,
    edges: &[(usize, usize)],
    source: usize,
    target: Option<usize>,
) -> SearchResult {
    let adj = build_adjacency(n_nodes, edges);

    let mut visited = vec![false; n_nodes];
    let mut predecessors = vec![-1i64; n_nodes];
    let mut stack = vec![source];
    let mut visited_order = Vec::new();
    let mut iterations = 0;

    while let Some(node) = stack.pop() {
        if visited[node] {
            continue;
        }

        visited[node] = true;
        iterations += 1;
        visited_order.push(node);

        // Check if target reached
        if let Some(t) = target {
            if node == t {
                let path = reconstruct_path(&predecessors, source, t);
                return SearchResult {
                    path,
                    target_reached: true,
                    visited_order,
                    iterations,
                };
            }
        }

        // Explore neighbors (reverse order for consistent traversal)
        for &neighbor in adj[node].iter().rev() {
            if !visited[neighbor] {
                predecessors[neighbor] = node as i64;
                stack.push(neighbor);
            }
        }
    }

    SearchResult {
        path: vec![],
        target_reached: false,
        visited_order,
        iterations,
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
            return vec![];
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
    fn test_bfs_simple() {
        let edges = vec![(0, 1), (0, 2), (1, 3), (2, 3)];
        let result = bfs(4, &edges, 0, Some(3));

        assert!(result.target_reached);
        assert_eq!(result.path.len(), 3); // 0 -> 1 -> 3 or 0 -> 2 -> 3
    }

    #[test]
    fn test_dfs_simple() {
        let edges = vec![(0, 1), (0, 2), (1, 3), (2, 3)];
        let result = dfs(4, &edges, 0, Some(3));

        assert!(result.target_reached);
    }
}

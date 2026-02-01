//! PageRank algorithm for node importance scoring.

/// Result of PageRank algorithm.
pub struct PageRankResult {
    /// PageRank scores for each node.
    pub scores: Vec<f64>,
    /// Number of iterations until convergence.
    pub iterations: usize,
    /// Whether the algorithm converged.
    pub converged: bool,
}

/// Compute PageRank scores.
///
/// # Arguments
///
/// * `n_nodes` - Number of nodes (0 to n_nodes-1)
/// * `edges` - List of (from, to) tuples (directed edges)
/// * `damping` - Damping factor (typically 0.85)
/// * `max_iter` - Maximum iterations
/// * `tol` - Convergence tolerance
///
/// # Returns
///
/// PageRank scores for each node.
pub fn pagerank(
    n_nodes: usize,
    edges: &[(usize, usize)],
    damping: f64,
    max_iter: usize,
    tol: f64,
) -> PageRankResult {
    if n_nodes == 0 {
        return PageRankResult {
            scores: vec![],
            iterations: 0,
            converged: true,
        };
    }

    // Build incoming edges and outgoing counts
    let mut incoming: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
    let mut outgoing_count: Vec<usize> = vec![0; n_nodes];

    for &(u, v) in edges {
        if u < n_nodes && v < n_nodes {
            incoming[v].push(u);
            outgoing_count[u] += 1;
        }
    }

    // Initialize scores uniformly
    let initial = 1.0 / n_nodes as f64;
    let mut scores = vec![initial; n_nodes];
    let mut new_scores = vec![0.0; n_nodes];

    let base = (1.0 - damping) / n_nodes as f64;
    let mut converged = false;

    for iteration in 0..max_iter {
        // Compute new scores
        for i in 0..n_nodes {
            let mut sum = 0.0;
            for &j in &incoming[i] {
                if outgoing_count[j] > 0 {
                    sum += scores[j] / outgoing_count[j] as f64;
                }
            }
            new_scores[i] = base + damping * sum;
        }

        // Handle dangling nodes (no outgoing edges)
        let dangling_sum: f64 = scores
            .iter()
            .enumerate()
            .filter(|&(i, _)| outgoing_count[i] == 0)
            .map(|(_, &s)| s)
            .sum();

        let dangling_contrib = damping * dangling_sum / n_nodes as f64;
        for score in &mut new_scores {
            *score += dangling_contrib;
        }

        // Check convergence
        let diff: f64 = scores
            .iter()
            .zip(new_scores.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        std::mem::swap(&mut scores, &mut new_scores);

        if diff < tol {
            return PageRankResult {
                scores,
                iterations: iteration + 1,
                converged: true,
            };
        }
    }

    PageRankResult {
        scores,
        iterations: max_iter,
        converged,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_pagerank() {
        // Simple chain: 0 -> 1 -> 2
        let edges = vec![(0, 1), (1, 2)];
        let result = pagerank(3, &edges, 0.85, 100, 1e-6);

        assert!(result.converged);
        // Node 2 should have highest score (receives from 1)
        assert!(result.scores[2] > result.scores[0]);
    }

    #[test]
    fn test_cycle() {
        // Cycle: 0 -> 1 -> 2 -> 0
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let result = pagerank(3, &edges, 0.85, 100, 1e-6);

        assert!(result.converged);
        // All nodes should have equal scores in a cycle
        let diff = (result.scores[0] - result.scores[1]).abs();
        assert!(diff < 0.01);
    }
}

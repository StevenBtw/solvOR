//! PageRank algorithm for node importance scoring.

/// Result of PageRank algorithm.
pub struct PageRankResult {
    /// PageRank scores for each node.
    pub scores: Vec<f64>,
    /// Number of iterations until convergence.
    pub iterations: usize,
    /// Whether the algorithm converged.
    pub converged: bool,
    /// Largest absolute score change in the last iteration.
    pub residual: f64,
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
            residual: 0.0,
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
    let converged = false;
    let mut residual = 0.0;

    // Same arithmetic as the Python backend, operation for operation, so both
    // backends return bit-identical scores: multiply by 1/out_degree, sum with
    // CPython's compensated sum, add terms in the same order.
    let inv_out: Vec<f64> = outgoing_count
        .iter()
        .map(|&d| if d > 0 { 1.0 / d as f64 } else { 0.0 })
        .collect();
    let dangling: Vec<usize> = (0..n_nodes).filter(|&u| outgoing_count[u] == 0).collect();
    let mut share = vec![0.0; n_nodes];

    for iteration in 0..max_iter {
        for (s, (&score, &inv)) in share.iter_mut().zip(scores.iter().zip(inv_out.iter())) {
            *s = score * inv;
        }

        // Dangling nodes (no outgoing edges) spread their rank over all nodes
        let dangling_sum = python_sum(dangling.iter().map(|&u| scores[u]));
        let dangling_contrib = damping * dangling_sum / n_nodes as f64;

        for (i, new_score) in new_scores.iter_mut().enumerate() {
            let rank_sum = python_sum(incoming[i].iter().map(|&u| share[u]));
            *new_score = base + damping * rank_sum + dangling_contrib;
        }

        // Converged when no score moved by more than tol (max-norm, same as the Python backend)
        let diff = scores
            .iter()
            .zip(new_scores.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        residual = diff;

        std::mem::swap(&mut scores, &mut new_scores);

        if diff < tol {
            return PageRankResult {
                scores,
                iterations: iteration + 1,
                converged: true,
                residual,
            };
        }
    }

    PageRankResult {
        scores,
        iterations: max_iter,
        converged,
        residual,
    }
}

/// Float sum exactly as CPython's built-in `sum()` computes it (3.12+): Neumaier's
/// compensated summation, compensation added once at the end if finite and nonzero.
/// (Same algorithm in CPython 3.12, 3.13, 3.14 and main; 3.14 only moved it into `cs_add`.)
fn python_sum(values: impl Iterator<Item = f64>) -> f64 {
    let mut total = 0.0_f64;
    let mut compensation = 0.0_f64;
    for x in values {
        let t = total + x;
        if total.abs() >= x.abs() {
            compensation += (total - t) + x;
        } else {
            compensation += (x - t) + total;
        }
        total = t;
    }
    if compensation != 0.0 && compensation.is_finite() {
        total += compensation;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    // Expected values are what CPython 3.14's sum() returns for the same lists
    #[test]
    fn test_python_sum_compensation() {
        assert_eq!(python_sum([1e100, 1.0, -1e100].into_iter()), 1.0);
        assert_eq!(python_sum([0.1; 10].into_iter()), 1.0);
        let zero = python_sum([-0.0, -0.0].into_iter());
        assert!(zero == 0.0 && zero.is_sign_positive());
    }

    #[test]
    fn test_python_sum_non_finite() {
        assert_eq!(
            python_sum([f64::INFINITY, 1e100, -1e100].into_iter()),
            f64::INFINITY
        );
        assert_eq!(
            python_sum([1e308, 1e308, -1e308].into_iter()),
            f64::INFINITY
        );
        assert!(python_sum([1.0, f64::NAN, 2.0].into_iter()).is_nan());
        assert!(python_sum([f64::INFINITY, f64::NEG_INFINITY].into_iter()).is_nan());
    }

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

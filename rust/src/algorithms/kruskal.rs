//! Kruskal's algorithm for minimum spanning tree.

/// Union-Find data structure for Kruskal's algorithm.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]); // Path compression
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) -> bool {
        let root_x = self.find(x);
        let root_y = self.find(y);

        if root_x == root_y {
            return false; // Already in same set
        }

        // Union by rank
        match self.rank[root_x].cmp(&self.rank[root_y]) {
            std::cmp::Ordering::Less => self.parent[root_x] = root_y,
            std::cmp::Ordering::Greater => self.parent[root_y] = root_x,
            std::cmp::Ordering::Equal => {
                self.parent[root_y] = root_x;
                self.rank[root_x] += 1;
            }
        }
        true
    }
}

/// Result of Kruskal's algorithm.
pub struct KruskalResult {
    /// MST edges as (from, to, weight) tuples.
    pub mst_edges: Vec<(usize, usize, f64)>,
    /// Total weight of the MST.
    pub total_weight: f64,
    /// Number of iterations (edges considered).
    pub iterations: usize,
    /// Whether the graph is connected (MST spans all nodes).
    pub is_connected: bool,
}

/// Runs Kruskal's algorithm for minimum spanning tree.
///
/// # Arguments
///
/// * `n_nodes` - Number of nodes (0 to n_nodes-1)
/// * `edges` - List of (from, to, weight) tuples
///
/// # Returns
///
/// MST edges and total weight.
pub fn kruskal(n_nodes: usize, edges: &[(usize, usize, f64)]) -> KruskalResult {
    if n_nodes == 0 {
        return KruskalResult {
            mst_edges: vec![],
            total_weight: 0.0,
            iterations: 0,
            is_connected: true,
        };
    }

    // Sort edges by weight
    let mut sorted_edges: Vec<(usize, usize, f64)> = edges.to_vec();
    sorted_edges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

    let mut uf = UnionFind::new(n_nodes);
    let mut mst_edges = Vec::with_capacity(n_nodes - 1);
    let mut total_weight = 0.0;
    let mut iterations = 0;

    for (u, v, w) in sorted_edges {
        iterations += 1;
        if uf.union(u, v) {
            mst_edges.push((u, v, w));
            total_weight += w;

            if mst_edges.len() == n_nodes - 1 {
                break; // MST complete
            }
        }
    }

    let is_connected = mst_edges.len() == n_nodes - 1;

    KruskalResult {
        mst_edges,
        total_weight,
        iterations,
        is_connected,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_mst() {
        // Triangle graph
        let edges = vec![(0, 1, 1.0), (1, 2, 2.0), (0, 2, 3.0)];
        let result = kruskal(3, &edges);

        assert!(result.is_connected);
        assert_eq!(result.mst_edges.len(), 2);
        assert_eq!(result.total_weight, 3.0); // 1 + 2 = 3
    }

    #[test]
    fn test_disconnected() {
        let edges = vec![(0, 1, 1.0)]; // Only connects 0 and 1
        let result = kruskal(3, &edges);

        assert!(!result.is_connected);
        assert_eq!(result.mst_edges.len(), 1);
    }
}

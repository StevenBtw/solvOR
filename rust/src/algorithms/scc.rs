//! Strongly connected components and topological sort.

/// Result of SCC algorithm.
pub struct SCCResult {
    /// List of strongly connected components (each is a list of node indices).
    pub components: Vec<Vec<usize>>,
    /// Number of components.
    pub n_components: usize,
}

/// Result of topological sort.
pub struct TopologicalResult {
    /// Nodes in topological order (None if cycle exists).
    pub order: Option<Vec<usize>>,
    /// Whether the graph is acyclic.
    pub is_acyclic: bool,
    /// Number of iterations.
    pub iterations: usize,
}

/// Build adjacency list from edge list.
fn build_adjacency(n_nodes: usize, edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let mut adj = vec![Vec::new(); n_nodes];
    for &(u, v) in edges {
        if u < n_nodes && v < n_nodes {
            adj[u].push(v);
        }
    }
    adj
}

/// Compute strongly connected components using Tarjan's algorithm.
///
/// Iterative DFS with an explicit frame stack, so graph depth is not limited
/// by the native stack. Components come out in the same order as the
/// recursive formulation (reverse topological order).
pub fn strongly_connected_components(n_nodes: usize, edges: &[(usize, usize)]) -> SCCResult {
    let adj = build_adjacency(n_nodes, edges);

    let mut index_counter = 0usize;
    let mut indices: Vec<Option<usize>> = vec![None; n_nodes];
    let mut lowlinks = vec![0usize; n_nodes];
    let mut on_stack = vec![false; n_nodes];
    let mut stack: Vec<usize> = Vec::new();
    let mut components: Vec<Vec<usize>> = Vec::new();
    // DFS frames: (node, position of the next neighbor to visit in adj[node])
    let mut frames: Vec<(usize, usize)> = Vec::new();

    for root in 0..n_nodes {
        if indices[root].is_some() {
            continue;
        }
        indices[root] = Some(index_counter);
        lowlinks[root] = index_counter;
        index_counter += 1;
        stack.push(root);
        on_stack[root] = true;
        frames.push((root, 0));

        while let Some(frame) = frames.last_mut() {
            let v = frame.0;
            if frame.1 < adj[v].len() {
                let w = adj[v][frame.1];
                frame.1 += 1;
                match indices[w] {
                    None => {
                        indices[w] = Some(index_counter);
                        lowlinks[w] = index_counter;
                        index_counter += 1;
                        stack.push(w);
                        on_stack[w] = true;
                        frames.push((w, 0));
                    }
                    Some(index_w) if on_stack[w] => {
                        lowlinks[v] = lowlinks[v].min(index_w);
                    }
                    Some(_) => {}
                }
                continue;
            }

            // All neighbors of v are done: propagate its lowlink to the parent.
            frames.pop();
            if let Some(&(parent, _)) = frames.last() {
                lowlinks[parent] = lowlinks[parent].min(lowlinks[v]);
            }

            // If v is a root node, pop the stack and generate an SCC
            if indices[v] == Some(lowlinks[v]) {
                let mut component = Vec::new();
                while let Some(w) = stack.pop() {
                    on_stack[w] = false;
                    component.push(w);
                    if w == v {
                        break;
                    }
                }
                components.push(component);
            }
        }
    }

    let n_components = components.len();
    SCCResult {
        components,
        n_components,
    }
}

/// Compute topological ordering using Kahn's algorithm.
pub fn topological_sort(n_nodes: usize, edges: &[(usize, usize)]) -> TopologicalResult {
    let adj = build_adjacency(n_nodes, edges);

    // Compute in-degrees
    let mut in_degree = vec![0usize; n_nodes];
    for neighbors in &adj {
        for &v in neighbors {
            in_degree[v] += 1;
        }
    }

    // Initialize queue with nodes having in-degree 0
    let mut queue: Vec<usize> = (0..n_nodes).filter(|&v| in_degree[v] == 0).collect();

    let mut order = Vec::with_capacity(n_nodes);
    let mut iterations = 0;

    while let Some(u) = queue.pop() {
        iterations += 1;
        order.push(u);

        for &v in &adj[u] {
            in_degree[v] -= 1;
            if in_degree[v] == 0 {
                queue.push(v);
            }
        }
    }

    if order.len() == n_nodes {
        TopologicalResult {
            order: Some(order),
            is_acyclic: true,
            iterations,
        }
    } else {
        TopologicalResult {
            order: None,
            is_acyclic: false,
            iterations,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scc_simple() {
        // Graph with 2 SCCs: {0, 1, 2} and {3}
        let edges = vec![(0, 1), (1, 2), (2, 0), (2, 3)];
        let result = strongly_connected_components(4, &edges);

        assert_eq!(result.n_components, 2);
    }

    #[test]
    fn test_scc_order_matches_recursive_tarjan() {
        // Same graph as test_scc_simple: {3} is emitted first (reverse topological order)
        let edges = vec![(0, 1), (1, 2), (2, 0), (2, 3)];
        let result = strongly_connected_components(4, &edges);

        assert_eq!(result.components, vec![vec![3], vec![2, 1, 0]]);
    }

    #[test]
    fn test_scc_deep_chain_does_not_overflow() {
        // Test threads have a 2 MiB stack; the recursive version overflowed here.
        let n = 1_000_000;
        let edges: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
        let result = strongly_connected_components(n, &edges);

        assert_eq!(result.n_components, n);
        assert_eq!(result.components[0], vec![n - 1]);
        assert_eq!(result.components[n - 1], vec![0]);
    }

    #[test]
    fn test_scc_deep_cycle_is_one_component() {
        let n = 1_000_000;
        let edges: Vec<(usize, usize)> = (0..n).map(|i| (i, (i + 1) % n)).collect();
        let result = strongly_connected_components(n, &edges);

        assert_eq!(result.n_components, 1);
        assert_eq!(result.components[0].len(), n);
    }

    #[test]
    fn test_topological_sort_dag() {
        // DAG: 0 -> 1 -> 2, 0 -> 2
        let edges = vec![(0, 1), (1, 2), (0, 2)];
        let result = topological_sort(3, &edges);

        assert!(result.is_acyclic);
        let order = result.order.unwrap();
        assert_eq!(order[0], 0); // 0 must come first
    }

    #[test]
    fn test_topological_sort_cycle() {
        // Cycle: 0 -> 1 -> 2 -> 0
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let result = topological_sort(3, &edges);

        assert!(!result.is_acyclic);
        assert!(result.order.is_none());
    }
}

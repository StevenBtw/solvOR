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
pub fn strongly_connected_components(n_nodes: usize, edges: &[(usize, usize)]) -> SCCResult {
    let adj = build_adjacency(n_nodes, edges);

    let mut index_counter = 0usize;
    let mut indices = vec![None; n_nodes];
    let mut lowlinks = vec![0usize; n_nodes];
    let mut on_stack = vec![false; n_nodes];
    let mut stack = Vec::new();
    let mut components = Vec::new();

    fn strongconnect(
        v: usize,
        adj: &[Vec<usize>],
        index_counter: &mut usize,
        indices: &mut [Option<usize>],
        lowlinks: &mut [usize],
        on_stack: &mut [bool],
        stack: &mut Vec<usize>,
        components: &mut Vec<Vec<usize>>,
    ) {
        indices[v] = Some(*index_counter);
        lowlinks[v] = *index_counter;
        *index_counter += 1;
        stack.push(v);
        on_stack[v] = true;

        for &w in &adj[v] {
            if indices[w].is_none() {
                strongconnect(
                    w,
                    adj,
                    index_counter,
                    indices,
                    lowlinks,
                    on_stack,
                    stack,
                    components,
                );
                lowlinks[v] = lowlinks[v].min(lowlinks[w]);
            } else if on_stack[w] {
                lowlinks[v] = lowlinks[v].min(indices[w].unwrap());
            }
        }

        // If v is a root node, pop the stack and generate an SCC
        if lowlinks[v] == indices[v].unwrap() {
            let mut component = Vec::new();
            loop {
                let w = stack.pop().unwrap();
                on_stack[w] = false;
                component.push(w);
                if w == v {
                    break;
                }
            }
            components.push(component);
        }
    }

    for v in 0..n_nodes {
        if indices[v].is_none() {
            strongconnect(
                v,
                &adj,
                &mut index_counter,
                &mut indices,
                &mut lowlinks,
                &mut on_stack,
                &mut stack,
                &mut components,
            );
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

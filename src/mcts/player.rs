use crate::mcts::node::MCTSNode;

pub struct MCTSPlayer {
    name: String,
    iterations: usize,
    exploration_constant: f64,
    max_time_ms: Option<u64>,
    use_heuristics: bool,
}

impl MCTSPlayer {
    /// Perform selection phase: traverse from root to leaf (read-only to get path)
    fn select_path(root: &MCTSNode, exploration_constant: f64) -> Vec<usize> {
        let mut path = Vec::new();
        let mut current = root;
        
        // Traverse until we find an unexpanded or terminal node
        while current.is_expanded() && !current.is_terminal() && current.has_children() {
            let child_idx = current.select_child(exploration_constant);
            path.push(child_idx);
            // Get child reference (immutable)
            current = current.get_child(child_idx)
                .expect("Selected child index should be valid");
        }
        
        path
    }
    
    /// Get mutable reference to node at path
    fn get_node_mut_at_path<'a>(root: &'a mut MCTSNode, path: &[usize]) -> &'a mut MCTSNode {
        let mut current = root;
        for &idx in path {
            current = current.get_child_mut(idx)
                .expect("Path index should be valid");
        }
        current
    }
}


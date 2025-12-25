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
    /// Returns a path (vector of child indices) from root to the selected leaf node.
    /// The leaf node is either:
    /// - Not yet expanded (needs expansion)
    /// - Terminal (game over)
    /// - Has no children (no valid moves available)
    fn select_path(root: &MCTSNode, exploration_constant: f64) -> Vec<usize> {
        let mut path = Vec::new();
        let mut current = root;
        
        // Keep traversing while:
        // 1. Node is expanded (has children created)
        // 2. Node is not terminal (game still ongoing)
        // 3. Node has children (has valid moves)
        while current.is_expanded() && !current.is_terminal() && current.has_children() {
            let child_idx = current.select_child(exploration_constant);
            
            // Safety check: ensure the selected index is valid
            if child_idx >= current.num_children() {
                // This shouldn't happen, but if it does, stop traversal
                break;
            }
            
            path.push(child_idx);
            
            // Move to child node (immutable reference)
            match current.get_child(child_idx) {
                Some(child) => current = child,
                None => {
                    // Invalid child index, stop traversal
                    path.pop(); // Remove the invalid index
                    break;
                }
            }
        }
        
        path
    }
    
    /// Get mutable reference to node at the given path
    /// The path is a vector of child indices from root to the target node.
    /// Returns a mutable reference to the node at the end of the path.
    fn get_node_mut_at_path<'a>(root: &'a mut MCTSNode, path: &[usize]) -> &'a mut MCTSNode {
        let mut current = root;
        for &idx in path {
            // Safety: verify index is valid before accessing
            if idx >= current.num_children() {
                panic!("Invalid path index {} for node with {} children", idx, current.num_children());
            }
            current = current.get_child_mut(idx)
                .expect("Path index should be valid");
        }
        current
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Game;
    
    #[test]
    fn test_select_path_from_unexpanded_root() {
        let game = Game::new();
        let root = MCTSNode::new(game);
        
        // Root is not expanded, so path should be empty (root itself is the leaf)
        let path = MCTSPlayer::select_path(&root, 1.414);
        assert_eq!(path.len(), 0);
    }
    
    #[test]
    fn test_select_path_from_expanded_root() {
        let game = Game::new();
        let mut root = MCTSNode::new(game);
        root.expand();
        
        // Root is expanded, so we should traverse to a child
        let path = MCTSPlayer::select_path(&root, 1.414);
        
        // Path should have at least one element (to a child)
        // or be empty if no children were created (edge case)
        if root.has_children() {
            assert!(path.len() <= 1); // Should select a child (path length 1)
            // Verify the path index is valid
            if !path.is_empty() {
                assert!(path[0] < root.num_children());
            }
        }
    }
    
    #[test]
    fn test_select_path_stops_at_terminal() {
        // Create a game and play it to completion
        let game = Game::new();
        
        // Create root and expand
        let mut root = MCTSNode::new(game);
        root.expand();
        
        if root.has_children() {
            // Select path should work even if we encounter terminal nodes
            let path = MCTSPlayer::select_path(&root, 1.414);
            // Path should be valid (not exceed number of children)
            assert!(path.len() <= root.num_children());
            
            // Verify we can access the node at the path
            let leaf = MCTSPlayer::get_node_mut_at_path(&mut root, &path);
            // Leaf should be either terminal, unexpanded, or have no children
            assert!(leaf.is_terminal() || !leaf.is_expanded() || !leaf.has_children());
        }
    }
    
    #[test]
    fn test_get_node_mut_at_path_empty_path() {
        let game = Game::new();
        let mut root = MCTSNode::new(game);
        
        // Empty path should return root
        let node = MCTSPlayer::get_node_mut_at_path(&mut root, &[]);
        assert_eq!(node.visits(), 0);
        assert!(!node.is_expanded());
    }
    
    #[test]
    fn test_get_node_mut_at_path_single_level() {
        let game = Game::new();
        let mut root = MCTSNode::new(game);
        root.expand();
        
        if root.has_children() {
            let path = vec![0]; // First child
            let child = MCTSPlayer::get_node_mut_at_path(&mut root, &path);
            
            // Verify we got the correct child
            assert!(child.move_from_parent().is_some());
            assert_eq!(child.visits(), 0);
            
            // Verify we have mutable access by expanding the child
            // (this requires mutable access to work)
            child.expand();
            // If we got here without error, we have mutable access
        }
    }
    
    #[test]
    fn test_get_node_mut_at_path_multi_level() {
        let game = Game::new();
        let mut root = MCTSNode::new(game);
        root.expand();
        
        if root.has_children() {
            // Get first child and expand it
            let first_child_path = vec![0];
            let first_child = MCTSPlayer::get_node_mut_at_path(&mut root, &first_child_path);
            first_child.expand();
            
            if first_child.has_children() {
                // Now traverse to a grandchild
                let grandchild_path = vec![0, 0];
                let grandchild = MCTSPlayer::get_node_mut_at_path(&mut root, &grandchild_path);
                
                // Verify we got the grandchild
                assert!(grandchild.move_from_parent().is_some());
                
                // Verify mutable access by expanding the grandchild
                grandchild.expand();
                // If we got here without error, we have mutable access
            }
        }
    }
    
    #[test]
    fn test_select_path_and_get_node_consistency() {
        let game = Game::new();
        let mut root = MCTSNode::new(game);
        root.expand();
        
        if root.has_children() {
            // Select a path
            let path = MCTSPlayer::select_path(&root, 1.414);
            
            // Get the node at that path
            let leaf = MCTSPlayer::get_node_mut_at_path(&mut root, &path);
            
            // The leaf should either be:
            // - Not expanded (needs expansion)
            // - Terminal (game over)
            // - Has no children
            assert!(!leaf.is_expanded() || leaf.is_terminal() || !leaf.has_children());
        }
    }
    
    #[test]
    fn test_select_path_handles_no_children() {
        // Test that selection stops correctly when a node has no children
        let game = Game::new();
        let root = MCTSNode::new(game);
        
        // Don't expand - root has no children
        let path = MCTSPlayer::select_path(&root, 1.414);
        assert_eq!(path.len(), 0); // Should return empty path (root is leaf)
    }
}


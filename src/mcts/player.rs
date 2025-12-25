use crate::mcts::node::MCTSNode;
use crate::game::{Game, Player, GameState};
use rand::seq::SliceRandom;
use rand::thread_rng;

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
    
    /// Simulate a random game from given state
    /// Returns: 1.0 if root_player wins, 0.0 if loses, 0.5 if draw
    fn simulate(&self, game: &mut Game, root_player: Player) -> f64 {
        let mut rng = thread_rng();
        
        // Play random moves until game ends
        while matches!(game.get_game_state(), GameState::Playing) {
            let valid_moves = game.get_valid_moves();
            
            if valid_moves.is_empty() {
                // No valid moves, skip turn
                game.skip_turn().ok();
                continue;
            }
            
            // Choose random move
            if let Some(&move_pos) = valid_moves.choose(&mut rng) {
                game.make_move(move_pos).ok();
            }
        }
        
        // Determine result from root player's perspective
        match game.get_game_state() {
            GameState::GameOver { winner } => {
                match winner {
                    Some(player) if player == root_player => 1.0,
                    Some(_) => 0.0,
                    None => 0.5,
                }
            }
            _ => 0.5, // Should not happen
        }
    }
    
    /// Backpropagate result up the tree
    /// Updates all nodes along the path from leaf to root
    fn backpropagate(root: &mut MCTSNode, path: &[usize], result: f64, root_player: Player) {
        // First, update root (from root player's perspective)
        root.update_statistics(result);
        
        // Then update nodes along the path
        // We need to work backwards to get the correct perspective for each node
        for i in 0..path.len() {
            // Path up to and including this node (inclusive)
            let node_path = &path[..=i];
            let node = Self::get_node_mut_at_path(root, node_path);
            let node_player = node.current_player();
            
            // Result from this node's player perspective
            // If node_player is the same as root_player, use result as-is
            // If node_player is opposite, flip the result (1.0 - result)
            let node_result = if node_player == root_player {
                result
            } else {
                1.0 - result
            };
            
            node.update_statistics(node_result);
        }
    }
    
    /// Perform one MCTS iteration
    /// Combines selection, expansion, simulation, and backpropagation
    fn mcts_iteration(&self, root: &mut MCTSNode, root_player: Player) {
        // 1. Selection: Find path to leaf (read-only)
        let path = Self::select_path(root, self.exploration_constant);
        
        // 2. Get the leaf node (mutable)
        let leaf = Self::get_node_mut_at_path(root, &path);
        
        // 3. Expansion: Expand if not terminal and not expanded
        if !leaf.is_terminal() && !leaf.is_expanded() {
            leaf.expand();
        }
        
        // 4. Simulation: Play random game from leaf state
        // Clone the game state before simulation (simulate needs to mutate it)
        let mut sim_game = leaf.game_state().clone();
        let result = self.simulate(&mut sim_game, root_player);
        
        // 5. Backpropagation: Update statistics along path
        Self::backpropagate(root, &path, result, root_player);
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
    
    #[test]
    fn test_simulate_always_terminates() {
        let player = MCTSPlayer {
            name: "Test".to_string(),
            iterations: 100,
            exploration_constant: 1.414,
            max_time_ms: None,
            use_heuristics: false,
        };
        
        let mut game = Game::new();
        let root_player = game.current_player();
        
        // Run simulation - should always reach terminal state
        let result = player.simulate(&mut game, root_player);
        
        // Game should be over
        assert!(matches!(game.get_game_state(), GameState::GameOver { .. }));
        
        // Result should be valid (0.0, 0.5, or 1.0)
        assert!(result >= 0.0 && result <= 1.0);
    }
    
    #[test]
    fn test_backpropagate_updates_statistics() {
        let game = Game::new();
        let mut root = MCTSNode::new(game);
        let root_player = root.current_player();
        
        // Expand root to create children
        root.expand();
        
        if root.has_children() {
            // Select a path to a child
            let path = MCTSPlayer::select_path(&root, 1.414);
            
            // Initial visits should be 0
            assert_eq!(root.visits(), 0);
            
            // Backpropagate a win (1.0) for root player
            MCTSPlayer::backpropagate(&mut root, &path, 1.0, root_player);
            
            // Root should now have 1 visit
            assert_eq!(root.visits(), 1);
            
            // Root value should be 1.0 (win from root player's perspective)
            assert_eq!(root.average_value(), 1.0);
            
            // If path has nodes, they should also be updated
            if !path.is_empty() {
                // Get the leaf node
                let leaf = MCTSPlayer::get_node_mut_at_path(&mut root, &path);
                assert_eq!(leaf.visits(), 1);
            }
        }
    }
    
    #[test]
    fn test_mcts_iteration_completes() {
        let player = MCTSPlayer {
            name: "Test".to_string(),
            iterations: 100,
            exploration_constant: 1.414,
            max_time_ms: None,
            use_heuristics: false,
        };
        
        let game = Game::new();
        let mut root = MCTSNode::new(game);
        let root_player = root.current_player();
        
        // Run one iteration
        player.mcts_iteration(&mut root, root_player);
        
        // Root should have been visited
        assert_eq!(root.visits(), 1);
        
        // Root should have been expanded (unless it's terminal)
        if !root.is_terminal() {
            assert!(root.is_expanded());
            assert!(root.has_children());
        }
    }
    
    #[test]
    fn test_backpropagate_flips_result_correctly() {
        let game = Game::new();
        let mut root = MCTSNode::new(game);
        let root_player = root.current_player();
        
        // Expand root to create children
        root.expand();
        
        if root.has_children() {
            // Select a path to a child
            let path = MCTSPlayer::select_path(&root, 1.414);
            
            // Get the child node to check its player
            let child = MCTSPlayer::get_node_mut_at_path(&mut root, &path);
            let child_player = child.current_player();
            
            // Backpropagate a loss (0.0) for root player
            // For root: 0.0 (loss from root player's perspective)
            // For child: 1.0 - 0.0 = 1.0 (win from child player's perspective)
            MCTSPlayer::backpropagate(&mut root, &path, 0.0, root_player);
            
            // Root should have 0.0 average value (loss)
            assert_eq!(root.average_value(), 0.0);
            
            // If child player is different from root, child should have 1.0
            if !path.is_empty() && child_player != root_player {
                let child_after = MCTSPlayer::get_node_mut_at_path(&mut root, &path);
                assert_eq!(child_after.average_value(), 1.0);
            }
        }
    }
}

